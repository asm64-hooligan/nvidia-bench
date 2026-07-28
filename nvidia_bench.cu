/*
 * Uses vectorized 128-bit memory transactions (float4 / ulong2) to achieve
 * peak hardware memory bandwidth.
 *
 * Build: nvcc -O3 -o nvidia_bench nvidia_bench.cu -lnvidia-ml \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86 \
          -gencode arch=compute_89,code=sm_89 \
          -gencode arch=compute_90,code=sm_90 \
          -gencode arch=compute_100,code=sm_100 \
          -gencode arch=compute_120,code=sm_120 \
          -gencode arch=compute_120,code=compute_120
 
 * Run:   ./nvidia_bench [gpu_id] [iterations]
 *        iterations <= 0 (default) auto-sizes each timed run to ~3 s.
 *        Note: that exceeds the ~2 s display watchdog — on a display-attached
 *        GPU pass a small explicit iteration count instead.
 *
 * Besides latency and bandwidth, measures dense tensor-core throughput via
 * WMMA: TF32 (FP32 inputs), BF16, INT8. On Hopper/Blackwell datacenter parts
 * WMMA can understate absolute peak (wgmma/tcgen05 paths are not used).
 */

#include <cuda_runtime.h>
#include <nvml.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ─── Vectorized float4 memory access kernels (128-bit / 16 bytes per thread) ───

// Read-Only: read float4 from global memory and reduce into scalar register
__global__ void bench_hbm_read(const float4 * __restrict__ A, float * __restrict__ dummy_out, size_t N_vec, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (int iter = 0; iter < iters; iter++) {
        for (size_t i = idx; i < N_vec; i += stride) {
            float4 v = A[i];
            acc += v.x + v.y + v.z + v.w;
        }
    }
    // A holds only positive values, so acc can never be -1.0f — but the compiler
    // can't prove that, which keeps every thread's loads live
    if (acc == -1.0f) dummy_out[0] = acc;
}

// Write-Only: store float4 values to global memory
__global__ void bench_hbm_write(float4 * __restrict__ A, float4 val, size_t N_vec, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < iters; iter++) {
        float4 v = val;
        v.x += (float)iter;  // distinct value each pass so passes can't be merged
        for (size_t i = idx; i < N_vec; i += stride) {
            A[i] = v;
        }
    }
}

// Copy (D2D): dst[i] = src[i] (1 float4 read + 1 float4 write = 32 bytes traffic per element).
// Direction alternates each pass, so every store feeds the next pass's loads and
// no pass is a repeat of the previous one.
__global__ void bench_hbm_copy(float4 * __restrict__ A, float4 * __restrict__ B, size_t N_vec, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < iters; iter++) {
        const float4 *src = (iter & 1) ? B : A;
        float4 *dst = (iter & 1) ? A : B;
        for (size_t i = idx; i < N_vec; i += stride) {
            dst[i] = src[i];
        }
    }
}

// STREAM Triad: dst[i] = src[i] + alpha * B[i] (2 float4 reads + 1 float4 write = 48 bytes traffic per element).
// A and C swap roles each pass for the same reason as in bench_hbm_copy.
__global__ void bench_hbm_triad(float4 * __restrict__ A, const float4 * __restrict__ B, float4 * __restrict__ C, float alpha, size_t N_vec, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int iter = 0; iter < iters; iter++) {
        const float4 *src = (iter & 1) ? C : A;
        float4 *dst = (iter & 1) ? A : C;
        for (size_t i = idx; i < N_vec; i += stride) {
            float4 a = src[i];
            float4 b = B[i];
            float4 c;
            c.x = a.x + alpha * b.x;
            c.y = a.y + alpha * b.y;
            c.z = a.z + alpha * b.z;
            c.w = a.w + alpha * b.w;
            dst[i] = c;
        }
    }
}

// ─── Memory latency kernel (Pointer Chasing / Stride) ───
__global__ void bench_hbm_latency(const uint32_t * __restrict__ chain, uint32_t * __restrict__ out_idx, int steps) {
    uint32_t idx = 0;
    for (int i = 0; i < steps; i++) {
        idx = chain[idx];
    }
    out_idx[0] = idx;
}

// Scatter the permutation into the padded chain: one slot per 128-byte cache line
// (stride 32 uint32_t), indices pre-scaled by 32 so the chase loop needs no math.
__global__ void build_chain(const uint32_t * __restrict__ perm, uint32_t * __restrict__ chain, size_t n_slots) {
    size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; k < n_slots; k += stride) {
        chain[k * 32] = perm[k] * 32u;
    }
}

// ─── Tensor core throughput kernels (WMMA) ───
// Operands live in registers; each warp drives TC_ACC independent MMA
// accumulation chains so the tensor pipes stay saturated with no memory
// traffic. FLOPs are counted host-side as 2*M*N*K per mma_sync.
#define TC_ACC 4

__global__ void bench_tensor_tf32(float * __restrict__ dummy_out, int iters) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c[TC_ACC];
    wmma::fill_fragment(a, 1.0f);
    wmma::fill_fragment(b, 1.0f);
    for (int k = 0; k < TC_ACC; k++) wmma::fill_fragment(c[k], 0.0f);
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < TC_ACC; k++) wmma::mma_sync(c[k], a, b, c[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < TC_ACC; k++)
        for (int e = 0; e < c[k].num_elements; e++) sum += c[k].x[e];
    // c accumulates only non-negative products, so sum can never be -1.0f —
    // but the compiler can't prove that, which keeps the MMA chains live
    if (sum == -1.0f) dummy_out[0] = sum;
#endif
}

__global__ void bench_tensor_bf16(float * __restrict__ dummy_out, int iters) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[TC_ACC];
    wmma::fill_fragment(a, __float2bfloat16(1.0f));
    wmma::fill_fragment(b, __float2bfloat16(1.0f));
    for (int k = 0; k < TC_ACC; k++) wmma::fill_fragment(c[k], 0.0f);
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < TC_ACC; k++) wmma::mma_sync(c[k], a, b, c[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < TC_ACC; k++)
        for (int e = 0; e < c[k].num_elements; e++) sum += c[k].x[e];
    if (sum == -1.0f) dummy_out[0] = sum;
#endif
}

__global__ void bench_tensor_int8(int * __restrict__ dummy_out, int iters) {
#if __CUDA_ARCH__ >= 720
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c[TC_ACC];
    wmma::fill_fragment(a, 1);
    wmma::fill_fragment(b, 1);
    for (int k = 0; k < TC_ACC; k++) wmma::fill_fragment(c[k], 0);
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < TC_ACC; k++) wmma::mma_sync(c[k], a, b, c[k]);
    }
    long long sum = 0;
    for (int k = 0; k < TC_ACC; k++)
        for (int e = 0; e < c[k].num_elements; e++) sum += c[k].x[e];
    if (sum == -1) dummy_out[0] = (int)sum;
#endif
}

static const char *pci_ids_paths[] = {
    "/usr/share/hwdata/pci.ids",
    "/usr/share/misc/pci.ids",
    "/usr/local/share/pci.ids",
    NULL
};

// Lookup device name from pci.ids database (e.g. "GA100 [CMP 170HX]")
static bool lookup_pci_device_name(unsigned int dev_id, char *out, size_t out_sz) {
    FILE *f = NULL;
    for (int i = 0; pci_ids_paths[i]; i++) {
        f = fopen(pci_ids_paths[i], "r");
        if (f) break;
    }
    if (!f) return false;

    char line[512];
    bool in_nvidia = false;
    char target[8];
    snprintf(target, sizeof(target), "\t%04x", dev_id);

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (!in_nvidia) {
            if (strncmp(line, "10de", 4) == 0) in_nvidia = true;
            continue;
        }
        if (line[0] != '\t') break;
        if (line[0] == '\t' && line[1] == '\t') continue;
        if (strncasecmp(line, target, 5) == 0) {
            char *name = line + 5;
            while (*name == ' ') name++;
            size_t len = strlen(name);
            while (len > 0 && (name[len-1] == '\n' || name[len-1] == '\r')) len--;
            if (len >= out_sz) len = out_sz - 1;
            memcpy(out, name, len);
            out[len] = '\0';
            fclose(f);
            return true;
        }
    }
    fclose(f);
    return false;
}

// Timed iteration count: explicit user value, or sized from a calibration
// sample so the run lasts ~kTargetMs.
static const double kTargetMs = 3000.0;
static int auto_iters(int user_iters, double per_iter_ms) {
    if (user_iters > 0) return user_iters;
    if (per_iter_ms <= 0.0) return 5;
    double it = kTargetMs / per_iter_ms;
    if (it < 1.0) it = 1.0;
    if (it > 100000000.0) it = 100000000.0;
    return (int)it;
}

// One tensor iteration is ~1000x cheaper than one 512 MiB bandwidth sweep, so
// an explicit user count sized for the bandwidth kernels would leave the
// tensor tests a microsecond timed window dominated by launch overhead and
// event resolution; floor the window at ~100 ms regardless.
static int tensor_iters(int user_iters, double per_iter_ms) {
    int it = auto_iters(user_iters, per_iter_ms);
    if (per_iter_ms > 0.0 && (double)it * per_iter_ms < 100.0) {
        double min_it = 100.0 / per_iter_ms;
        it = min_it > 100000000.0 ? 100000000 : (int)min_it;
    }
    return it;
}

void print_bar() {
    printf("─────────────────────────────────────────────────────────────────────────────\n");
}

int main(int argc, char **argv) {
    size_t size_mb = 512;
    int device = 0;
    int iters = 0;  // <= 0: auto-size each timed run to ~kTargetMs

    if (argc > 1) device = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (device < 0 || device >= dev_count) {
        fprintf(stderr, "Error: GPU %d not found (%d GPU(s) available)\n", device, dev_count);
        return 1;
    }
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    CHECK_CUDA(cudaSetDevice(device));

    int l2_bytes = 0;
    if (cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device) != cudaSuccess) l2_bytes = 0;

    // NVML is best-effort (identification and clock info only): resolve the handle
    // by PCI bus id — NVML indices are PCI-ordered and ignore CUDA_VISIBLE_DEVICES,
    // so reusing the CUDA ordinal can address a different physical GPU.
    bool nvml_inited = (nvmlInit() == NVML_SUCCESS);
    bool nvml_ok = nvml_inited;
    nvmlDevice_t nvml_dev = NULL;
    if (nvml_ok) {
        char pci_bus[32];
        nvml_ok = cudaDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), device) == cudaSuccess
               && nvmlDeviceGetHandleByPciBusId_v2(pci_bus, &nvml_dev) == NVML_SUCCESS;
    }

    unsigned int nvml_cur_mhz = 0, nvml_max_mhz = 0;
    if (nvml_ok) {
        if (nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_MEM, &nvml_cur_mhz) != NVML_SUCCESS) nvml_cur_mhz = 0;
        if (nvmlDeviceGetMaxClockInfo(nvml_dev, NVML_CLOCK_MEM, &nvml_max_mhz) != NVML_SUCCESS) nvml_max_mhz = 0;
    }
    int cuda_max_khz = 0;
    if (cudaDeviceGetAttribute(&cuda_max_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess) cuda_max_khz = 0;
    unsigned int mem_clock_mhz = nvml_cur_mhz;  // current clock wins if overclocked
    if (nvml_max_mhz > mem_clock_mhz) mem_clock_mhz = nvml_max_mhz;
    if ((unsigned int)(cuda_max_khz / 1000) > mem_clock_mhz) mem_clock_mhz = cuda_max_khz / 1000;

    unsigned int dev_id = 0;
    char pci_name[256] = {0};
    bool has_pci_name = false;
    if (nvml_ok) {
        nvmlPciInfo_t pci;
        if (nvmlDeviceGetPciInfo(nvml_dev, &pci) == NVML_SUCCESS) {
            dev_id = (pci.pciDeviceId >> 16) & 0xFFFF;
            has_pci_name = lookup_pci_device_name(dev_id, pci_name, sizeof(pci_name));
        }
    }

    print_bar();
    printf("  NVIDIA Memory Bandwidth Benchmark\n");
    print_bar();
    if (has_pci_name)
        printf("  GPU:               %s [%04X] (%s)\n", prop.name, dev_id, pci_name);
    else if (dev_id)
        printf("  GPU:               %s [%04X]\n", prop.name, dev_id);
    else
        printf("  GPU:               %s\n", prop.name);
    printf("  SMs:               %d\n", prop.multiProcessorCount);
    printf("  Memory Bus Width:  %d-bit\n", prop.memoryBusWidth);
    if (l2_bytes > 0)
        printf("  L2 Cache:          %.0f MiB\n", l2_bytes / 1048576.0);
    if (mem_clock_mhz > 0) {
        printf("  Memory Clock:      %u MHz\n", mem_clock_mhz);
        double theo_bw = (double)mem_clock_mhz * 2 * prop.memoryBusWidth / 8 / 1000;
        printf("  Theoretical BW:    %.2f GB/s\n", theo_bw);
    } else {
        printf("  Memory Clock:      unknown\n");
    }
    unsigned long long total_mem = (unsigned long long)prop.totalGlobalMem;
    if (nvml_ok) {
        nvmlMemory_t mem_info;
        memset(&mem_info, 0, sizeof(mem_info));
        if (nvmlDeviceGetMemoryInfo(nvml_dev, &mem_info) == NVML_SUCCESS && mem_info.total > 0)
            total_mem = mem_info.total;
    }
    printf("  Total Memory:      %.2f GiB (%llu MiB)\n", total_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024 * 1024));
    printf("  Test Buffer Size:  %zu MiB per array\n", size_mb);
    if (iters > 0)
        printf("  Kernel Iterations: %d\n", iters);
    else
        printf("  Kernel Iterations: auto (~%.0f s per test)\n", kTargetMs / 1000.0);
    print_bar();

    size_t total_bytes = size_mb * 1024 * 1024;
    size_t N_vec = total_bytes / sizeof(float4);
    total_bytes = N_vec * sizeof(float4); // Align to exact multiple of sizeof(float4)

    // Configure grid geometry to maximize HBM memory controller utilization
    int threads = 256;
    int blocks = prop.multiProcessorCount * 16;
    if ((size_t)blocks * threads > N_vec) blocks = (int)((N_vec + threads - 1) / threads);

    // Device memory allocation
    float4 *d_A, *d_B, *d_C;
    float *d_dummy;
    CHECK_CUDA(cudaMalloc(&d_A, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_dummy, sizeof(float)));

    // Initialize buffers
    CHECK_CUDA(cudaMemset(d_A, 0x3F, total_bytes));
    CHECK_CUDA(cudaMemset(d_B, 0x40, total_bytes));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float time_ms = 0.0f;
    float sample_ms = 0.0f;

    // ─── Memory Latency (Pointer Chasing) ───
    // One chain slot per 128-byte cache line, chain sized >= 16x L2 (min 512 MiB):
    // a random walk over a footprint comparable to L2 partially hits cache and
    // reads low (e.g. a 128 MiB chain under-reports ~5% at 32 MiB L2, up to ~2x
    // on 96+ MiB L2 parts).
    size_t chain_bytes = (size_t)l2_bytes * 16;
    if (chain_bytes < 512ull * 1024 * 1024) chain_bytes = 512ull * 1024 * 1024;
    size_t free_bytes = 0, total_dev_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_dev_bytes));
    while (chain_bytes > free_bytes / 2 && chain_bytes > 64ull * 1024 * 1024) chain_bytes /= 2;
    if ((size_t)l2_bytes * 8 > chain_bytes)
        printf("  Warning: latency chain limited to %zu MiB (< 8x L2), latency may read low\n", chain_bytes >> 20);

    size_t n_slots = chain_bytes / 128;

    uint32_t *h_perm = (uint32_t *)malloc(n_slots * sizeof(uint32_t));
    if (!h_perm) { fprintf(stderr, "Error: host allocation of %zu MiB failed\n", (n_slots * sizeof(uint32_t)) >> 20); return 1; }
    for (size_t i = 0; i < n_slots; i++) h_perm[i] = (uint32_t)i;
    // Sattolo shuffle — single cycle visiting all slots
    for (size_t i = n_slots - 1; i > 0; i--) {
        size_t j = (size_t)rand() % i;
        uint32_t tmp = h_perm[i]; h_perm[i] = h_perm[j]; h_perm[j] = tmp;
    }

    uint32_t *d_perm, *d_chain, *d_out;
    CHECK_CUDA(cudaMalloc(&d_perm, n_slots * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_chain, chain_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(d_perm, h_perm, n_slots * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(h_perm);
    build_chain<<<blocks, threads>>>(d_perm, d_chain, n_slots);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_perm));

    // Warmup doubles as calibration: time a 1M-step walk to size the measured one
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_latency<<<1, 1>>>(d_chain, d_out, 1000000);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
    int lat_steps = 1000000;
    if (iters <= 0) {
        double s = kTargetMs / ((double)sample_ms / 1000000.0);
        if (s < 1000000.0) s = 1000000.0;
        if (s > 50000000.0) s = 50000000.0;
        lat_steps = (int)s;
    }
    // The warmup walked the same path the timed run will take; stream 512 MiB of
    // writes through L2 to evict that footprint (keeps clocks/TLB warm only)
    CHECK_CUDA(cudaMemset(d_A, 0x3F, total_bytes));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_latency<<<1, 1>>>(d_chain, d_out, lat_steps);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    double latency_ns = (double)time_ms * 1e6 / lat_steps;
    printf("  Memory Latency:         %6.1f ns   (chain %zu MiB, 128 B stride)\n", latency_ns, chain_bytes >> 20);
    fflush(stdout);

    CHECK_CUDA(cudaFree(d_chain));
    CHECK_CUDA(cudaFree(d_out));

    // ─── Memory Bandwidth ───
    bench_hbm_read<<<blocks, threads>>>(d_A, d_dummy, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_read<<<blocks, threads>>>(d_A, d_dummy, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
    int read_iters = auto_iters(iters, sample_ms / 5.0);
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_read<<<blocks, threads>>>(d_A, d_dummy, N_vec, read_iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    printf("  Global Read Bandwidth:  %8.2f GB/s\n", ((double)total_bytes * read_iters) / 1e9 / (time_ms / 1000.0));
    fflush(stdout);

    float4 val = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    bench_hbm_write<<<blocks, threads>>>(d_A, val, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_write<<<blocks, threads>>>(d_A, val, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
    int write_iters = auto_iters(iters, sample_ms / 5.0);
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_write<<<blocks, threads>>>(d_A, val, N_vec, write_iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    printf("  Global Write Bandwidth: %8.2f GB/s\n", ((double)total_bytes * write_iters) / 1e9 / (time_ms / 1000.0));
    fflush(stdout);

    bench_hbm_copy<<<blocks, threads>>>(d_A, d_B, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_copy<<<blocks, threads>>>(d_A, d_B, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
    int copy_iters = auto_iters(iters, sample_ms / 5.0);
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_copy<<<blocks, threads>>>(d_A, d_B, N_vec, copy_iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    printf("  Global Copy Bandwidth:  %8.2f GB/s\n", ((double)total_bytes * 2 * copy_iters) / 1e9 / (time_ms / 1000.0));
    fflush(stdout);

    bench_hbm_triad<<<blocks, threads>>>(d_A, d_B, d_C, 2.5f, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_triad<<<blocks, threads>>>(d_A, d_B, d_C, 2.5f, N_vec, 5);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
    int triad_iters = auto_iters(iters, sample_ms / 5.0);
    CHECK_CUDA(cudaEventRecord(start));
    bench_hbm_triad<<<blocks, threads>>>(d_A, d_B, d_C, 2.5f, N_vec, triad_iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    printf("  Global Triad Bandwidth: %8.2f GB/s\n", ((double)total_bytes * 3 * triad_iters) / 1e9 / (time_ms / 1000.0));
    fflush(stdout);

    // Intra-device cudaMemcpy runs as an SM copy kernel, not on the DMA engines,
    // hence the honest label. Window auto-sized like the kernels above.
    CHECK_CUDA(cudaMemcpy(d_B, d_A, total_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_B, d_A, total_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
    int dma_iters = auto_iters(iters, sample_ms);
    if (dma_iters < 20) dma_iters = 20;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < dma_iters; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_B, d_A, total_bytes, cudaMemcpyDeviceToDevice, 0));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    printf("  cudaMemcpy D2D BW:      %8.2f GB/s\n", ((double)total_bytes * 2 * dma_iters) / 1e9 / (time_ms / 1000.0));
    print_bar();

    // ─── Tensor Core Throughput ───
    // 48 warps/SM is plenty to saturate the tensor pipes; calibration samples
    // use 10000 iters because a single iteration is far below event resolution
    int tc_blocks = prop.multiProcessorCount * 6;
    int tc_threads = 256;
    double tc_warps = (double)tc_blocks * tc_threads / 32.0;
    const int tc_sample = 10000;

    if (prop.major >= 8) {
        bench_tensor_tf32<<<tc_blocks, tc_threads>>>(d_dummy, tc_sample);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        bench_tensor_tf32<<<tc_blocks, tc_threads>>>(d_dummy, tc_sample);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
        int tf32_iters = tensor_iters(iters, (double)sample_ms / tc_sample);
        CHECK_CUDA(cudaEventRecord(start));
        bench_tensor_tf32<<<tc_blocks, tc_threads>>>(d_dummy, tf32_iters);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        double ops = tc_warps * (double)tf32_iters * TC_ACC * (2.0 * 16 * 16 * 8);
        printf("  FP32 Tensor (TF32):     %8.2f TFLOPS\n", ops / 1e12 / (time_ms / 1000.0));
    } else {
        printf("  FP32 Tensor (TF32):          n/a (requires sm_80+)\n");
    }
    fflush(stdout);

    if (prop.major >= 8) {
        bench_tensor_bf16<<<tc_blocks, tc_threads>>>(d_dummy, tc_sample);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start));
        bench_tensor_bf16<<<tc_blocks, tc_threads>>>(d_dummy, tc_sample);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
        int bf16_iters = tensor_iters(iters, (double)sample_ms / tc_sample);
        CHECK_CUDA(cudaEventRecord(start));
        bench_tensor_bf16<<<tc_blocks, tc_threads>>>(d_dummy, bf16_iters);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        double ops = tc_warps * (double)bf16_iters * TC_ACC * (2.0 * 16 * 16 * 16);
        printf("  BF16 Tensor:            %8.2f TFLOPS\n", ops / 1e12 / (time_ms / 1000.0));
    } else {
        printf("  BF16 Tensor:                 n/a (requires sm_80+)\n");
    }
    fflush(stdout);

    // TU116/TU117 report CC 7.5 but have no tensor cores — the s8 IMMA path
    // faults at runtime and no device property distinguishes them, so probe
    // non-fatally. INT8 runs last: a faulted context has only cleanup left.
    bench_tensor_int8<<<tc_blocks, tc_threads>>>((int *)d_dummy, tc_sample);
    bool int8_ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    if (int8_ok) {
        CHECK_CUDA(cudaEventRecord(start));
        bench_tensor_int8<<<tc_blocks, tc_threads>>>((int *)d_dummy, tc_sample);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&sample_ms, start, stop));
        int int8_iters = tensor_iters(iters, (double)sample_ms / tc_sample);
        CHECK_CUDA(cudaEventRecord(start));
        bench_tensor_int8<<<tc_blocks, tc_threads>>>((int *)d_dummy, int8_iters);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        double int8_ops = tc_warps * (double)int8_iters * TC_ACC * (2.0 * 16 * 16 * 16);
        printf("  INT8 Tensor:            %8.2f TOPS\n", int8_ops / 1e12 / (time_ms / 1000.0));
    } else {
        cudaGetLastError();  // clear the sticky error
        printf("  INT8 Tensor:                 n/a (no tensor cores)\n");
    }
    print_bar();

    // Cleanup (skipped if the INT8 probe faulted the context; process exit reclaims)
    if (int8_ok) {
        CHECK_CUDA(cudaFree(d_dummy));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    if (nvml_inited) nvmlShutdown();

    return 0;
}
