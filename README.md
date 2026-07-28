# NVIDIA Memory, PCIe & Compute Benchmark

A single-file CUDA benchmark that measures memory latency, peak memory bandwidth, PCIe throughput and dense CUDA-core / Tensor-core compute on NVIDIA consumer GPUs and datacenter accelerators.

![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)

---

## Metrics

| Metric Category | Tests / Details |
| :--- | :--- |
| **GPU Telemetry** | Model, SM count, CUDA cores, bus width, L2 size, SM clock (measured), memory clock (sampled under load), theoretical max BW, VRAM |
| **Memory Latency** | Pointer chasing over a chain scaled to ≥ 16× L2 (min 512 MiB), one slot per 128-byte line |
| **Memory Bandwidth** | Global Read, Write, Copy, Triad, `cudaMemcpy` D2D |
| **PCIe** | H2D / D2H with pinned memory, trained link (gen × width) and measured-vs-link efficiency |
| **CUDA-core Compute** | FP64, FP32, FP16 FMA throughput (TFLOPS) |
| **Tensor Compute** | FP32 Tensor (TF32), BF16 Tensor, INT8 Tensor (TFLOPS / TOPS) |
| **Features** | Tensor cores, NVENC/NVDEC engine counts, MIG, ECC, NVLink, copy engines, BAR1 size, power limit |

---

## Prerequisites

* **NVIDIA CUDA Toolkit** (`nvcc`) — the source builds with 11.0+; the multi-arch line below needs 12.8+ for the Blackwell gencodes
* **NVIDIA driver** with NVML (`-lnvidia-ml`); `libnvidia-encode` / `libnvcuvid` are used when present to probe NVENC/NVDEC
* **Linux**

---

## Building

```bash
nvcc -O3 -o nvidia_bench nvidia_bench.cu -lnvidia-ml -ldl \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120
```

Drop the gencodes you don't need — a single one that matches your GPU builds much faster.

---

## Run

```bash
./nvidia_bench                    # GPU 0, ~17 s
./nvidia_bench [gpu_id] [iterations] [--csv]
```

| Argument | Meaning |
| :--- | :--- |
| `gpu_id` | CUDA device index (default `0`) |
| `iterations` | Explicit per-test iteration count. `<= 0` (default) auto-sizes every timed run to ~1 s, which stays under the ~2 s display watchdog; compute tests keep a 100 ms floor |
| `--csv` | Print one CSV header line and one data line instead of the report — for scripted clock/BIOS sweeps |
| `-h`, `--help` | Usage |

---

## Output example

```
─────────────────────────────────────────────────────────────────────────────
  NVIDIA Performance And Memory Benchmark
─────────────────────────────────────────────────────────────────────────────
  GPU:               NVIDIA Graphics Device [20C2] (GA100 [CMP 170HX])
  SMs:               70
  CUDA Cores:        4480 (64 per SM)
  Memory Bus Width:  4096-bit
  L2 Cache:          32 MiB
  SM Clock:          1695 MHz (measured, max 1935 MHz)
  Memory Clock:      1971 MHz (under load, rated 1728 MHz)
  Theoretical BW:    2018.30 GB/s
  Total Memory:      64.00 GiB (65536 MiB)
  Test Buffer Size:  512 MiB per array
  Kernel Iterations: auto (~1.0 s per test)
─────────────────────────────────────────────────────────────────────────────
  PCIe H2D Bandwidth:         0.81 GB/s   (pinned, 256 MiB blocks)
  PCIe D2H Bandwidth:         0.84 GB/s
  PCIe Link (NVML):       Gen1 x4
  PCIe BW vs link:        95% of ~0.88 GB/s expected for Gen1 x4
─────────────────────────────────────────────────────────────────────────────
  Features:
    Tensor Cores:     yes            NVENC:            no
    ECC:              no             NVDEC:            no
    MIG:              supported, off NVLink:           no
    Copy Engines:     8              Managed Memory:   yes
    Coop Launch:      yes            Compute Preempt:  yes
    Display Active:   no             Power Limit:      300 W
    BAR1 Size:        64 MiB
─────────────────────────────────────────────────────────────────────────────
  Memory Latency:          315.7 ns   (chain 512 MiB, 128 B stride)
  Global Read Bandwidth:   1903.46 GB/s
  Global Write Bandwidth:  1726.38 GB/s
  Global Copy Bandwidth:   1742.68 GB/s
  Global Triad Bandwidth:  1835.21 GB/s
  cudaMemcpy D2D BW:       1825.48 GB/s
─────────────────────────────────────────────────────────────────────────────
  FP64 (CUDA cores):          7.46 TFLOPS
  FP32 (CUDA cores):         15.11 TFLOPS
  FP16 (CUDA cores):         53.60 TFLOPS
  FP32 Tensor (TF32):       112.94 TFLOPS
  BF16 Tensor:              228.54 TFLOPS
  INT8 Tensor:              465.73 TOPS
─────────────────────────────────────────────────────────────────────────────
```

---

## Notes

* **Clocks are measured, not read from spec tables.** The SM clock comes from on-device `clock64()` versus `%globaltimer`; the memory clock is sampled while the GPU is busy. An overclock or underclock therefore shows up in the report and in the theoretical-bandwidth figure.
* **The latency chain scales with L2** so the walk lands in DRAM rather than cache — a chain comparable to L2 can under-report latency by up to 2× on large-L2 parts.
* **Tensor numbers use WMMA.** On Hopper/Blackwell datacenter parts that can understate absolute peak, since the `wgmma` / `tcgen05` paths are not used.
* **Test buffers shrink automatically** when VRAM is short, and the tool says so.
* On a display-attached GPU, pass a small explicit iteration count if the driver watchdog is aggressive.
