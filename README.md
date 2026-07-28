# NVIDIA Memory & Tensor Bandwidth Benchmark

A lightweight, high-performance CUDA benchmark utility designed to measure memory latency, peak memory bandwidth, and dense Tensor Core performance on NVIDIA consumer GPUs and datacenter accelerators.

![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)

---

## Metrics

| Metric Category | Tests / Details |
| :--- | :--- |
| **GPU Telemetry** | Model name, SM count, Bus Width, Memory Clock, Theoretical Max BW, Total VRAM |
| **Memory Latency** | Pointer-chasing latency (512 MiB chain, 128-byte stride) |
| **Memory Bandwidth** | Global Read, Global Write, Global Copy, Global Triad, `cudaMemcpy` D2D |
| **Tensor Compute** | FP32 Tensor (TF32), BF16 Tensor, INT8 Tensor (TFLOPS / TOPS) |

---

## Prerequisites

* **NVIDIA CUDA Toolkit** (`nvcc` compiler)
* **NVIDIA Display Driver** with NVML development headers (`-lnvidia-ml`)
* **Linux environment**

---

## Building

Compile the benchmark using `nvcc` with optimization level `-O3` and target compute architectures:

```bash
nvcc -O3 -o nvidia_bench nvidia_bench.cu -lnvidia-ml \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120  
 ```

---
 
## Run
```bash
./nvidia_bench
./nvidia_bench [gpu_id] [iterations]
```

---

## Output example
<img width="742" height="516" alt="image" src="https://github.com/user-attachments/assets/1a883881-d1c1-43a9-bcd1-008217bcb0a6" />
