# Kernel Benchmarking (SM75-only fork)

Here we provide scripts to benchmark SageAttention kernels on NVIDIA T4 (Turing, SM75).

Available scripts:
- `bench_sm75_output_store.py` — SM75 CUDA kernel benchmark
- `bench_qk_int8_pv_fp16_triton.py` — Triton kernel benchmark
- `bench_baseline.py` — Baseline FlashAttention2 comparison

> **Note:** The TOPS results refer only to the Attention Kernel, excluding the quantization and smoothing.