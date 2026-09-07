# SM100 Phase-6 feasibility experiment

This directory isolates the first FlashKDA K2 Phase-6 question:

```text
K^T [128,16] @ U [16,128] -> C [128,128]
BF16 inputs, FP32 accumulation and output
```

It intentionally does not modify `csrc/smxx/fwd_kernel2.cuh`, and P0 does not
include the old state, decay, TMA pipelines, or K2 warp specialization.

## Current status

P0 contains two implemented kernels plus host-side correctness, timing,
resource-reporting, and comparison plumbing:

- `baseline.cu`: four-warp implementation using the same
  `SM80_16x8x16_F32BF16BF16F32_TN` atom as K2 Phase 6
- `tcgen05.cu`: one-CTA `SM100_MMA_F16BF16_SS` implementation with an FP32
  accumulator in 128 TMEM columns

Both still require compilation and validation on the target SM100 machine; do
not treat unverified source code as a performance result.

The sources were compile-checked with CUDA 13.0 for `sm_100a`. PTXAS reported
32 registers/thread for the baseline and 154 for tcgen05, with zero spills in
both. SASS inspection confirmed `HMMA.16816.F32.BF16` in the baseline and
`UTCHMMA` plus `LDTM` in the Blackwell binary. These are build-time checks only;
correctness, latency, and residency still need to be measured on SM100 hardware.

Both programs use the same storage contract:

- `kt`: row-major `[128,16]`
- `u`: row-major `[16,128]`
- `c`: row-major `[128,128]`, FP32
- one CTA computes one output matrix; all CTAs reuse the same inputs so a large
  grid can measure steady-state throughput

## Prerequisites

- An SM100 GPU (GB200/B200 class) for `tcgen05`
- CUDA 12.9 or newer, with `nvcc` on `PATH`
- The repository's CUTLASS submodule populated

```bash
git submodule update --init --recursive
```

If `cutlass/include` is still absent (some course snapshots retain
`.gitmodules` but omit the corresponding gitlink), create a shallow checkout:

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass
```

The most useful checked-in reference after that command is:

```text
cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu
```

It demonstrates the essential SM100 sequence: tcgen05-compatible SMEM layouts,
TMEM allocation, `cute::gemm`, the UMMA completion barrier, TMEM-to-register
copy, and TMEM deallocation. For the baseline MMA atom, follow the existing use
of `SM80_16x8x16_F32BF16BF16F32_TN` in
`csrc/smxx/fwd_kernel2.cuh` around Phase 6.

## Build and run

Inspect commands and prerequisites without building:

```bash
python experiments/sm100_phase6/benchmark.py --dry-run
```

On the SM100 machine:

```bash
python experiments/sm100_phase6/benchmark.py
```

Useful controls:

```bash
python experiments/sm100_phase6/benchmark.py --impl baseline
python experiments/sm100_phase6/benchmark.py --impl tcgen05
python experiments/sm100_phase6/benchmark.py --batch 4096 --warmup 30 --iters 200 --repeats 5
```

The harness reports correctness, median launch latency, amortized time per CTA,
static plus dynamic shared memory, registers per thread, TMEM columns, the CUDA
runtime occupancy estimate, the TMEM-derived occupancy limit, and their minimum
as effective blocks per SM. Raw results are written to
`experiments/sm100_phase6/build/results.json` (the build directory is ignored by
the repository's existing `.gitignore`).

## P0 acceptance rule

Do not compare timings until both rows say `PASS`. Use several batch sizes to
separate launch overhead from steady-state throughput. P0 supports moving on to
P1 only if the tcgen05 version is correct, is no slower than the current-style
baseline at steady state, and has acceptable SMEM/TMEM/occupancy costs.

The `<66 KiB/CTA` target belongs to the later K2 integration question. A small
P0 SMEM number alone does not prove three resident K2 CTAs: registers, threads,
barriers, and TMEM allocation may also cap residency. Confirm the final limit
with Nsight Compute after the prototype is complete.

## Stage boundary

Keep each later step separately measurable:

1. P0: `C = K^T U`
2. P1: `C = S_old + K^T U`
3. P2: `C = S_old * g + K^T U`
4. P3: repeated recurrence for 16, 64, 128, and 512 iterations

Do not edit the production K2 kernel until the P2/P3 measurements justify it.
