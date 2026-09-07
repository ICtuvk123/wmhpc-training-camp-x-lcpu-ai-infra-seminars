# Blackwell Phase-6 feasibility experiment (B300 / SM103)

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

The default target is B300 (`sm_103a`, compute capability 10.3). Both still
require compilation and validation on the target B300 machine; do
not treat unverified source code as a performance result.

The corrected sources were compile-checked with CUDA 13.0 for `sm_103a`
using CUTLASS commit `59e3a3338d516ca6ce0e073af8da65289678a35c`. PTXAS reported
32 registers/thread for the baseline and 154 for tcgen05, with zero spills in
both. The regression runner passed Python syntax, dry-run, and input-coverage
checks. These are build-time and host-side checks only; numerical correctness,
Compute Sanitizer, latency, and residency still need to be measured on B300.

Both programs use the same storage contract:

- `kt`: row-major `[128,16]`
- `u`: row-major `[16,128]`
- `c`: row-major `[128,128]`, FP32
- one CTA computes one output matrix; all CTAs reuse the same inputs so a large
  grid can measure steady-state throughput

## Prerequisites

- A B300 GPU (SM103, compute capability 10.3)
- CUDA 13.0 or newer, with `nvcc` on `PATH`
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

On the B300 machine (defaults to `--arch sm_103a`):

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

## Baseline correctness diagnostics

The baseline presents MMA B as logical `[N,K]`, viewing the row-major `U[K,N]`
allocation with stride `(1,N)`, then copying it into K-major shared memory.
Passing a `[K,N]` tile directly to `partition_fragment_B` silently exchanges
the tile's output-column and reduction coordinates because both extents are 16.
See the [CuTe MMA atom documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0t_mma_atom.md)
for the `(M,K)`, `(N,K)`, `(M,N)` operand convention.

Build for B300 using the commands above, then run on the B300 machine:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./experiments/sm100_phase6/build/baseline --warmup 1 --iters 1 --batch 1

# One nonzero at C[5,37], exercising different K and tile-local N indices.
./experiments/sm100_phase6/build/baseline --warmup 1 --iters 1 --batch 1 \
  --input one-hot --m 5 --k 3 --n 37

python experiments/sm100_phase6/check_baseline.py \
  experiments/sm100_phase6/build/baseline
python experiments/sm100_phase6/check_baseline.py \
  experiments/sm100_phase6/build/baseline --sanitizer compute-sanitizer
```

On Windows, use `build/baseline.exe`. The regression runner requires CC 10.3 and checks 22 input
cases at batch sizes 1 and 3: origin and off-diagonal one-hot probes across
tiles/warps and all 16 reduction coordinates, an exactly representable dense
pattern, and three random seeds. Deterministic inputs require exact equality;
random inputs use `atol=rtol=1e-5` against the reference computed from the actual
BF16 inputs. Both the first and last CTA outputs are checked, and failures
print the first mismatching block, row, column, expected value, and actual value.

Both executables accept SM100 (`10.0`) and SM103 (`10.3`). Build for the actual
device: the default `--arch sm_103a` is for B300/GB300; a separate B200/GB200
experiment would need `--arch sm_100a`. An architecture-specific binary must match
its target device. Compilation alone does not validate these regressions.
No results from a different GPU architecture qualify as Blackwell validation.
See NVIDIA's [GPU capability table](https://developer.nvidia.com/cuda/gpus) and
[Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-compatibility-guide/index.html).

A clean memcheck run only means it detected no checked memory errors in that
execution. It does not establish mathematical correctness or rule out races;
the runner also accepts `--tool racecheck`, `--tool synccheck`, and
`--tool initcheck` with `--sanitizer`. See the
[Compute Sanitizer manual](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

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

P1 is now implemented separately; see [P1 experiment and validation](P1.md).
Use `benchmark_p1.py` for P1. The original `benchmark.py` still runs P0.

Keep each later step separately measurable:

1. P0: `C = K^T U`
2. P1: `C = S_old + K^T U`
3. P2: `C = S_old * g + K^T U`
4. P3: repeated recurrence for 16, 64, 128, and 512 iterations

Do not edit the production K2 kernel until the P2/P3 measurements justify it.
