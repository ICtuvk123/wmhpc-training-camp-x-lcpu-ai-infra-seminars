# Production K2: opt-in Phase-6 V0 and isolated Phase-1 probe

Status: compiled and host layout checks passed; target GPU correctness and
timing are pending. P0-P3 in `experiments/sm100_phase6` are unchanged.

## V0 contract

`FLASH_KDA_ENABLE_SM100_V0=1` compiles an additional K2 specialization. At runtime,
`FLASH_KDA_K2_IMPL=sm100_v0` selects it; unset or `baseline` selects the existing
implementation. Explicit V0 selection requires SM100/SM103. Python signatures
and the default device path are unchanged.

Canonical state remains `state_acc` in BF16 SMEM, including when initial/final
state I/O uses FP32. Phase 1 still reads it through the existing LDSM path.
Only Phase 6 changes:

```text
existing BF16 U C fragments -> STSM -> dedicated U buffer -> UMMA B staging
production k_restored SMEM ------------------------------> UMMA A staging
                                                          |
                                                 tcgen05 K^T U
                                                          |
                                                  TMEM accumulator
                                                          |
                                                   FP32 registers
                                                          |
state_acc BF16 SMEM + row-wise FP32 g_total ----------> S*g + acc
                                                          |
                                                    round to BF16
                                                          |
                                                   state_acc SMEM
                                                          |
                                                  next chunk Phase 1
```

The transposed state view is `[key,value]`; `g_total[key]` scales key-feature
rows, matching the baseline epilogue. State initialization and final store,
Phases 1-5, and the input/output pipelines retain their existing code.

TMEM is allocated once per compute group and freed after the recurrence loop.
The new synchronization uses the existing 128-thread named barrier and a
tcgen05 completion barrier with alternating parity. No CTA-wide barrier is
introduced inside the compute-only branch of the 192-thread CTA.

The existing 32 KiB BF16 state allocation remains. V0 adds 12,416 bytes for
U staging, UMMA operands, barrier storage, and alignment. This is a feasibility
implementation; extra staging and register pressure may outweigh Phase-6 gains.
Production already amortizes GMEM state I/O across chunks, so P3's timings do
not predict the benefit of this integration.

## Build and validate on B300

Commands below run from the repository root, with CUDA 13, CUDA-enabled PyTorch,
and the repository CUTLASS submodule available. Force rebuilding when changing
the compile-time flag:

```bash
FLASH_KDA_ENABLE_SM100_V0=1 FLASH_KDA_CUDA_ARCHS=103a \
  python setup.py build_ext --inplace --force
python -m pytest -p no:cacheprovider experiments/sm100_k2_integration/test_v0.py -x -q
FLASH_KDA_K2_IMPL=baseline python -m pytest tests/test_fwd_full.py -x -q
FLASH_KDA_K2_IMPL=sm100_v0 python -m pytest tests/test_fwd_full.py -x -q
```

The new tests alternate baseline/V0 on identical inputs and require exact
output/final-state equality, finite results, and unmodified initial state.
They cover all state I/O combinations, BF16/FP32 state, partial chunks, batches,
varlen including empty sequences, and the three frozen workloads. The existing
reference tests remain the independent numerical acceptance check. Compilation
does not establish equivalence between the SM80 and tcgen05 arithmetic; an
exact-match failure must be investigated before accepting V0.

To test the default build separately, rebuild without
`FLASH_KDA_ENABLE_SM100_V0` for the target architecture and rerun the baseline
tests. Runtime selection alone does not test a build with V0 compiled out.

## Standalone K2 timing, without PyTorch or NCU

This executable links the actual production launcher. It executes K1 once to
prepare the real workspace, captures each full forward, verifies the graph
contains K1 followed by a terminal 192-thread K2, and copies that K2 node into
an isolated graph. It compares complete outputs and final states before timing,
then alternates baseline/V0 for five rounds, reversing order each round.
State is reinitialized from the same initial input for every K2 launch.

```bash
mkdir -p experiments/sm100_k2_integration/build
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  --ptxas-options=-v,--register-usage-level=10,--warn-on-spills \
  -DFLASH_KDA_ENABLE_SM100_V0=1 -Icutlass/include -Icsrc \
  -c csrc/smxx/fwd_launch.cu \
  -o experiments/sm100_k2_integration/build/fwd_v0.o
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -Icutlass/include -Icsrc \
  experiments/sm100_k2_integration/k2_compare.cu \
  experiments/sm100_k2_integration/build/fwd_v0.o \
  -o experiments/sm100_k2_integration/build/k2_compare
experiments/sm100_k2_integration/build/k2_compare --batch 4 --tokens 2048 --warmup 30 --iters 200 --rounds 5
experiments/sm100_k2_integration/build/k2_compare --batch 1 --tokens 8192 --warmup 30 --iters 200 --rounds 5
experiments/sm100_k2_integration/build/k2_compare --batch 8 --tokens 1024 --warmup 30 --iters 200 --rounds 5
```

H=64 and D=128 are fixed. The comparator uses BF16 initial/final state; FP32
and other API cases are covered by the Python tests. Each timing JSON includes
`k2_us`, `registers_per_thread`, `dynamic_smem_bytes`, and
`local_bytes_per_thread`; the last field is not a spill instruction count.
Use ptxas output for spills. Timing uses CUDA events around repeated launches
of a single-node K2 graph. Compare the paired means from this executable;
comparison with an earlier ~460 us baseline requires matching measurement
method and input/state configuration.

## Independent Phase-1 register-state probe

`phase1_register_state.cu` loads state once in the **SM80 Phase-6 C distribution**,
uses four packed `MOVM_T` operations per 16x16 tile to form Phase-1 B fragments,
and keeps those fragments in registers while computing kS/qS for changing K/Q
tiles. It checks every result against a CPU reference and checks the complete
state reconstructed from B fragments. Dyadic input values make the reference
sums exactly representable in FP32.

This tests a narrow feasibility question: can Phase 1 consume persistent state
in this warp-local representation? State is constant across its steps. It does
not implement recurrent updates, redistribute the Blackwell TMEM-copy fragment
into this representation, or reproduce production operand staging. K/Q load
directly from GMEM in this probe. A passing result is not a full V1 GO.

The host layout check emulates the documented
[PTX movmatrix transpose](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-movmatrix)
on coordinate tags and compares against CuTe's B partition. It also checks
production state/U/K transpose aliases and the V0 state/decay partition across
all 128 compute threads, with each state element visited exactly once.

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  --ptxas-options=-v,--warn-on-spills -Icutlass/include \
  experiments/sm100_k2_integration/phase1_register_state.cu \
  -o experiments/sm100_k2_integration/build/phase1_register_state
nvcc -std=c++17 -O3 -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -Icutlass/include \
  experiments/sm100_k2_integration/check_layouts.cu \
  -o experiments/sm100_k2_integration/build/check_layouts
experiments/sm100_k2_integration/build/check_layouts
for steps in 1 2 4 8 16; do
  experiments/sm100_k2_integration/build/phase1_register_state --steps "$steps" --batch 3
done
```

## Local validation record

CUDA 13.0, MSVC 2022, SM103a target, cached CUTLASS headers:

| Compiled kernel | Registers/thread | Dynamic SMEM bytes | Spill stores/loads |
|---|---:|---:|---:|
| Baseline K2, fixed-length BF16 state in+out | 73 | 98,432 | 0 / 0 |
| V0 K2, fixed-length BF16 state in+out | 163 | 110,848 | 0 / 0 |
| Baseline K2, fixed-length FP32 state in+out | 74 | 98,432 | 0 / 0 |
| V0 K2, fixed-length FP32 state in+out | 164 | 110,848 | 0 / 0 |
| Isolated Phase-1 probe | 126 | 0 | 0 / 0 |

All 14 baseline and 14 V0 state/varlen specializations compiled for SM103a.
Across those V0 variants, register counts are 161-165; all have zero stack and
spills. The production launcher also compiled for SM90a with V0 disabled and
enabled; all 14 SM90 baseline variants retain identical register/stack/spill
counts across those two builds. This is compile evidence, not runtime fallback
validation.
The host layout checker ran successfully. The standalone comparator and the
Phase-1 executable linked successfully.

The local GPU is an RTX 4060 Laptop, and PyTorch is 2.8.0+cpu. Both GPU
executables stop at `cudaGetDevice` with status 801 before executing a kernel;
the new pytest module skips without CUDA-enabled PyTorch. The full PyTorch
extension build, GPU correctness, synchronization behavior, runtime fallback,
and all latency/speedup measurements therefore remain unverified on target
hardware. No NCU run or performance GO/NO-GO claim has been made.

## V1a register-state experiment

V1a is a separate opt-in specialization. It supports fixed-length execution
with BF16 initial and final state only. Build and select it with:

```bash
FLASH_KDA_ENABLE_V1A=1 FLASH_KDA_CUDA_ARCHS=103a \
  python setup.py build_ext --inplace --force
FLASH_KDA_K2_IMPL=v1a python -m pytest -p no:cacheprovider \
  experiments/sm100_k2_integration/test_v1a.py -x -q
```

The test runs recurrence depths 1, 2, 4, 8, 16, and 128 chunks. Every depth
requires exact output and final-state equality with baseline and verifies that
the initial state is unchanged. Run this ladder before the full suite.

V1a removes canonical `state_acc` from SharedStorage. Each compute thread owns
eight key blocks for each of its two 16-column blocks, totaling 128 BF16 values
or 64 packed 32-bit registers. Phase 1 consumes those persistent B fragments
directly. Phase 6 retains the production SM80 MMA, converts each B fragment to
the matching C distribution with `MOVM_T`, applies FP32 decay/update, rounds to
BF16, and converts it back for the next chunk. Initial/final state use direct,
coalesced GMEM access at recurrence boundaries. No full-state staging or
CTA-wide barrier was added.

Local SM103a compilation reports 137 registers/thread, 62,464 bytes dynamic
SMEM, zero stack, and zero spill loads/stores. Compared with the matching
baseline specialization, this removes 35,968 bytes of dynamic SMEM. These are
compiler/layout results only; the recurrence ladder and latency remain pending
on B300.

For an isolated, paired production-K2 timing, keep the V0 comparator intact and
build the V1a-specific comparator against a V1a launcher object:

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  --ptxas-options=-v,--register-usage-level=10,--warn-on-spills \
  -DFLASH_KDA_ENABLE_V1A=1 -Icutlass/include -Icsrc \
  -c csrc/smxx/fwd_launch.cu \
  -o experiments/sm100_k2_integration/build/fwd_v1a.o
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -Icutlass/include -Icsrc \
  experiments/sm100_k2_integration/k2_compare_v1a.cu \
  experiments/sm100_k2_integration/build/fwd_v1a.o \
  -o experiments/sm100_k2_integration/build/k2_compare_v1a
experiments/sm100_k2_integration/build/k2_compare_v1a \
  --batch 4 --tokens 2048 --warmup 30 --iters 200 --rounds 5
```

This comparator invokes the specializations directly; it does not read
`FLASH_KDA_K2_IMPL`. Before interpreting latency, require the per-round JSON to
identify baseline as 73 registers / 98,432 bytes SMEM and V1a as 137 registers /
62,464 bytes SMEM. A 163-register / 110,848-byte second path is V0 and invalidates
the comparison. Exact full-output and final-state equality is checked before and
after timing.

### V1a maximum-carveout experiment

The V1a specialization requests `cudaSharedmemCarveoutMaxShared` with
`cudaFuncAttributePreferredSharedMemoryCarveout` before launch. Baseline and V0
do not set this preference. Kernel math, SharedStorage, and register lifetimes
are unchanged. The CUDA attribute is a preference, so verify the effective
configuration on B300 rather than assuming the request was honored:

```bash
ncu --section LaunchStats --section Occupancy \
  -o experiments/sm100_k2_integration/v1a_carveout_b8 \
  experiments/sm100_k2_integration/build/k2_compare_v1a \
  --batch 8 --tokens 1024 --warmup 1 --iters 1 --rounds 1
```

For the V1a kernel, first require approximately 200.7 KiB Shared Memory
Configuration Size and `Block Limit Shared Mem >= 3`. Registers remain 137 and
should still report `Block Limit Registers = 2`; performance and waves are not
expected to improve from this carveout-only experiment. Rebuild both
`fwd_v1a.o` and `k2_compare_v1a` before profiling because the preference is set
by the host launcher.

The first B300 structured-pattern run localized an error to the 16x16 tile
interior. `partition_B(identity)` for the SM80 TN atom exposes its identity
coordinate as `(N,K)`. V1a initially interpreted it as `(K,N)` in the direct
initial load and final store, transposing each tile. Both boundaries now swap
those coordinate components. The host checker uses coordinate-coded tiles to
require exact baseline C-to-B `MOVM_T` mapping, B-to-C-to-B roundtrip, full
128x128 register ownership, and B-fragment-to-GMEM logical indexing. The B300
recurrence ladder must be rerun after this fix before considering V1a correct.
