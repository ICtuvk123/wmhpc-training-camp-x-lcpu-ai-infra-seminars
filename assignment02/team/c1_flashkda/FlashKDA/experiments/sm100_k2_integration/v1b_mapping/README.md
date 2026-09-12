# V1b Phase-6 mapping probe

This standalone SM103a probe does not instantiate or modify production V1b.
It reconstructs production Phase-4 SM80 U B fragments, stages their logical
`(N,K)` coordinates into the 4 KiB tcgen05 B operand, computes
`[128,16] @ [16,128]` into TMEM, and matches every TMEM C coordinate against
the existing V1a persistent SM80-B state coordinates owned by that thread.

Build and run on B300:

```bash
mkdir -p experiments/sm100_k2_integration/v1b_mapping/build
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  --ptxas-options=-v,--register-usage-level=10,--warn-on-spills \
  -Icutlass/include -Icsrc \
  experiments/sm100_k2_integration/v1b_mapping/v1b_phase6_mapping.cu \
  -o experiments/sm100_k2_integration/v1b_mapping/build/v1b_phase6_mapping

experiments/sm100_k2_integration/v1b_mapping/build/v1b_phase6_mapping \
  --warmup 30 --iters 200 \
  --dump-ownership experiments/sm100_k2_integration/v1b_mapping/ownership.csv
```

Run only the host/static ownership sweep (no diagnostic kernel launch):

```bash
experiments/sm100_k2_integration/v1b_mapping/build/v1b_phase6_mapping \
  --topology-only
```

The probe reports `DIRECT` only if TMEM and persistent-state ownership match
for every logical element and the sparse exact product and BF16 state update
both match bitwise. Any missing, duplicate, cross-thread, or unmapped element
causes a nonzero exit. The probe intentionally contains no full-state shared
intermediate and uses a 128-participant named barrier inside its 192-thread CTA.

Use the ptxas output for spill loads/stores. `local_bytes_per_thread` and CUDA
function attributes are also included in the JSON result.

The JSON also reports `same_thread`, `same_warp_different_lane`,
`different_warp`, the 4x4 `warp_transfer_matrix`, a per-16x16-tile topology
grid, and inferred lane/warp maps. `D`, `W`, and `X` in the tile grid mean
same-thread, warp-local, and cross-warp ownership respectively. The optional
CSV contains the source and destination thread, warp, lane, and register slot
for every logical `(row,col)`.

Cross-warp permutations are classified as `SMALL_SCRATCH`, since an arbitrary
bijection can be realized with a 1,024-byte FP32 16x16 tile and two named
compute barriers per tile. The output also reports the 8,192-byte 16x128 strip
alternative, which needs 16 barriers for the full matrix. This is a mapping
feasibility result; the probe's 255-register diagnostic latency is not a
production performance result.

Before the numerical probe, the executable performs a host/static topology
sweep over the legal non-packed FP32 TMEM load families exposed by CUTLASS:

```text
SM100_TMEM_LOAD_32dp32b{1,2,4,8,16,32,64,128}x
SM100_TMEM_LOAD_16dp256b1x
SM100_TMEM_LOAD_16dp128b{1,2}x
SM100_TMEM_LOAD_16dp64b{1,2,4}x
SM100_TMEM_LOAD_16dp32b{1,2,4,8}x
```

The `_16b` forms are excluded because they change value packing and do not
match the FP32 accumulator. Each candidate must instantiate through
`make_tmem_copy` for the unchanged accumulator and produce a complete,
duplicate-free ownership map. Candidates are ranked by cross-warp count,
direct ownership, then register values per thread. These are static topology
records, so `local_bytes_per_thread` is reported as `-1`.

The final topology decision is `REGISTER_SHUFFLE_CANDIDATE`,
`MATERIALLY_REDUCED_CROSS_WARP`, or `ALL_TO_ALL_CROSS_WARP_INTRINSIC`.
