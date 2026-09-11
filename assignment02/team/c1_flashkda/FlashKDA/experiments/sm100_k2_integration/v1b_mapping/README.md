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
  --warmup 30 --iters 200
```

The probe reports `DIRECT` only if TMEM and persistent-state ownership match
for every logical element and the sparse exact product and BF16 state update
both match bitwise. Any missing, duplicate, cross-thread, or unmapped element
causes a nonzero exit. The probe intentionally contains no full-state shared
intermediate and uses a 128-participant named barrier inside its 192-thread CTA.

Use the ptxas output for spill loads/stores. `local_bytes_per_thread` and CUDA
function attributes are also included in the JSON result.
