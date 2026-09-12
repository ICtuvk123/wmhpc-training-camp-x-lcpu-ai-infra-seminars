# V1a fixed-overhead attribution

This directory contains diagnostic-only tools. Production V1a and adaptive
dispatch are unchanged.

`fit_recurrence.py` fits the frozen B=4 sweep to
`time = fixed_cost + chunks * steady_state_cost`. Its built-in data produces:

```text
baseline: fixed 14.0154 us, steady 3.53257 us/chunk
V1a:      fixed 29.8879 us, steady 3.38177 us/chunk
delta:    fixed +15.8725 us, steady -0.150793 us/chunk
crossover: 105.26 chunks
```

This establishes that the observed crossover is caused by a fixed boundary or
setup cost, while V1a's recurrence body is cheaper per chunk.

`v1a_boundary_costs.cu` reproduces the exact V1a SM80-B persistent-fragment
coordinate ownership in a 192-thread, B*H-block launch. It times three kernels
in alternating order:

- `setup`: fragment/coordinate construction with an observable checksum;
- `final_store`: materialize a persistent fragment and store it to GMEM;
- `initial_load_plus_final_store`: exact GMEM-to-fragment-to-GMEM roundtrip.

The roundtrip is checked bitwise. `roundtrip - final_store` is an estimate of
the initial-load increment. These are attribution probes rather than additive
cycle accounting: independent kernels include launch overhead, and memory
operations can overlap in production.

Build and run on B300:

```bash
mkdir -p experiments/sm100_k2_integration/v1a_overhead/build
nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  --ptxas-options=-v,--register-usage-level=10,--warn-on-spills \
  -Icutlass/include -Icsrc \
  experiments/sm100_k2_integration/v1a_overhead/v1a_boundary_costs.cu \
  -o experiments/sm100_k2_integration/v1a_overhead/build/v1a_boundary_costs

experiments/sm100_k2_integration/v1a_overhead/build/v1a_boundary_costs \
  --batch 4 --heads 64 --warmup 30 --iters 200 --rounds 5

python experiments/sm100_k2_integration/v1a_overhead/fit_recurrence.py
```

Report ptxas spill loads/stores together with the emitted register and local
memory fields. Do not use this probe to change the frozen dispatch threshold.

## Boundary audit

The production source has two explicit V1a-only operations outside the
recurrence loop:

1. direct GMEM load into `state_regs[2][8]` before the loop;
2. direct store from the same fragments to GMEM after the loop.

The `state_mma`, thread slice, identity tensor, reference tensor, and fragment
types are layout/setup expressions shared by the generated kernel body. They
introduce address calculation where coordinates are consumed, but no explicit
memory transaction or synchronization on their own. There is no V1a-only
named barrier before or after the recurrence. The compute-group barrier at the
bottom of the loop is paid once per chunk by both baseline and V1a.

Consequently, if the measured boundary roundtrip accounts for the fitted
15.87-us intercept delta, classify the cause as **A: initial/final state
boundary movement**. If it is substantially smaller, inspect the generated
V1a prologue/epilogue and the scheduling effect of its 137-register live state
as **D: another fixed kernel cost**. The fitted negative per-chunk delta rules
out **C: per-chunk overhead** for this sweep.
