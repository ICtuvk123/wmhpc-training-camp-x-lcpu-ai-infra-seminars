#!/usr/bin/env python3
"""Fit baseline/V1a K2 time = fixed + chunks * steady-state cost."""
import argparse
import json

DEFAULT = [
    (32, 127.559, 138.493),
    (64, 240.300, 246.361),
    (96, 352.898, 354.435),
    (128, 465.830, 462.413),
    (192, 691.230, 678.653),
    (256, 919.287, 896.175),
]


def fit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return intercept, slope, max(abs(x) for x in residuals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="JSON list of [chunks, baseline_us, v1a_us]")
    args = parser.parse_args()
    rows = json.loads(args.json) if args.json else DEFAULT
    chunks = [float(r[0]) for r in rows]
    baseline = fit(chunks, [float(r[1]) for r in rows])
    v1a = fit(chunks, [float(r[2]) for r in rows])
    fixed_delta = v1a[0] - baseline[0]
    per_chunk_delta = v1a[1] - baseline[1]
    crossover = fixed_delta / -per_chunk_delta if per_chunk_delta < 0 else None
    print(json.dumps({
        "baseline": {"fixed_us": baseline[0], "steady_us_per_chunk": baseline[1], "max_residual_us": baseline[2]},
        "v1a": {"fixed_us": v1a[0], "steady_us_per_chunk": v1a[1], "max_residual_us": v1a[2]},
        "v1a_minus_baseline": {"fixed_us": fixed_delta, "steady_us_per_chunk": per_chunk_delta},
        "fitted_crossover_chunks": crossover,
        "first_level_attribution": "FIXED_BOUNDARY_OR_SETUP_COST" if fixed_delta > 0 and per_chunk_delta < 0 else "PER_CHUNK_OR_MIXED",
    }, indent=2))


if __name__ == "__main__":
    main()
