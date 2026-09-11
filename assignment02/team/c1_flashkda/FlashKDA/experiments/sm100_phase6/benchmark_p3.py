#!/usr/bin/env python3
"""Interleave normal P3 timings across recurrence lengths; no profiler calls."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def validate(record, args, steps):
    expected = {
        "stage": "p3", "implementation": "tcgen05_p3_recurrence",
        "correct": True, "state_unchanged": True, "decay_unchanged": True,
        "steps": steps, "batch": args.batch, "checked_ctas": args.batch,
        "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
        "input": "random", "state": "tagged", "decay_axis": "row",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"invalid {key}: expected {value!r}, got {record.get(key)!r}")
    if record.get("cc") not in ("10.0", "10.3"):
        raise ValueError("P3 timings require SM100/SM103")
    for key in ("launch_us", "update_us"):
        if not math.isfinite(record[key]) or record[key] <= 0:
            raise ValueError(f"invalid {key}")
    if not math.isclose(record["update_us"], record["launch_us"] / steps, abs_tol=1e-6):
        raise ValueError("update_us must equal launch_us / steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", nargs="?", type=Path,
                        default=HERE / ("tcgen05_p3_recurrence" + (".exe" if os.name == "nt" else "")))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--warmup", type=positive, default=30)
    parser.add_argument("--iters", type=positive, default=200)
    parser.add_argument("--batch", type=positive, default=1024)
    parser.add_argument("--seed", type=positive, default=2026)
    parser.add_argument("--output", type=Path, default=HERE / "build" / "p3_results.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(n < 1 or n > 16 for n in args.steps):
        parser.error("--steps requires distinct values in [1,16]")
    samples = []
    for repeat in range(args.repeats):
        # Reverse alternate rounds to reduce a consistent ordering bias.
        order = args.steps if repeat % 2 == 0 else list(reversed(args.steps))
        for steps in order:
            command = [str(args.executable.resolve()), "--steps", str(steps),
                       "--warmup", str(args.warmup), "--iters", str(args.iters),
                       "--batch", str(args.batch), "--seed", str(args.seed)]
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            process = subprocess.run(command, capture_output=True, text=True, timeout=300)
            attempt = {"repeat": repeat + 1, "command": command,
                       "returncode": process.returncode, "stdout": process.stdout,
                       "stderr": process.stderr}
            samples.append(attempt)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"P3 failed; stopped sweep. Log: {args.output}\n{process.stdout}{process.stderr}")
            records = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
            if len(records) != 1:
                raise ValueError("expected exactly one P3 JSON record")
            record = records[0]
            validate(record, args, steps)
            attempt["result"] = record
            args.output.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
            print(f"round={repeat + 1} steps={steps} correct=true "
                  f"launch_us={record['launch_us']:.3f} update_us={record['update_us']:.3f}", flush=True)
    if args.dry_run:
        return
    print("\n| steps | mean T(N), us | mean T(N)/N, us | stddev T(N)/N, us |")
    print("|---:|---:|---:|---:|")
    for steps in args.steps:
        values = [sample["result"]["launch_us"] for sample in samples
                  if sample["result"]["steps"] == steps]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"| {steps} | {mean:.3f} | {mean / steps:.3f} | {std / steps:.3f} |")


if __name__ == "__main__":
    main()
