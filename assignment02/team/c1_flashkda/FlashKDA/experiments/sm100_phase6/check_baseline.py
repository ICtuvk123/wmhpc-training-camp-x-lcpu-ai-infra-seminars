#!/usr/bin/env python3
"""Run numerical/layout regressions on the B300 (SM103) baseline executable."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def cases():
    # All reduction coordinates, all eight N tiles, and both sides of the
    # 8/16/32 boundaries. Off-diagonal probes expose tile-local transposes.
    yield "origin", ["--input", "one-hot", "--m", "0", "--k", "0", "--n", "0"]
    edges = [0, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 111, 112, 119, 120, 127]
    for k, m in enumerate(edges):
        n = (k * 17 + 5) % 128
        yield f"probe-{m}-{k}-{n}", [
            "--input", "one-hot", "--m", str(m), "--k", str(k), "--n", str(n)
        ]
    yield "last-element", ["--input", "one-hot", "--m", "127", "--k", "15", "--n", "127"]
    yield "tagged", ["--input", "tagged"]
    for seed in (1, 2026, 98765):
        yield f"random-{seed}", ["--seed", str(seed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--sanitizer", help="Path to compute-sanitizer; omitted for numerical checks")
    parser.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batch):
        parser.error("--batch values must be positive")
    executable = args.executable.resolve()
    if not args.dry_run and not executable.is_file():
        parser.error(f"executable not found: {executable}")

    count = 0
    for batch in args.batch:
        for label, flags in cases():
            command = [str(executable), "--warmup", "1", "--iters", "1", "--batch", str(batch), *flags]
            if args.sanitizer:
                command = [args.sanitizer, "--tool", args.tool, "--error-exitcode", "99", *command]
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            records = []
            for line in process.stdout.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
            record = records[-1] if records else {}
            passed = (
                process.returncode == 0
                and record.get("correct") is True
                and record.get("implementation") == "baseline"
                and record.get("cc") == "10.3"
                and all(isinstance(record.get(key), (int, float))
                        and math.isfinite(record[key]) for key in ("max_abs", "max_rel"))
            )
            if label.startswith("random-"):
                passed = passed and record.get("input") == "random"
            else:
                expected_input = "tagged" if label == "tagged" else "one-hot"
                passed = passed and record.get("input") == expected_input and record.get("max_abs") == 0
                if expected_input == "one-hot":
                    values = dict(zip(flags[::2], flags[1::2]))
                    passed = passed and all(
                        record.get(f"probe_{axis}") == int(values[f"--{axis}"])
                        for axis in ("m", "k", "n")
                    )
            if not passed:
                print(f"FAIL batch={batch} {label}")
                print(process.stdout)
                print(process.stderr)
                return 1
            print(f"PASS batch={batch} {label} max_abs={record['max_abs']}")
            count += 1
    if not args.dry_run:
        print(f"All {count} B300/SM103 baseline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
