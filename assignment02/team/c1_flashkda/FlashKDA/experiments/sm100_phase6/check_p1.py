#!/usr/bin/env python3
"""Check P1 state preservation and MMA layouts on SM100/SM103."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def cases():
    # Zero product isolates the state transfer, including full FP32 precision.
    for state in ("zero", "tagged", "random"):
        yield f"state-only-{state}", ["--input", "zero", "--state", state]
    for state in ("zero", "tagged"):
        yield f"dense-{state}", ["--input", "tagged", "--state", state]
    edges = [0, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 111, 112, 119, 120, 127]
    for k, m in enumerate(edges):
        n = (17 * k + 5) % 128
        yield f"probe-{m}-{k}-{n}", [
            "--input", "one-hot", "--state", "tagged",
            "--m", str(m), "--k", str(k), "--n", str(n),
        ]
    for seed in (1, 2026, 98765):
        for state in ("zero", "tagged", "random"):
            yield f"random-{seed}-{state}", [
                "--input", "random", "--state", state, "--seed", str(seed)
            ]


def parse_record(output):
    records = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "implementation" in item:
            records.append(item)
    if len(records) != 1:
        raise ValueError(f"expected one kernel result, found {len(records)}")
    return records[0]


def validate(record, implementation, batch, flags):
    expected = dict(zip(flags[::2], flags[1::2]))
    input_kind = expected.get("--input", "random")
    state = expected.get("--state", "tagged")
    exact = input_kind == "zero" or (input_kind != "random" and state != "random")
    valid = (
        record.get("stage") == "p1"
        and record.get("implementation") == implementation
        and record.get("cc") in ("10.0", "10.3")
        and record.get("correct") is True
        and record.get("state_unchanged") is True
        and record.get("batch") == batch
        and record.get("checked_ctas") == batch
        and record.get("input") == input_kind
        and record.get("state") == state
        and record.get("exact") is exact
        and record.get("seed") == int(expected.get("--seed", "2026"))
    )
    for key in ("max_abs", "max_rel", "launch_us", "cta_ns"):
        value = record.get(key)
        valid = valid and type(value) in (int, float) and math.isfinite(value) and value >= 0
    if exact:
        valid = valid and record.get("max_abs") == 0
    if input_kind == "one-hot":
        valid = valid and all(record.get(f"probe_{axis}") == int(expected[f"--{axis}"])
                              for axis in ("m", "k", "n"))
    if not valid:
        raise ValueError(f"P1 validation failed: {record}")


def run_checks(executable, implementation, batches=(1, 3), sanitizer=None,
               tool="memcheck", dry_run=False):
    count = 0
    for batch in batches:
        for label, flags in cases():
            # Multiple launches expose accidental in-place accumulation.
            command = [str(executable), "--batch", str(batch), "--warmup", "2",
                       "--iters", "3", *flags]
            if sanitizer:
                command = [sanitizer, "--tool", tool, "--error-exitcode", "99", *command]
            if dry_run:
                print(subprocess.list2cmdline(command))
                continue
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            if process.returncode:
                raise RuntimeError(f"FAIL {implementation} batch={batch} {label}\n"
                                   f"{process.stdout}\n{process.stderr}")
            validate(parse_record(process.stdout), implementation, batch, flags)
            count += 1
    if not dry_run:
        print(f"PASS {implementation}: {count} P1 regressions", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--impl", required=True, choices=["baseline", "tcgen05"])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--sanitizer")
    parser.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(b <= 0 for b in args.batch):
        parser.error("--batch must be positive")
    executable = args.executable.resolve()
    if not args.dry_run and not executable.is_file():
        parser.error(f"missing executable: {executable}")
    run_checks(executable, args.impl, args.batch, args.sanitizer, args.tool, args.dry_run)


if __name__ == "__main__":
    main()
