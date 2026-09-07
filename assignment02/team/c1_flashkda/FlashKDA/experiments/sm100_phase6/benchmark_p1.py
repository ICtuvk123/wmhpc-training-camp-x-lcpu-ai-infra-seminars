#!/usr/bin/env python3
"""Build, validate and measure P1; optionally collect achieved occupancy with NCU."""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from check_p1 import parse_record, run_checks, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = HERE / "build"
OCCUPANCY_METRIC = "sm__warps_active.avg.pct_of_peak_sustained_active"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["all", "baseline", "tcgen05"], default="all")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--cutlass", type=Path, default=REPO / "cutlass")
    parser.add_argument("--arch", choices=["sm_100a", "sm_103a"], default="sm_103a")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--state", choices=["zero", "tagged", "random"], default="tagged")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ncu", help="NCU executable: collect achieved occupancy in a separate run")
    args = parser.parse_args()
    for key in ("batch", "warmup", "iters", "repeats", "seed"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.skip_build and args.build_only:
        parser.error("--skip-build and --build-only cannot be combined")
    return args


def executable(name):
    return BUILD / (f"{name}_p1" + (".exe" if os.name == "nt" else ""))


def compile_command(args, name):
    command = [args.nvcc, "-std=c++17", "-O3", "-lineinfo", f"-arch={args.arch}",
               "--expt-relaxed-constexpr", "--expt-extended-lambda",
               "--ptxas-options=-v,--warn-on-spills", f"-I{args.cutlass / 'include'}",
               str(HERE / f"{name}_p1.cu"), "-o", str(executable(name))]
    if os.name == "nt":
        command += ["-Xcompiler", "/utf-8"]
    return command


def run_command(args, name):
    return [str(executable(name)), "--batch", str(args.batch), "--warmup", str(args.warmup),
            "--iters", str(args.iters), "--seed", str(args.seed), "--state", args.state]


def profile_command(args, name):
    return [args.ncu, "--csv", "--page", "raw", "--metrics", OCCUPANCY_METRIC,
            "--kernel-name", f"regex:{name}_p1_kernel", "--launch-skip", str(args.warmup),
            "--launch-count", "1", *run_command(args, name)]


def build_metrics(log):
    def maximum(pattern):
        values = [int(x) for x in re.findall(pattern, log)]
        return max(values) if values else None
    return {
        "ptxas_registers": maximum(r"Used (\d+) registers"),
        "spill_store_bytes": maximum(r"(\d+) bytes spill stores"),
        "spill_load_bytes": maximum(r"(\d+) bytes spill loads"),
        "stack_frame_bytes": maximum(r"(\d+) bytes stack frame"),
    }


def parse_occupancy(output):
    rows = list(csv.reader(io.StringIO(output)))
    for i, header in enumerate(rows):
        if "Metric Name" not in header or "Metric Value" not in header:
            continue
        name_idx, value_idx = header.index("Metric Name"), header.index("Metric Value")
        values = []
        for row in rows[i + 1:]:
            if len(row) > max(name_idx, value_idx) and row[name_idx] == OCCUPANCY_METRIC:
                value = float(row[value_idx].replace(",", ""))
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError("invalid achieved occupancy")
                values.append(value)
        if len(values) == 1:
            return values[0]
        raise ValueError(f"expected one profiled launch, found {len(values)}")
    raise ValueError("NCU did not report achieved occupancy; inspect the saved log")


def table(results):
    columns = ["Metric", *[f"P1 {r['implementation']}" for r in results]]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    metrics = [
        ("Operation", lambda r: "S + K^T U"),
        ("Correct", lambda r: "PASS" if r["correct"] else "FAIL"),
        ("Latency (us)", lambda r: f"{r['launch_us']:.3f}"),
        ("SMEM (bytes/CTA)", lambda r: str(r["static_smem_bytes"] + r["dynamic_smem_bytes"])),
        ("Registers/thread", lambda r: str(r["registers_per_thread"])),
        ("Spill stores / loads (bytes)", lambda r: f"{r.get('spill_store_bytes')} / {r.get('spill_load_bytes')}"),
        ("TMEM columns/CTA", lambda r: str(r["tmem_columns"])),
        ("Estimated max CTAs/SM", lambda r: str(r["blocks_per_sm"])),
        ("Achieved occupancy (%)", lambda r: "unmeasured" if r.get("achieved_occupancy_pct") is None
         else f"{r['achieved_occupancy_pct']:.2f}"),
    ]
    for label, render in metrics:
        lines.append("| " + " | ".join([label, *[render(r) for r in results]]) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    names = ["baseline", "tcgen05"] if args.impl == "all" else [args.impl]
    if args.dry_run:
        for name in names:
            if not args.skip_build:
                print(subprocess.list2cmdline(compile_command(args, name)))
            if not args.build_only:
                print(f"regressions: {name} at batch=1,3")
                print(subprocess.list2cmdline(run_command(args, name)))
                if args.ncu:
                    print(subprocess.list2cmdline(profile_command(args, name)))
        return 0
    if not args.skip_build:
        if not shutil.which(args.nvcc):
            raise RuntimeError(f"nvcc not found: {args.nvcc}")
        if not (args.cutlass / "include" / "cute" / "tensor.hpp").is_file():
            raise RuntimeError(f"CUTLASS headers missing: {args.cutlass}")
    if args.ncu and not shutil.which(args.ncu):
        raise RuntimeError(f"NCU not found: {args.ncu}")
    BUILD.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name in names:
        log_path = BUILD / f"{name}_p1_build.log"
        if not args.skip_build:
            process = subprocess.run(compile_command(args, name), capture_output=True, text=True)
            log = process.stdout + process.stderr
            log_path.write_text(log, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"build failed: {log_path}\n{log}")
            metadata[name] = build_metrics(log)
            print(f"built {name}: {metadata[name]}", flush=True)
        elif not executable(name).is_file():
            raise RuntimeError(f"missing executable: {executable(name)}")
        # A skip-build run cannot attribute an old log to the current binary.
        else:
            metadata[name] = {}
    if args.build_only:
        return 0
    for name in names:
        run_checks(executable(name), name)
    samples = {name: [] for name in names}
    # Interleave implementations to reduce bias from long sequential runs.
    for _ in range(args.repeats):
        for name in names:
            process = subprocess.run(run_command(args, name), capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stdout + process.stderr)
            record = parse_record(process.stdout)
            validate(record, name, args.batch, ["--state", args.state, "--seed", str(args.seed)])
            samples[name].append(record)
    results = []
    for name in names:
        records = samples[name]
        result = dict(records[-1], **metadata[name])
        result["launch_us"] = statistics.median(r["launch_us"] for r in records)
        result["cta_ns"] = statistics.median(r["cta_ns"] for r in records)
        result["samples"] = records
        result["arch"] = args.arch
        if args.ncu:
            command = profile_command(args, name)
            process = subprocess.run(command, capture_output=True, text=True)
            output = process.stdout + "\n" + process.stderr
            (BUILD / f"{name}_p1_ncu.csv").write_text(output, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(output)
            validate(parse_record(process.stdout), name, args.batch,
                     ["--state", args.state, "--seed", str(args.seed)])
            result["achieved_occupancy_pct"] = parse_occupancy(output)
            result["ncu_command"] = command
        results.append(result)
    report = table(results)
    (BUILD / "p1_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (BUILD / "p1_results.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
