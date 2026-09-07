#!/usr/bin/env python3
"""Build and compare the isolated Phase-6 P0 kernels.

Each executable emits one JSON record.  Exit code 2 means the source is still
the intentional assignment scaffold; other nonzero codes are real failures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = HERE / "build"
SOURCES = {"baseline": HERE / "baseline.cu", "tcgen05": HERE / "tcgen05.cu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["all", *SOURCES], default="all")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--arch", default="sm_100a")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def executable(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return BUILD / f"{name}{suffix}"


def selected(implementation: str) -> list[str]:
    return list(SOURCES) if implementation == "all" else [implementation]


def compile_command(args: argparse.Namespace, name: str) -> list[str]:
    return [
        args.nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        f"-arch={args.arch}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--warn-on-spills",
        f"-I{REPO / 'cutlass' / 'include'}",
        str(SOURCES[name]),
        "-o",
        str(executable(name)),
    ]


def run_command(args: argparse.Namespace, name: str) -> list[str]:
    return [
        str(executable(name)),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--batch",
        str(args.batch),
        "--seed",
        str(args.seed),
    ]


def quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def preflight(args: argparse.Namespace, names: list[str]) -> list[str]:
    problems: list[str] = []
    if not args.skip_build and shutil.which(args.nvcc) is None:
        problems.append(f"nvcc not found: {args.nvcc}")
    if not args.skip_build and not (REPO / "cutlass" / "include").is_dir():
        problems.append("CUTLASS submodule is absent; run git submodule update --init --recursive")
    for name in names:
        if not SOURCES[name].is_file():
            problems.append(f"missing source: {SOURCES[name]}")
        if args.skip_build and not executable(name).is_file():
            problems.append(f"missing executable for --skip-build: {executable(name)}")
    for value, flag in [
        (args.warmup, "--warmup"),
        (args.iters, "--iters"),
        (args.repeats, "--repeats"),
        (args.batch, "--batch"),
    ]:
        if value <= 0:
            problems.append(f"{flag} must be positive")
    return problems


def build(args: argparse.Namespace, name: str) -> dict[str, Any]:
    BUILD.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        compile_command(args, name), text=True, capture_output=True, check=False
    )
    log = process.stdout + process.stderr
    registers = [int(value) for value in re.findall(r"Used (\d+) registers", log)]
    smem = [int(value) for value in re.findall(r"(\d+) bytes smem", log)]
    if process.returncode != 0:
        sys.stderr.write(log)
        raise RuntimeError(f"build failed for {name}")
    return {
        "ptxas_max_registers": max(registers) if registers else None,
        "ptxas_max_static_smem": max(smem) if smem else None,
    }


def run_once(args: argparse.Namespace, name: str) -> tuple[int, dict[str, Any]]:
    process = subprocess.run(
        run_command(args, name), text=True, capture_output=True, check=False
    )
    if process.stderr:
        sys.stderr.write(process.stderr)
    records = []
    for line in process.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[{name}] {line}")
    if not records:
        raise RuntimeError(f"{name} emitted no JSON result")
    return process.returncode, records[-1]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(records[-1])
    result["launch_us"] = median([float(item["launch_us"]) for item in records])
    result["cta_ns"] = median([float(item["cta_ns"]) for item in records])
    return result


def print_table(results: list[dict[str, Any]]) -> None:
    headers = ["impl", "correct", "us/launch", "ns/CTA", "SMEM", "regs", "TMEM", "blocks/SM"]
    rows = []
    for item in results:
        status = "PASS" if item.get("correct") else ("TODO" if not item.get("implemented") else "FAIL")
        rows.append(
            [
                str(item["implementation"]),
                status,
                f"{item['launch_us']:.3f}",
                f"{item['cta_ns']:.3f}",
                str(item["static_smem_bytes"] + item["dynamic_smem_bytes"]),
                str(item["registers_per_thread"]),
                "-" if item["tmem_columns"] is None else str(item["tmem_columns"]),
                str(item["blocks_per_sm"]),
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> int:
    args = parse_args()
    names = selected(args.impl)
    problems = preflight(args, names)

    if args.dry_run:
        if problems:
            print("Preflight findings:")
            for problem in problems:
                print(f"  - {problem}")
        for name in names:
            if not args.skip_build:
                print(f"build {name}: {quote_command(compile_command(args, name))}")
            print(f"run   {name}: {quote_command(run_command(args, name))}")
        return 0

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    build_metadata: dict[str, dict[str, Any]] = {}
    if not args.skip_build:
        for name in names:
            build_metadata[name] = build(args, name)

    results: list[dict[str, Any]] = []
    unfinished = False
    for name in names:
        records = []
        for _ in range(args.repeats):
            returncode, record = run_once(args, name)
            if returncode == 2 and not record.get("implemented"):
                unfinished = True
            elif returncode != 0:
                raise RuntimeError(f"{name} failed with exit code {returncode}")
            records.append(record)
        result = summarize(records)
        result.update(build_metadata.get(name, {}))
        results.append(result)

    print_table(results)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if unfinished:
        print("\nP0 kernel body still marked TODO; timing is not a valid performance result.")
        return 2
    if not all(item.get("correct") for item in results):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
