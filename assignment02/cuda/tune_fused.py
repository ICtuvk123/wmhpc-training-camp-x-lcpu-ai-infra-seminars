import csv
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "m5_lowprec/04_fused_rms_nvfp4.cu"
BIN = ROOT / "bin/m5_lowprec/04_fused_rms_nvfp4_tune"

BLOCKS = [128, 256, 512]
GRID_MULS = [1, 2, 4]

REPEATS = 3

pattern = re.compile(
    r"^\s*(\d+)\s+"
    r"(\d+)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)\s+"
    r"([\d.]+)x\s+"
    r"PASS\(bad=(\d+)\)"
)


def compile_kernel(block, grid_mul):
    cmd = [
        "nvcc",
        "-O2",
        "-std=c++17",
        "-I.",
        "--expt-relaxed-constexpr",
        "-gencode",
        "arch=compute_100f,code=sm_100f",
        f"-DFUSED_BLOCK={block}",
        f"-DFUSED_GRID_MUL={grid_mul}",
        "-o",
        str(BIN),
        str(SRC),
    ]

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
    )


def run_once():
    p = subprocess.run(
        [str(BIN)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    rows = []

    for line in p.stdout.splitlines():
        m = pattern.match(line)

        if not m:
            continue

        M = int(m.group(1))
        K = int(m.group(2))
        two_step_us = float(m.group(3))
        fused_us = float(m.group(4))
        speedup = float(m.group(5))
        bad = int(m.group(6))

        rows.append({
            "M": M,
            "K": K,
            "two_step_us": two_step_us,
            "fused_us": fused_us,
            "speedup": speedup,
            "bad": bad,
        })

    return rows


all_results = []

for block in BLOCKS:
    for grid_mul in GRID_MULS:
        print()
        print(
            f"=== BLOCK={block}, "
            f"GRID_MUL={grid_mul} ==="
        )

        compile_kernel(block, grid_mul)

        samples = {}

        for repeat in range(REPEATS):
            rows = run_once()

            for r in rows:
                key = (r["M"], r["K"])

                samples.setdefault(
                    key,
                    {
                        "two_step": [],
                        "fused": [],
                        "bad": [],
                    },
                )

                samples[key]["two_step"].append(
                    r["two_step_us"]
                )

                samples[key]["fused"].append(
                    r["fused_us"]
                )

                samples[key]["bad"].append(
                    r["bad"]
                )

        for (M, K), s in samples.items():
            two_step = statistics.median(
                s["two_step"]
            )

            fused = statistics.median(
                s["fused"]
            )

            speedup = two_step / fused

            row = {
                "block": block,
                "grid_mul": grid_mul,
                "M": M,
                "K": K,
                "two_step_us": two_step,
                "fused_us": fused,
                "speedup": speedup,
                "bad": max(s["bad"]),
            }

            all_results.append(row)

            print(
                f"M={M:<6} "
                f"K={K:<6} "
                f"fused={fused:8.2f} us "
                f"speedup={speedup:.2f}x"
            )


with open(
    ROOT / "tune_fused_raw.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "block",
            "grid_mul",
            "M",
            "K",
            "two_step_us",
            "fused_us",
            "speedup",
            "bad",
        ],
    )

    writer.writeheader()
    writer.writerows(all_results)


best = {}

for r in all_results:
    key = (r["M"], r["K"])

    if (
        key not in best
        or r["fused_us"] < best[key]["fused_us"]
    ):
        best[key] = r


print()
print("========== BEST FUSED ==========")

for key in sorted(best):
    r = best[key]

    print(
        f"M={r['M']:<6} "
        f"K={r['K']:<6} "
        f"BLOCK={r['block']:<3} "
        f"GRID={r['grid_mul']}xSM "
        f"fused={r['fused_us']:8.2f} us "
        f"speedup={r['speedup']:.2f}x"
    )


with open(
    ROOT / "tune_fused_best.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "block",
            "grid_mul",
            "M",
            "K",
            "two_step_us",
            "fused_us",
            "speedup",
            "bad",
        ],
    )

    writer.writeheader()

    for key in sorted(best):
        writer.writerow(best[key])
