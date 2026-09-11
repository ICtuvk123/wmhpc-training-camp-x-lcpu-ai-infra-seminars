"""Final public-API benchmark for adaptive production K2 dispatch."""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys


WORKLOADS = {
    "b1_t8192": (1, 8192, "v1a"),
    "b4_t2048": (4, 2048, "v1a"),
    "b8_t1024": (8, 1024, "baseline"),
}
MODES = ("baseline", "v1a", "auto")
HEADS = 64
DIM = 128
CHUNK = 16
SEED = 20260911


def tensor_digest(tensor):
    import torch

    raw = tensor.detach().contiguous().view(-1).view(torch.uint16)
    return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()


def run_worker(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA-enabled PyTorch")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("benchmark must run on SM103/B300")

    import flash_kda

    batch, tokens, expected_auto = WORKLOADS[args.workload]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    shape = (batch, tokens, HEADS, DIM)
    q, k, v, g = [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(4)
    ]
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    initial = torch.randn(
        (batch, HEADS, DIM, DIM), dtype=torch.bfloat16, device="cuda"
    )
    initial_digest = tensor_digest(initial)
    a_log = torch.rand(HEADS, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(HEADS, DIM, dtype=torch.float32, device="cuda")
    output = torch.empty_like(q)
    final_state = torch.empty_like(initial)

    def forward():
        flash_kda.fwd(
            q, k, v, g, beta, 1.0 / math.sqrt(DIM), output,
            A_log=a_log, dt_bias=dt_bias, lower_bound=-5.0,
            initial_state=initial, final_state=final_state,
        )

    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for start, end in zip(starts, ends):
        start.record()
        forward()
        end.record()
    torch.cuda.synchronize()
    latencies_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]

    if tensor_digest(initial) != initial_digest:
        raise RuntimeError("initial state was modified")
    if not torch.isfinite(output).all() or not torch.isfinite(final_state).all():
        raise RuntimeError("forward produced non-finite output or final state")

    record = {
        "workload": args.workload,
        "mode": os.environ["FLASH_KDA_K2_IMPL"],
        "batch": batch,
        "tokens": tokens,
        "heads": HEADS,
        "dim": DIM,
        "chunks_per_sequence": tokens // CHUNK,
        "expected_auto": expected_auto,
        "warmup": args.warmup,
        "iters": args.iters,
        "mean_us": statistics.fmean(latencies_us),
        "median_us": statistics.median(latencies_us),
        "output_sha256": tensor_digest(output),
        "final_state_sha256": tensor_digest(final_state),
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def run_parent(args):
    script = os.path.abspath(__file__)
    results = {}
    for workload in WORKLOADS:
        results[workload] = {}
        for mode in MODES:
            env = os.environ.copy()
            env["FLASH_KDA_K2_IMPL"] = mode
            command = [
                sys.executable, script, "--worker", "--workload", workload,
                "--warmup", str(args.warmup), "--iters", str(args.iters),
            ]
            completed = subprocess.run(
                command, env=env, check=True, text=True,
                stdout=subprocess.PIPE, stderr=None,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            record = json.loads(lines[-1])
            results[workload][mode] = record
            print(json.dumps(record, sort_keys=True), flush=True)

        baseline = results[workload]["baseline"]
        for mode in ("v1a", "auto"):
            candidate = results[workload][mode]
            if candidate["output_sha256"] != baseline["output_sha256"]:
                raise RuntimeError(f"{workload} {mode}: output differs from baseline")
            if candidate["final_state_sha256"] != baseline["final_state_sha256"]:
                raise RuntimeError(f"{workload} {mode}: final_state differs from baseline")

    print("\nworkload    baseline mean/median    v1a mean/median        auto mean/median        auto expected")
    for workload, (_, _, expected) in WORKLOADS.items():
        row = results[workload]
        print(
            f"{workload:<11}"
            f"{row['baseline']['mean_us']:9.3f}/{row['baseline']['median_us']:9.3f} us  "
            f"{row['v1a']['mean_us']:9.3f}/{row['v1a']['median_us']:9.3f} us  "
            f"{row['auto']['mean_us']:9.3f}/{row['auto']['median_us']:9.3f} us  "
            f"{expected}"
        )
    print("\nExact output and final_state equality: PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    if args.worker and args.workload is None:
        parser.error("worker requires --workload")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        run_parent(parsed)
