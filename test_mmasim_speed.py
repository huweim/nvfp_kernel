#!/usr/bin/env python3
"""
Reference speed probe for MMA-Sim tcgen05 block-scale simulation.

This benchmarks the CPU-side simulator call:
    mmasim.simulator.nv_ptx.tcgen05mma_block_scale

It is intentionally separate from our BEAM/Triton profiler because MMA-Sim is
an instruction-level software simulator with very different performance goals.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MMA-Sim tcgen05 block-scale simulator latency.")
    parser.add_argument(
        "--mmasim-root",
        type=str,
        default="./MMA-Sim",
        help="Path to the cloned MMA-Sim repo root",
    )
    parser.add_argument("--arch", type=str, default="Blackwell", help="Simulator architecture")
    parser.add_argument(
        "--qualifier",
        type=str,
        default="m128n256k64.block16.f32.e2m1.e2m1.ue4m3",
        help="tcgen05mma_block_scale qualifier",
    )
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--scale-value",
        type=float,
        default=1.0,
        help="Constant positive block scale value used for A/B scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mmasim_tcgen05_block_scale",
        help="Filename prefix for CSV/JSON outputs",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Skip JSON output and only write CSV",
    )
    return parser.parse_args()


def load_simulator(mmasim_root: Path):
    if not mmasim_root.exists():
        raise FileNotFoundError(f"MMA-Sim root not found: {mmasim_root}")
    sys.path.insert(0, str(mmasim_root.resolve()))
    from mmasim.simulator.nv_ptx import tcgen05mma_block_scale

    return tcgen05mma_block_scale


def build_inputs(sim, scale_value: float) -> tuple[torch.Tensor, ...]:
    m, n, k = sim.m, sim.n, sim.k
    packing = sim.packing
    block_size = sim.block_size

    A = torch.randint(0, 256, (m, k // packing), dtype=sim.a_type)
    B = torch.randint(0, 256, (k // packing, n), dtype=sim.b_type)
    C = torch.zeros((m, n), dtype=sim.c_type)
    scale_A = torch.full((m, k // block_size), scale_value, dtype=sim.s_type)
    scale_B = torch.full((k // block_size, n), scale_value, dtype=sim.s_type)
    return A, B, C, scale_A, scale_B


def benchmark(fn, warmup: int, iters: int) -> tuple[dict[str, float], torch.Tensor]:
    for _ in range(warmup):
        _ = fn()

    times_sec = []
    last_output = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last_output = fn()
        times_sec.append(time.perf_counter() - t0)

    avg_sec = statistics.mean(times_sec)
    stats = {
        "avg_sec": avg_sec,
        "median_sec": statistics.median(times_sec),
        "min_sec": min(times_sec),
        "max_sec": max(times_sec),
        "std_sec": statistics.pstdev(times_sec) if len(times_sec) > 1 else 0.0,
        "avg_ms": avg_sec * 1000.0,
        "median_ms": statistics.median(times_sec) * 1000.0,
        "min_ms": min(times_sec) * 1000.0,
        "max_ms": max(times_sec) * 1000.0,
        "std_ms": (statistics.pstdev(times_sec) if len(times_sec) > 1 else 0.0) * 1000.0,
    }
    return stats, last_output


def make_output_paths(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    timestamp = time.strftime("%m%d%H%M")
    return output_dir / f"{prefix}_{timestamp}.csv", output_dir / f"{prefix}_{timestamp}.json"


def write_csv(path: Path, row: dict) -> None:
    fieldnames = [
        "arch",
        "qualifier",
        "m",
        "n",
        "k",
        "block_size",
        "packing",
        "a_type",
        "b_type",
        "c_type",
        "s_type",
        "iters",
        "warmup",
        "scale_value",
        "avg_sec",
        "median_sec",
        "min_sec",
        "max_sec",
        "std_sec",
        "avg_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "output_sum",
        "output_abs_mean",
        "mmasim_root",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    mmasim_root = Path(args.mmasim_root)
    tcgen05mma_block_scale = load_simulator(mmasim_root)
    sim = tcgen05mma_block_scale(args.arch, args.qualifier)

    A, B, C, scale_A, scale_B = build_inputs(sim, args.scale_value)
    fn = lambda: sim(A, B, C, scale_A, scale_B)

    print("=" * 96)
    print("MMA-Sim tcgen05mma_block_scale Speed Probe")
    print(f"mmasim_root={mmasim_root}")
    print(f"arch={args.arch}")
    print(f"qualifier={args.qualifier}")
    print(f"shape=(m={sim.m}, n={sim.n}, k={sim.k}), block_size={sim.block_size}, packing={sim.packing}")
    print(f"iters={args.iters}, warmup={args.warmup}, scale_value={args.scale_value}")
    print("=" * 96)

    stats, out = benchmark(fn, warmup=args.warmup, iters=args.iters)
    out_f32 = out.float()

    row = {
        "arch": args.arch,
        "qualifier": args.qualifier,
        "m": sim.m,
        "n": sim.n,
        "k": sim.k,
        "block_size": sim.block_size,
        "packing": sim.packing,
        "a_type": str(sim.a_type).replace("torch.", ""),
        "b_type": str(sim.b_type).replace("torch.", ""),
        "c_type": str(sim.c_type).replace("torch.", ""),
        "s_type": str(sim.s_type).replace("torch.", ""),
        "iters": args.iters,
        "warmup": args.warmup,
        "scale_value": args.scale_value,
        "output_sum": float(out_f32.sum().item()),
        "output_abs_mean": float(out_f32.abs().mean().item()),
        "mmasim_root": str(mmasim_root.resolve()),
        **stats,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = make_output_paths(output_dir, args.output_prefix)
    write_csv(csv_path, row)
    if not args.skip_json:
        json_path.write_text(json.dumps(row, indent=2) + "\n")

    print(f"avg={row['avg_sec']:.4f}s ({row['avg_ms']:.2f} ms)")
    print(f"median={row['median_sec']:.4f}s ({row['median_ms']:.2f} ms)")
    print(f"min={row['min_sec']:.4f}s, max={row['max_sec']:.4f}s, std={row['std_sec']:.4f}s")
    print(f"output_sum={row['output_sum']:.6f}, output_abs_mean={row['output_abs_mean']:.6f}")
    print(f"[Save] CSV written to {csv_path}")
    if not args.skip_json:
        print(f"[Save] JSON written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
