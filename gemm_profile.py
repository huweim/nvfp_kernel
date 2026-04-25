#!/usr/bin/env python3
"""
GEMM latency profiler for NVFP4 case-study tables.

Benchmarked methods:
1. real_nvfp4: native FP4 GEMM via ops.cutlass_scaled_fp4_mm (RTX 5090 only)
2. pseudo_fp16: regular FP16 GEMM via torch.nn.functional.linear
3. beam_base: unfused BEAM emulation with explicit CUDA-core stage-1
4. beam_fused: Triton-fused Stage3+4 BEAM emulation via MMAEngine.emulation_scaled_fp4_mm_triton
5. beam_234fusion: Triton-fused Stage2+3+4 with einsum Stage-1
6. beam_234fusion_bmm: Triton-fused Stage2+3+4 with batched-matmul Stage-1 (e2e default fast path)

Input quantization is prepared once per K and excluded from timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from nvfp import ops
from emulation.core import MMAEngine
from emulation.utils import DataGenerator


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
ALPHA_SCALE = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX
DEFAULT_KS = (512, 1024, 2048, 4096, 8192)
DEFAULT_METHODS = ("real_nvfp4", "pseudo_fp16", "beam_base", "beam_fused", "beam_234fusion", "beam_234fusion_bmm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile NVFP4 GEMM latency across methods and K sizes.")
    parser.add_argument("--m", type=int, default=1024, help="M dimension")
    parser.add_argument("--n", type=int, default=2048, help="N dimension")
    parser.add_argument(
        "--k-values",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated K values",
    )
    parser.add_argument("--iters", type=int, default=1000, help="Measured iterations per method")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations per method")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Dense input dtype for pseudo GEMM and quantization source tensors",
    )
    parser.add_argument("--dist-a", type=str, default="normal", help="Input distribution for A")
    parser.add_argument("--dist-b", type=str, default="normal", help="Input distribution for B")
    parser.add_argument(
        "--run-real",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Whether to benchmark native real NVFP4 GEMM",
    )
    parser.add_argument("--w-stage3", type=int, default=36, help="BEAM stage-3 accumulation bits")
    parser.add_argument("--w-stage4", type=int, default=36, help="BEAM stage-4 accumulation bits")
    parser.add_argument("--m-chunk-size", type=int, default=128, help="BEAM M chunk size")
    parser.add_argument("--triton-block-size", type=int, default=256, help="Triton block size")
    parser.add_argument(
        "--disable-stage34-fusion",
        action="store_true",
        help="Disable fused Triton stage3+4 path",
    )
    parser.add_argument(
        "--disable-stage3-triton",
        action="store_true",
        help="Disable Triton stage-3 reduction",
    )
    parser.add_argument("--device", type=str, default="cuda", help="CUDA device")
    parser.add_argument("--device-label", type=str, default=None, help="Optional short label for output naming")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output filename prefix",
    )
    parser.add_argument(
        "--skip-json",
        action="store_true",
        help="Skip JSON output and only write CSV",
    )
    return parser.parse_args()


def parse_k_values(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one K value is required.")
    for k in values:
        if k % 64 != 0:
            raise ValueError(f"K={k} is invalid: expected a multiple of 64 for NVFP4 reduction structure.")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def sanitize_label(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "gpu"


def get_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    amax = torch.abs(tensor).max().to(torch.float32).item()
    denom = amax if amax > 0 else 1.0
    return torch.tensor([ALPHA_SCALE / denom], device=tensor.device, dtype=torch.float32)


@contextmanager
def nvfp_backend(backend: str | None):
    old = os.environ.get("NVFP_GEMM_BACKEND")
    if backend is None:
        os.environ.pop("NVFP_GEMM_BACKEND", None)
    else:
        os.environ["NVFP_GEMM_BACKEND"] = backend
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("NVFP_GEMM_BACKEND", None)
        else:
            os.environ["NVFP_GEMM_BACKEND"] = old


def real_backend_available() -> tuple[bool, str]:
    try:
        with nvfp_backend("real"):
            active = ops.get_active_backend()
        return active == "real", active
    except Exception:
        return False, "unavailable"


def build_inputs(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    dist_a: str,
    dist_b: str,
    device: str,
    quant_backend: str,
) -> dict[str, torch.Tensor | float | str]:
    a = DataGenerator.get_random_tensor((m, k), dist_a, device=device, dtype=dtype).contiguous()
    b = DataGenerator.get_random_tensor((n, k), dist_b, device=device, dtype=dtype).contiguous()

    a_gs = get_global_scale(a)
    b_gs = get_global_scale(b)
    alpha = torch.tensor([1.0 / (a_gs.item() * b_gs.item())], device=device, dtype=torch.float32)

    with nvfp_backend(quant_backend):
        active_backend = ops.get_active_backend()
        a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)

    return {
        "a_dense": a,
        "b_dense": b,
        "a_fp4": a_fp4,
        "b_fp4": b_fp4,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "alpha": alpha,
        "quant_backend": active_backend,
    }


def benchmark_callable(fn, warmup: int, iters: int) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            _ = fn()
            ends[i].record()
        torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    avg_ms = statistics.mean(times_ms)
    return {
        "avg_ms": avg_ms,
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "times_ms": times_ms,
    }


def build_method_callables(
    prepared: dict[str, torch.Tensor | float | str],
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    run_real: bool,
    w_stage3: int,
    w_stage4: int,
    m_chunk_size: int,
    triton_block_size: int,
    use_stage3_triton: bool,
    fuse_stage34: bool,
):
    a_dense = prepared["a_dense"]
    b_dense = prepared["b_dense"]
    a_fp4 = prepared["a_fp4"]
    b_fp4 = prepared["b_fp4"]
    scale_a = prepared["scale_a"]
    scale_b = prepared["scale_b"]
    alpha = prepared["alpha"]

    def pseudo_fp16():
        return F.linear(a_dense, b_dense)

    def beam_fused():
        return MMAEngine.emulation_scaled_fp4_mm_triton(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            alpha,
            m,
            n,
            k,
            W_stage3=w_stage3,
            W_stage4=w_stage4,
            m_chunk_size=m_chunk_size,
            triton_block_size=triton_block_size,
            triton_use_stage3=use_stage3_triton,
            triton_fuse_stage34=fuse_stage34,
        )

    def beam_234fusion():
        return MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            alpha,
            m,
            n,
            k,
            W_stage3=w_stage3,
            W_stage4=w_stage4,
            m_chunk_size=m_chunk_size,
            triton_block_size=triton_block_size,
        )

    def beam_234fusion_bmm():
        return MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused_bmm(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            alpha,
            m,
            n,
            k,
            W_stage3=w_stage3,
            W_stage4=w_stage4,
            m_chunk_size=m_chunk_size,
            triton_block_size=triton_block_size,
        )

    def beam_base():
        return MMAEngine.emulation_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            alpha,
            m,
            n,
            k,
            W_stage3=w_stage3,
            W_stage4=w_stage4,
            m_chunk_size=m_chunk_size,
            stage1_impl="cuda_core",
        )

    def real_nvfp4():
        with nvfp_backend("real"):
            return ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, out_dtype)

    methods = {
        "pseudo_fp16": pseudo_fp16,
        "beam_base": beam_base,
        "beam_fused": beam_fused,
        "beam_234fusion": beam_234fusion,
        "beam_234fusion_bmm": beam_234fusion_bmm,
    }
    if run_real:
        methods["real_nvfp4"] = real_nvfp4
    return methods


def tfops(m: int, n: int, k: int, latency_ms: float) -> float:
    return (2.0 * m * n * k) / (latency_ms / 1000.0) / 1e12


def choose_baseline(method_rows: list[dict]) -> str:
    methods = {row["method"] for row in method_rows if row["available"]}
    if not methods:
        raise RuntimeError("No available methods were successfully benchmarked for this K.")
    if "real_nvfp4" in methods:
        return "real_nvfp4"
    if "pseudo_fp16" in methods:
        return "pseudo_fp16"
    return next(iter(methods))


def print_summary(rows: list[dict], device_name: str, device_label: str, m: int, n: int) -> None:
    print("=" * 108)
    print(f"GEMM Profile Summary | device={device_name} | label={device_label} | M={m} N={n}")
    print("=" * 108)
    header = f"{'K':>6}  {'Method':<12}  {'Available':<9}  {'Avg(ms)':>10}  {'Median':>10}  {'TFLOPS':>10}  {'Ratio':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        ratio = "--" if row["ratio_to_baseline"] is None else f"{row['ratio_to_baseline']:.2f}x"
        avg_ms = "--" if row["avg_ms"] is None else f"{row['avg_ms']:.3f}"
        med_ms = "--" if row["median_ms"] is None else f"{row['median_ms']:.3f}"
        tflops = "--" if row["tflops"] is None else f"{row['tflops']:.3f}"
        print(
            f"{row['k']:>6}  {row['method']:<12}  {str(row['available']):<9}  "
            f"{avg_ms:>10}  {med_ms:>10}  {tflops:>10}  {ratio:>8}"
        )
    print("=" * 108)


def make_output_paths(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    timestamp = time.strftime("%m%d%H%M")
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    return csv_path, json_path


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "device_label",
        "device_name",
        "sm",
        "m",
        "n",
        "k",
        "method",
        "available",
        "reason",
        "avg_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "tflops",
        "ratio_to_baseline",
        "baseline_method",
        "iters",
        "warmup",
        "dtype",
        "dist_a",
        "dist_b",
        "quant_backend",
        "run_real_mode",
        "stage1_impl",
        "w_stage3",
        "w_stage4",
        "m_chunk_size",
        "triton_block_size",
        "triton_use_stage3",
        "triton_fuse_stage34",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GEMM profiling.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    k_values = parse_k_values(args.k_values)
    dtype = dtype_from_name(args.dtype)
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    sm = f"sm_{capability[0]}{capability[1]}"
    device_label = args.device_label or sanitize_label(device_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{device_label}_gemm_profile"
    csv_path, json_path = make_output_paths(output_dir, prefix)

    real_available, _ = real_backend_available()
    if args.run_real == "on" and not real_available:
        raise RuntimeError("Real NVFP4 GEMM requested, but the real backend is unavailable on this machine.")
    run_real = (args.run_real == "on") or (args.run_real == "auto" and real_available)

    torch_dtype_name = str(dtype).replace("torch.", "")
    print("=" * 108)
    print("NVFP4 GEMM Performance Profiling")
    print(f"device={device_name} ({sm})")
    print(f"device_label={device_label}")
    print(f"M={args.m}, N={args.n}, K={k_values}")
    print(f"dtype={torch_dtype_name}, dist_a={args.dist_a}, dist_b={args.dist_b}")
    print(f"iters={args.iters}, warmup={args.warmup}")
    print("methods=" + ", ".join(DEFAULT_METHODS))
    print(
        "beam_config="
        f"W3={args.w_stage3}, W4={args.w_stage4}, m_chunk_size={args.m_chunk_size}, "
        f"triton_block_size={args.triton_block_size}, "
        f"stage3_triton={not args.disable_stage3_triton}, "
        f"fuse_stage34={not args.disable_stage34_fusion}"
    )
    print(f"run_real={run_real} (requested={args.run_real}, real_available={real_available})")
    print("=" * 108)

    all_rows: list[dict] = []
    for k in k_values:
        quant_backend = "real" if run_real else "emulation"
        prep_t0 = time.perf_counter()
        prepared = build_inputs(
            m=args.m,
            n=args.n,
            k=k,
            dtype=dtype,
            dist_a=args.dist_a,
            dist_b=args.dist_b,
            device=args.device,
            quant_backend=quant_backend,
        )
        torch.cuda.synchronize()
        prep_ms = (time.perf_counter() - prep_t0) * 1000.0
        print(f"[Prepare] K={k} input build + quantization (excluded): {prep_ms:.2f} ms, quant_backend={prepared['quant_backend']}")

        method_rows: list[dict] = []
        methods = build_method_callables(
            prepared=prepared,
            m=args.m,
            n=args.n,
            k=k,
            out_dtype=dtype,
            run_real=run_real,
            w_stage3=args.w_stage3,
            w_stage4=args.w_stage4,
            m_chunk_size=args.m_chunk_size,
            triton_block_size=args.triton_block_size,
            use_stage3_triton=not args.disable_stage3_triton,
            fuse_stage34=not args.disable_stage34_fusion,
        )
        stage1_impl_map = {
            "real_nvfp4": None,
            "pseudo_fp16": None,
            "beam_base": "cuda_core",
            "beam_fused": "einsum",
            "beam_234fusion": "einsum",
            "beam_234fusion_bmm": "bmm",
        }

        for method in DEFAULT_METHODS:
            row = {
                "device_label": device_label,
                "device_name": device_name,
                "sm": sm,
                "m": args.m,
                "n": args.n,
                "k": k,
                "method": method,
                "available": method in methods,
                "reason": None,
                "avg_ms": None,
                "median_ms": None,
                "min_ms": None,
                "max_ms": None,
                "std_ms": None,
                "tflops": None,
                "ratio_to_baseline": None,
                "baseline_method": None,
                "iters": args.iters,
                "warmup": args.warmup,
                "dtype": args.dtype,
                "dist_a": args.dist_a,
                "dist_b": args.dist_b,
                "quant_backend": prepared["quant_backend"],
                "run_real_mode": args.run_real,
                "stage1_impl": stage1_impl_map[method],
                "w_stage3": args.w_stage3,
                "w_stage4": args.w_stage4,
                "m_chunk_size": args.m_chunk_size,
                "triton_block_size": args.triton_block_size,
                "triton_use_stage3": not args.disable_stage3_triton,
                "triton_fuse_stage34": not args.disable_stage34_fusion,
            }

            if method not in methods:
                row["reason"] = "real_backend_unavailable" if method == "real_nvfp4" else "disabled"
                method_rows.append(row)
                continue

            print(f"[Benchmark] K={k} method={method} ...")
            try:
                stats = benchmark_callable(methods[method], warmup=args.warmup, iters=args.iters)
            except Exception as exc:
                row["available"] = False
                row["reason"] = f"runtime_error: {type(exc).__name__}: {exc}"
                print(f"[Skip] K={k} method={method} unavailable: {exc}")
                method_rows.append(row)
                continue
            row["avg_ms"] = stats["avg_ms"]
            row["median_ms"] = stats["median_ms"]
            row["min_ms"] = stats["min_ms"]
            row["max_ms"] = stats["max_ms"]
            row["std_ms"] = stats["std_ms"]
            row["tflops"] = tfops(args.m, args.n, k, stats["avg_ms"])
            method_rows.append(row)

        try:
            baseline_method = choose_baseline(method_rows)
            baseline_latency = next(
                row["avg_ms"] for row in method_rows if row["method"] == baseline_method and row["avg_ms"] is not None
            )
            for row in method_rows:
                row["baseline_method"] = baseline_method
                if row["avg_ms"] is not None:
                    row["ratio_to_baseline"] = row["avg_ms"] / baseline_latency
        except Exception as exc:
            print(f"[Warn] K={k} baseline ratio unavailable: {exc}")
            for row in method_rows:
                row["baseline_method"] = None
                row["ratio_to_baseline"] = None

        all_rows.extend(method_rows)

        write_csv(csv_path, all_rows)
        if not args.skip_json:
            json_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "device_name": device_name,
                            "device_label": device_label,
                            "sm": sm,
                            "m": args.m,
                            "n": args.n,
                            "k_values": k_values,
                            "dtype": args.dtype,
                            "dist_a": args.dist_a,
                            "dist_b": args.dist_b,
                            "iters": args.iters,
                            "warmup": args.warmup,
                            "run_real": run_real,
                            "run_real_mode": args.run_real,
                            "real_available": real_available,
                            "methods": list(DEFAULT_METHODS),
                            "w_stage3": args.w_stage3,
                            "w_stage4": args.w_stage4,
                            "m_chunk_size": args.m_chunk_size,
                            "triton_block_size": args.triton_block_size,
                            "triton_use_stage3": not args.disable_stage3_triton,
                            "triton_fuse_stage34": not args.disable_stage34_fusion,
                            "seed": args.seed,
                        },
                        "rows": all_rows,
                    },
                    indent=2,
                )
                + "\n"
            )

    print_summary(all_rows, device_name, device_label, args.m, args.n)
    print(f"[Save] CSV written to {csv_path}")
    if not args.skip_json:
        print(f"[Save] JSON written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
