"""
Speed benchmark for emulation path only.

This script intentionally excludes real FP4 GEMM (`ops.cutlass_scaled_fp4_mm`).
It benchmarks `MMAEngine.emulation_scaled_fp4_mm` with pre-quantized inputs.
"""
import argparse
import os
import random
import statistics
import time

import torch

from emulation.core import MMAEngine
from emulation.utils import DataGenerator
from nvfp.pseudo_quant import linear_to_swizzled_128_4, pytorch_nvfp4_quantize


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
ALPHA_SCALE = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX


def get_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    amax = torch.abs(tensor).max().to(torch.float32).item()
    denom = amax if amax > 0 else 1.0
    return torch.tensor([ALPHA_SCALE / denom], device=tensor.device, dtype=torch.float32)


def build_inputs(
    m: int,
    n: int,
    k: int,
    dist_a: str,
    dist_b: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = DataGenerator.get_random_tensor((m, k), dist_a, device=device, dtype=torch.float16)
    b = DataGenerator.get_random_tensor((n, k), dist_b, device=device, dtype=torch.float16)

    a_gs = get_global_scale(a)
    b_gs = get_global_scale(b)
    alpha = torch.tensor([1.0 / (a_gs.item() * b_gs.item())], device=device, dtype=torch.float32)

    # Quantize via pure PyTorch path (no compiled real-quant kernel dependency).
    # `pytorch_nvfp4_quantize` returns fp4 dtype tensor + linear scale layout.
    # Emulation path expects packed uint8 fp4 and swizzled scale layout.
    a_fp4_f4x2, scale_a_linear = pytorch_nvfp4_quantize(a, a_gs)
    b_fp4_f4x2, scale_b_linear = pytorch_nvfp4_quantize(b, b_gs)

    a_fp4 = a_fp4_f4x2.contiguous().view(torch.uint8)
    b_fp4 = b_fp4_f4x2.contiguous().view(torch.uint8)
    scale_a = linear_to_swizzled_128_4(scale_a_linear).contiguous()
    scale_b = linear_to_swizzled_128_4(scale_b_linear).contiguous()
    return a_fp4, b_fp4, scale_a, scale_b, alpha


def run_benchmark(
    emu_fn,
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    alpha: torch.Tensor,
    m: int,
    n: int,
    k: int,
    w_stage3: int,
    w_stage4: int,
    m_chunk_size: int,
    warmup: int,
    iters: int,
    extra_kwargs: dict | None = None,
) -> list[float]:
    extra_kwargs = extra_kwargs or {}
    with torch.no_grad():
        for _ in range(warmup):
            _ = emu_fn(
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
                **extra_kwargs,
            )
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            _ = emu_fn(
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
                **extra_kwargs,
            )
            ends[i].record()
        torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def run_stage_profile(
    emu_fn,
    profile_emu_fn,
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    alpha: torch.Tensor,
    m: int,
    n: int,
    k: int,
    w_stage3: int,
    w_stage4: int,
    m_chunk_size: int,
    warmup: int,
    iters: int,
    extra_kwargs: dict | None = None,
) -> dict[str, float]:
    extra_kwargs = extra_kwargs or {}
    profile_emu_fn = profile_emu_fn or emu_fn
    with torch.no_grad():
        for _ in range(warmup):
            _ = profile_emu_fn(
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
                **extra_kwargs,
            )

        profiles = []
        for _ in range(iters):
            _, profile = profile_emu_fn(
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
                return_profile=True,
                **extra_kwargs,
            )
            profiles.append(profile)

    keys = profiles[0].keys()
    return {k: statistics.mean([p[k] for p in profiles]) for k in keys}


def print_stage_profile(profile: dict[str, float]) -> None:
    stage14_total = profile["stage1_ms"] + profile["stage2_ms"] + profile["stage3_ms"] + profile["stage4_ms"]

    print("Stage Breakdown (avg ms, synchronized boundary timing):")
    for key in ["stage1_ms", "stage2_ms", "stage3_ms", "stage4_ms"]:
        value = profile[key]
        pct = (value / stage14_total * 100.0) if stage14_total > 0 else 0.0
        print(f"  {key:<16} {value:>10.3f} ms   ({pct:>6.2f}% of Stage1-4)")

    print("Other Breakdown:")
    for key in ["preprocess_b_ms", "preprocess_a_ms", "concat_ms", "stage5_ms", "total_profiled_ms"]:
        print(f"  {key:<16} {profile[key]:>10.3f} ms")


def compare_exact_with_nan_equal(ref: torch.Tensor, got: torch.Tensor) -> tuple[bool, int, float]:
    diff = (ref.float() - got.float()).abs()
    equal_mask = (ref == got)
    both_nan = ref.isnan() & got.isnan()
    valid_mask = equal_mask | both_nan
    diff[valid_mask] = 0.0
    diff[diff.isnan()] = float("inf")
    mismatch_count = int((diff != 0).sum().item())
    max_diff = float(diff.max().item())
    return mismatch_count == 0, mismatch_count, max_diff


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark emulation_scaled_fp4_mm only")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--w-stage3", type=int, default=34)
    parser.add_argument("--w-stage4", type=int, default=28)
    parser.add_argument("--m-chunk-size", type=int, default=128)
    parser.add_argument(
        "--impl",
        type=str,
        default="optimized",
        choices=["baseline", "optimized", "triton", "triton_stage234", "triton_stage234_bmm"],
    )
    parser.add_argument("--skip-correctness-check", action="store_true")
    parser.add_argument("--triton-block-size", type=int, default=256)
    parser.add_argument("--triton-disable-stage3", action="store_true")
    parser.add_argument("--triton-fuse-stage34", action="store_true")
    parser.add_argument("--triton-verify-stage3", action="store_true")
    parser.add_argument("--triton-verify-stage4", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--dist-a", type=str, default="normal")
    parser.add_argument("--dist-b", type=str, default="normal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    seed = args.seed if args.seed is not None else (int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("=" * 90)
    print("Emulation Speed Benchmark (emulation_scaled_fp4_mm only)")
    print(f"seed={seed}")
    print(
        f"M={args.m}, N={args.n}, K={args.k}, iters={args.iters}, warmup={args.warmup}, "
        f"W3={args.w_stage3}, W4={args.w_stage4}, m_chunk_size={args.m_chunk_size}, impl={args.impl}"
    )
    print(f"dist_a={args.dist_a}, dist_b={args.dist_b}, device={args.device}")
    print("=" * 90)

    prep_t0 = time.time()
    a_fp4, b_fp4, scale_a, scale_b, alpha = build_inputs(
        args.m, args.n, args.k, args.dist_a, args.dist_b, args.device
    )
    torch.cuda.synchronize()
    prep_ms = (time.time() - prep_t0) * 1000.0
    print(f"Input build + quantization (excluded from benchmark): {prep_ms:.2f} ms")

    if args.impl == "optimized":
        emu_fn = MMAEngine.emulation_scaled_fp4_mm_optimized
        profile_emu_fn = MMAEngine.emulation_scaled_fp4_mm_optimized
        extra_kwargs = {}
    elif args.impl == "triton":
        emu_fn = MMAEngine.emulation_scaled_fp4_mm_triton
        profile_emu_fn = MMAEngine.emulation_scaled_fp4_mm_triton_profile
        extra_kwargs = {
            "triton_block_size": args.triton_block_size,
            "triton_use_stage3": not args.triton_disable_stage3,
            "triton_fuse_stage34": args.triton_fuse_stage34,
            "verify_stage3": args.triton_verify_stage3,
            "verify_stage4": args.triton_verify_stage4,
        }
    elif args.impl == "triton_stage234":
        emu_fn = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused
        profile_emu_fn = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused_profile
        extra_kwargs = {
            "triton_block_size": args.triton_block_size,
        }
    elif args.impl == "triton_stage234_bmm":
        emu_fn = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused_bmm
        profile_emu_fn = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused_bmm_profile
        extra_kwargs = {
            "triton_block_size": args.triton_block_size,
        }
    else:
        emu_fn = MMAEngine.emulation_scaled_fp4_mm
        profile_emu_fn = MMAEngine.emulation_scaled_fp4_mm
        extra_kwargs = {}

    if args.impl in ("optimized", "triton", "triton_stage234", "triton_stage234_bmm") and not args.skip_correctness_check:
        print("Running exact correctness check against baseline...")
        with torch.no_grad():
            ref_out = MMAEngine.emulation_scaled_fp4_mm(
                a_fp4, b_fp4, scale_a, scale_b, alpha, args.m, args.n, args.k,
                W_stage3=args.w_stage3, W_stage4=args.w_stage4, m_chunk_size=args.m_chunk_size
            )
            cand_out = emu_fn(
                a_fp4, b_fp4, scale_a, scale_b, alpha, args.m, args.n, args.k,
                W_stage3=args.w_stage3,
                W_stage4=args.w_stage4,
                m_chunk_size=args.m_chunk_size,
                **extra_kwargs,
            )
        torch.cuda.synchronize()
        ok, mismatch_count, max_diff = compare_exact_with_nan_equal(ref_out, cand_out)
        if not ok:
            raise RuntimeError(
                f"Correctness check failed: mismatch_count={mismatch_count}, max_diff={max_diff}"
            )
        print(f"Correctness check passed: {args.impl} output is bit-identical to baseline.")

    times_ms = run_benchmark(
        emu_fn=emu_fn,
        a_fp4=a_fp4,
        b_fp4=b_fp4,
        scale_a=scale_a,
        scale_b=scale_b,
        alpha=alpha,
        m=args.m,
        n=args.n,
        k=args.k,
        w_stage3=args.w_stage3,
        w_stage4=args.w_stage4,
        m_chunk_size=args.m_chunk_size,
        warmup=args.warmup,
        iters=args.iters,
        extra_kwargs=extra_kwargs,
    )

    avg_ms = statistics.mean(times_ms)
    med_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    stdev_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    tflops = (2.0 * args.m * args.n * args.k) / (avg_ms / 1000.0) / 1e12

    print("-" * 90)
    print(
        f"Emulation time (ms): avg={avg_ms:.3f}, median={med_ms:.3f}, "
        f"min={min_ms:.3f}, max={max_ms:.3f}, std={stdev_ms:.3f}"
    )
    print(f"Effective throughput (based on 2MNK): {tflops:.3f} TFLOPS")
    print("-" * 90)

    if args.profile_stages:
        print("Running stage profiling...")
        stage_profile = run_stage_profile(
            emu_fn=emu_fn,
            profile_emu_fn=profile_emu_fn,
            a_fp4=a_fp4,
            b_fp4=b_fp4,
            scale_a=scale_a,
            scale_b=scale_b,
            alpha=alpha,
            m=args.m,
            n=args.n,
            k=args.k,
            w_stage3=args.w_stage3,
            w_stage4=args.w_stage4,
            m_chunk_size=args.m_chunk_size,
            warmup=args.profile_warmup,
            iters=args.profile_iters,
            extra_kwargs=extra_kwargs,
        )
        print_stage_profile(stage_profile)
        print("-" * 90)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
