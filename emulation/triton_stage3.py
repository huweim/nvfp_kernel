"""
Triton Stage-3 (and optional Stage3+4 fused) kernels for NVFP4 emulation.

Numerical contract:
- Preserve per-element sequential order.
- Preserve the same RZ semantics used by the PyTorch reference path.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    TRITON_STAGE3_AVAILABLE = True
except Exception:  # pragma: no cover - runtime optional dependency
    triton = None
    tl = None
    libdevice = None
    TRITON_STAGE3_AVAILABLE = False


if TRITON_STAGE3_AVAILABLE:
    @triton.jit
    def _to_float32_rz(x):
        f32 = x.to(tl.float32)
        mask = tl.abs(f32).to(tl.float64) > tl.abs(x)
        towards_zero = libdevice.nextafter(f32, tl.zeros_like(f32))
        return tl.where(mask, towards_zero, f32)


    @triton.jit
    def _reduce4_to_wbits_rz(v0, v1, v2, v3, W: tl.constexpr):
        max_abs = tl.maximum(tl.maximum(tl.abs(v0), tl.abs(v1)), tl.maximum(tl.abs(v2), tl.abs(v3)))
        max_is_zero = max_abs == 0
        max_exp = tl.where(max_is_zero, 0, libdevice.ilogb(max_abs) + 1)
        scale = libdevice.ldexp(tl.full(v0.shape, 1.0, tl.float64), (W - max_exp).to(tl.int32))

        s = (
            libdevice.trunc(v0 * scale)
            + libdevice.trunc(v1 * scale)
            + libdevice.trunc(v2 * scale)
            + libdevice.trunc(v3 * scale)
        )
        return s / scale


    @triton.jit
    def _hardware_add_wbits_rz(acc_fp32, new_val_wbits, W: tl.constexpr):
        abs_acc = tl.abs(acc_fp32)
        acc_is_zero = abs_acc == 0
        acc_exp = tl.where(acc_is_zero, 0, libdevice.ilogb(abs_acc) + 1)
        scale = libdevice.ldexp(tl.full(acc_fp32.shape, 1.0, tl.float64), (W - acc_exp).to(tl.int32))

        acc_aligned = acc_fp32.to(tl.float64) * scale
        new_val_aligned = libdevice.trunc(new_val_wbits * scale)
        sum_f64 = (acc_aligned + new_val_aligned) / scale
        return _to_float32_rz(sum_f64)


    @triton.jit
    def _stage3_reduce4_rz_kernel(
        in_ptr,
        out_ptr,
        numel,
        in_row_stride,
        W: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = offs < numel

        row_base = offs * in_row_stride.to(tl.int64)
        v0 = tl.load(in_ptr + row_base + 0, mask=mask, other=0.0).to(tl.float64)
        v1 = tl.load(in_ptr + row_base + 1, mask=mask, other=0.0).to(tl.float64)
        v2 = tl.load(in_ptr + row_base + 2, mask=mask, other=0.0).to(tl.float64)
        v3 = tl.load(in_ptr + row_base + 3, mask=mask, other=0.0).to(tl.float64)

        out = _reduce4_to_wbits_rz(v0, v1, v2, v3, W=W)
        tl.store(out_ptr + offs, out, mask=mask)


    @triton.jit
    def _stage34_fused_rz_kernel(
        in_ptr,
        out_ptr,
        num_rows,
        in_row_stride,
        W_STAGE3: tl.constexpr,
        W_STAGE4: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = offs < num_rows

        row_base = offs * in_row_stride.to(tl.int64)
        base0 = row_base + 0 * 4
        v00 = tl.load(in_ptr + base0 + 0, mask=mask, other=0.0).to(tl.float64)
        v01 = tl.load(in_ptr + base0 + 1, mask=mask, other=0.0).to(tl.float64)
        v02 = tl.load(in_ptr + base0 + 2, mask=mask, other=0.0).to(tl.float64)
        v03 = tl.load(in_ptr + base0 + 3, mask=mask, other=0.0).to(tl.float64)
        first = _reduce4_to_wbits_rz(v00, v01, v02, v03, W=W_STAGE3)
        acc = _to_float32_rz(first)

        for b in range(1, NUM_BLOCKS):
            base = row_base + b * 4
            v0 = tl.load(in_ptr + base + 0, mask=mask, other=0.0).to(tl.float64)
            v1 = tl.load(in_ptr + base + 1, mask=mask, other=0.0).to(tl.float64)
            v2 = tl.load(in_ptr + base + 2, mask=mask, other=0.0).to(tl.float64)
            v3 = tl.load(in_ptr + base + 3, mask=mask, other=0.0).to(tl.float64)
            block_sum = _reduce4_to_wbits_rz(v0, v1, v2, v3, W=W_STAGE3)
            acc = _hardware_add_wbits_rz(acc, block_sum, W=W_STAGE4)

        tl.store(out_ptr + offs, acc, mask=mask)


def stage3_reduce_4to1_rz_triton(
    scaled_partials_4: torch.Tensor,
    w_stage3: int,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Triton Stage-3 kernel.

    Args:
        scaled_partials_4: [M, N, num_blocks, 4], float32
        w_stage3: Stage3 reduction bit width
    Returns:
        [M, N, num_blocks], float64
    """
    if not TRITON_STAGE3_AVAILABLE:
        raise RuntimeError("Triton stage3 is unavailable in current environment.")
    if not scaled_partials_4.is_cuda:
        raise RuntimeError("scaled_partials_4 must be CUDA tensor.")
    if scaled_partials_4.dtype != torch.float32:
        raise RuntimeError(f"scaled_partials_4 must be float32, got {scaled_partials_4.dtype}.")
    if scaled_partials_4.ndim != 4 or scaled_partials_4.shape[-1] != 4:
        raise RuntimeError(f"scaled_partials_4 must be [M, N, num_blocks, 4], got {tuple(scaled_partials_4.shape)}.")

    m, n, num_blocks, _ = scaled_partials_4.shape
    in2d = scaled_partials_4.contiguous().view(-1, 4)
    numel = in2d.shape[0]
    out = torch.empty((numel,), device=scaled_partials_4.device, dtype=torch.float64)

    grid = (triton.cdiv(numel, block_size),)
    _stage3_reduce4_rz_kernel[grid](
        in2d,
        out,
        numel,
        in2d.stride(0),
        W=w_stage3,
        BLOCK_SIZE=block_size,
    )
    return out.view(m, n, num_blocks)


def stage34_fused_rz_triton(
    scaled_partials_4: torch.Tensor,
    w_stage3: int,
    w_stage4: int,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Triton fused Stage3+4 kernel.

    Args:
        scaled_partials_4: [M, N, num_blocks, 4], float32
    Returns:
        [M, N], float32
    """
    if not TRITON_STAGE3_AVAILABLE:
        raise RuntimeError("Triton stage3 is unavailable in current environment.")
    if not scaled_partials_4.is_cuda:
        raise RuntimeError("scaled_partials_4 must be CUDA tensor.")
    if scaled_partials_4.dtype != torch.float32:
        raise RuntimeError(f"scaled_partials_4 must be float32, got {scaled_partials_4.dtype}.")
    if scaled_partials_4.ndim != 4 or scaled_partials_4.shape[-1] != 4:
        raise RuntimeError(f"scaled_partials_4 must be [M, N, num_blocks, 4], got {tuple(scaled_partials_4.shape)}.")

    m, n, num_blocks, _ = scaled_partials_4.shape
    in2d = scaled_partials_4.contiguous().view(-1, num_blocks * 4)
    num_rows = in2d.shape[0]
    out = torch.empty((num_rows,), device=scaled_partials_4.device, dtype=torch.float32)

    grid = (triton.cdiv(num_rows, block_size),)
    _stage34_fused_rz_kernel[grid](
        in2d,
        out,
        num_rows,
        in2d.stride(0),
        W_STAGE3=w_stage3,
        W_STAGE4=w_stage4,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
    )
    return out.view(m, n)
