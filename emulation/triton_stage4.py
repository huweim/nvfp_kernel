"""
Triton Stage-4 accumulator for NVFP4 emulation.

This module only targets the RZ path and preserves strict sequential block order.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    TRITON_STAGE4_AVAILABLE = True
except Exception:  # pragma: no cover - runtime optional dependency
    triton = None
    tl = None
    libdevice = None
    TRITON_STAGE4_AVAILABLE = False


if TRITON_STAGE4_AVAILABLE:
    @triton.jit
    def _to_float32_rz(x):
        f32 = x.to(tl.float32)
        mask = tl.abs(f32).to(tl.float64) > tl.abs(x)
        toward_zero = libdevice.nextafter(f32, tl.zeros_like(f32))
        return tl.where(mask, toward_zero, f32)


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
    def _stage4_rz_kernel(
        in_ptr,
        out_ptr,
        numel,
        in_stride,
        W: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = offs < numel

        row_base = offs * in_stride.to(tl.int64)
        v0 = tl.load(in_ptr + row_base + 0, mask=mask, other=0.0).to(tl.float64)
        acc = _to_float32_rz(v0)

        for i in range(1, NUM_BLOCKS):
            new_val = tl.load(in_ptr + row_base + i, mask=mask, other=0.0).to(tl.float64)
            acc = _hardware_add_wbits_rz(acc, new_val, W=W)

        tl.store(out_ptr + offs, acc, mask=mask)


def stage4_accumulate_rz_triton(
    summed_groups: torch.Tensor,
    w_stage4: int,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Sequential Stage-4 accumulation using Triton.

    Args:
        summed_groups: [M, N, num_blocks] float64 tensor
        w_stage4: Stage-4 bit width
        block_size: Triton program block size
    Returns:
        [M, N] float32 tensor
    """
    if not TRITON_STAGE4_AVAILABLE:
        raise RuntimeError("Triton stage4 is unavailable in current environment.")
    if not summed_groups.is_cuda:
        raise RuntimeError("summed_groups must be a CUDA tensor for Triton stage4.")
    if summed_groups.dtype != torch.float64:
        raise RuntimeError(f"summed_groups must be float64, got {summed_groups.dtype}.")
    if summed_groups.ndim != 3:
        raise RuntimeError(f"summed_groups must be [M, N, num_blocks], got shape={tuple(summed_groups.shape)}.")

    m, n, num_blocks = summed_groups.shape
    in2d = summed_groups.contiguous().view(-1, num_blocks)
    numel = in2d.shape[0]

    out = torch.empty((numel,), device=summed_groups.device, dtype=torch.float32)
    grid = (triton.cdiv(numel, block_size),)
    _stage4_rz_kernel[grid](
        in2d,
        out,
        numel,
        in2d.stride(0),
        W=w_stage4,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
    )
    return out.view(m, n)
