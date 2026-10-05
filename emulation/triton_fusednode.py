"""
Fused-node Stage2+3+4 kernel for NVFP4 (sm_120a OMMA, probed template; probe_triton/results/sm120_nvfp4.json).

Per k64 MMA instruction and output element, ONE fused node over the 4 group terms and C:
  term_j   = S_j * sa_j * sb_j          (S_j = exact 16-element group sum; sa, sb = ue4m3 scales)
  lead_j   = n(sa_j) + n(sb_j)          (scale-anchored: nominal scale exponents; subnormal ue4m3 -> -6)
  active_j = group j has a nonzero product and both scales are nonzero  (zero-sum groups still count)
  lead_C   = floor(log2|C|), active iff C != 0       (early C, aligned by its value)
  emax = max(active leads); every term and C truncated (RZ) to 2^(emax - F); exact sum; RZ to fp32
This replaces the older two-step model (_stage234_scale_fused_rz_kernel: value-anchored 4-group reduce, then a
2-term add anchored at the accumulator), which differs from hardware under large inter-group dynamic range with
cancellation (table3_v2: group_range / cancel families).

Inputs must be exact: ps1 is float32 (a 16-term e2m1 group sum can need 12 significand bits, e.g. 540.25); it is
produced by fp16 tensor-core bmm with fp32 output (products and partial sums are exact in fp32).
Not modelled (never produced by the NVFP4 quantizer): NaN scales (0x7F) and the ue4m3 sign bit.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    TRITON_FUSEDNODE_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    TRITON_FUSEDNODE_AVAILABLE = False


if TRITON_FUSEDNODE_AVAILABLE:
    @triton.jit
    def _to_float32_rz(x):
        f32 = x.to(tl.float32)
        mask = tl.abs(f32).to(tl.float64) > tl.abs(x)
        towards_zero = libdevice.nextafter(f32, tl.zeros_like(f32))
        return tl.where(mask, towards_zero, f32)

    @triton.jit
    def _fusednode_kernel(
        ps1_ptr, cnt_ptr, s_a_ptr, s_b_ptr, n_a_ptr, n_b_ptr, out_ptr,
        num_rows, N,
        ps1_stride_m, ps1_stride_n, ps1_stride_g,
        cnt_stride_m, cnt_stride_n, cnt_stride_g,
        s_a_stride_m, s_b_stride_n,
        F: tl.constexpr, NUM_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
        mask = offs < num_rows
        m_idx = offs // N
        n_idx = offs - m_idx * N
        pbase = m_idx * ps1_stride_m + n_idx * ps1_stride_n
        cbase = m_idx * cnt_stride_m + n_idx * cnt_stride_n
        abase = m_idx * s_a_stride_m
        bbase = n_idx * s_b_stride_n
        NEG: tl.constexpr = -100000
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for blk in range(NUM_BLOCKS):
            emax = tl.full((BLOCK_SIZE,), NEG, tl.int32)
            # C: aligned by its true leading bit
            acc_abs = tl.abs(acc)
            c_lead = tl.where(acc_abs == 0, NEG, libdevice.ilogb(acc_abs))
            emax = tl.maximum(emax, c_lead)
            t0 = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            t1 = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            t2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            t3 = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
            for j in tl.static_range(4):
                g = blk * 4 + j
                p = tl.load(ps1_ptr + pbase + g * ps1_stride_g, mask=mask, other=0).to(tl.float64)
                nz = tl.load(cnt_ptr + cbase + g * cnt_stride_g, mask=mask, other=0)
                sa = tl.load(s_a_ptr + abase + g, mask=mask, other=0).to(tl.float64)
                sb = tl.load(s_b_ptr + bbase + g, mask=mask, other=0).to(tl.float64)
                na = tl.load(n_a_ptr + abase + g, mask=mask, other=0)
                nb = tl.load(n_b_ptr + bbase + g, mask=mask, other=0)
                lead = tl.where((nz != 0) & (sa != 0) & (sb != 0), na + nb, NEG)
                emax = tl.maximum(emax, lead)
                t = p * sa * sb                      # exact in fp64
                if j == 0:
                    t0 = t
                elif j == 1:
                    t1 = t
                elif j == 2:
                    t2 = t
                else:
                    t3 = t
            any_active = emax > NEG
            q = tl.where(any_active, emax - F, 0)
            sc = libdevice.ldexp(tl.full((BLOCK_SIZE,), 1.0, tl.float64), (-q).to(tl.int32))
            s = (libdevice.trunc(acc.to(tl.float64) * sc) + libdevice.trunc(t0 * sc) + libdevice.trunc(t1 * sc)
                 + libdevice.trunc(t2 * sc) + libdevice.trunc(t3 * sc))
            v = s / sc
            acc = _to_float32_rz(v)
        tl.store(out_ptr + offs, acc, mask=mask)


def ue4m3_nominal_exponent(s: torch.Tensor) -> torch.Tensor:
    """floor(log2 s) for normal ue4m3 scales, -6 for subnormal ones (probed sub_scale_anchor = emin); 0 for s == 0."""
    s = s.float()
    e = torch.floor(torch.log2(torch.where(s > 0, s, torch.ones_like(s)))).to(torch.int32)
    e = torch.where(s < 2.0 ** -6, torch.full_like(e, -6), e)
    return torch.where(s > 0, e, torch.zeros_like(e))


def stage234_fusednode_rz_triton(ps1, cnt, s_a, s_b, F: int = 35, block_size: int = 256) -> torch.Tensor:
    """ps1 [M,N,G] float32 exact group sums and cnt [M,N,G] (any dtype; != 0 iff the group has a nonzero product),
    both as strided views (e.g. permuted [G,M,N] bmm outputs; no copy); s_a [M,G], s_b [N,G] scale values.
    A group takes part in the max iff cnt != 0 and both scales are nonzero. Returns [M,N] float32 (before alpha)."""
    if not TRITON_FUSEDNODE_AVAILABLE:
        raise RuntimeError("Triton is unavailable.")
    assert ps1.dtype == torch.float32 and ps1.shape == cnt.shape
    m, n, g = ps1.shape
    assert g % 4 == 0 and s_a.shape == (m, g) and s_b.shape == (n, g)
    s_a, s_b = s_a.float().contiguous(), s_b.float().contiguous()
    n_a, n_b = ue4m3_nominal_exponent(s_a).contiguous(), ue4m3_nominal_exponent(s_b).contiguous()
    out = torch.empty((m * n,), device=ps1.device, dtype=torch.float32)
    grid = (triton.cdiv(m * n, block_size),)
    _fusednode_kernel[grid](ps1, cnt, s_a, s_b, n_a, n_b, out, m * n, n,
                            ps1.stride(0), ps1.stride(1), ps1.stride(2), cnt.stride(0), cnt.stride(1), cnt.stride(2),
                            s_a.stride(0), s_b.stride(0), F=F, NUM_BLOCKS=g // 4, BLOCK_SIZE=block_size)
    return out.view(m, n)
