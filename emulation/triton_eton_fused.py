"""
ETON-Fused: one Triton kernel for the whole probed-template block-scaled FP4 GEMM (NVFP4 / MXFP4).

Same numerics as triton_fusednode.py (the staged ETON fused node), but nothing intermediate goes to memory: per output
tile and per k64 MMA instruction the 4 exact 16-element group sums come from int8 tensor-core tl.dot (operands are
2 x e2m1, i.e. integers in [-12, 12]; int32 accumulation is exact), and the fused node runs in registers:

  emax   = max(lead(C), max over active groups of anchor_j)    anchor_j = scale exponents (+ floors, see template)
  q      = emax - F
  acc'   = RZ_fp32( trunc(C / 2^q) + sum_j trunc(term_j / 2^q) ) * 2^q

The exact sum uses two int32 limbs (hi * 2^24 + lo) and the final RZ is integer shifting: no fp64 and no 64-bit
conversions (F2I/I2F.S64 issue on the FP64 pipe, 1/64 rate on GeForce parts; that was the first prototype's bottleneck).

Numerical parameters are read from the probe template (probe_triton/results/sm120_*.json -> FusedTemplate.from_probe);
fields outside the supported family raise instead of being silently ignored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    TRITON_ETON_FUSED_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    TRITON_ETON_FUSED_AVAILABLE = False

_NEG = -100000


@dataclass(frozen=True)
class FusedTemplate:
    F: int = 35                      # fractional bits of the fused-node grid (significand F + 1)
    scale_fmt: str = "ue4m3"         # 'ue4m3' (NVFP4) | 'ue8m0' (MXFP4)
    scale_block: int = 16            # elements per scale along K
    term_floor: Optional[int] = None  # group anchor clamped to >= term_floor
    c_floor: Optional[int] = None     # lead(C) clamped to >= c_floor (subnormal C anchored at emin)
    overflow_inf: bool = False       # True: |result| >= 2^128 -> inf; False: IEEE RZ -> FLT_MAX

    # the structure every supported template shares (checked in from_probe)
    _FIXED = {"K": 64, "igroup": 16, "intra": "exact", "anchor": "scale", "anchor_offset": 0,
              "anchor_zero_blocks": "products", "c_order": "early", "align_rm": "RZ", "out_rm": "RZ",
              "elem_sub_anchor": "emin", "sub_scale_anchor": "emin", "a_fmt": "e2m1", "b_fmt": "e2m1",
              "emax_floor": None}

    @classmethod
    def from_probe(cls, template_or_path) -> "FusedTemplate":
        t = template_or_path
        if isinstance(t, str):
            with open(t) as f:
                t = json.load(f)
        t = t.get("template", t)
        bad = {k: t.get(k) for k, v in cls._FIXED.items() if t.get(k) != v}
        if bad:
            raise NotImplementedError(f"ETON-Fused does not implement these template fields: {bad}")
        if t["scale_fmt"] not in ("ue4m3", "ue8m0"):
            raise NotImplementedError(f"scale_fmt {t['scale_fmt']}")
        return cls(F=int(t["F"]), scale_fmt=t["scale_fmt"], scale_block=int(t["block"]),
                   term_floor=t.get("term_anchor_floor"),
                   c_floor=-126 if t.get("c_sub_anchor") == "emin" else None,
                   overflow_inf=t.get("overflow") == "inf")


NVFP4_SM120 = FusedTemplate()
MXFP4_SM120 = FusedTemplate(scale_fmt="ue8m0", scale_block=32, term_floor=-139, c_floor=-126, overflow_inf=True)


def _configs():
    cfgs = []
    for bm, bn, w, s in [(16, 16, 2, 3), (16, 32, 2, 3), (16, 32, 4, 3), (32, 16, 2, 3), (32, 32, 2, 3),
                         (64, 16, 4, 3), (32, 32, 4, 3), (64, 32, 4, 3), (32, 64, 4, 3), (64, 64, 4, 3)]:
        cfgs.append(triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s))
    return cfgs


if TRITON_ETON_FUSED_AVAILABLE:
    @triton.jit
    def _pow2(k):
        """2^k as fp32 for integer k in [-126, 127]."""
        return ((k + 127) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _limbs(x):
        """trunc(x), |x| < 2^48, as exact int32 limbs hi * 2^24 + lo (|lo| < 2^24, same sign as x)."""
        h = libdevice.trunc(x * 5.9604644775390625e-08)            # 2^-24, exact
        l = x - h * 16777216.0                                    # exact
        return h.to(tl.int32), l.to(tl.int32)                     # F2I truncates

    @triton.jit
    def _scale_anchor(code, UE8M0: tl.constexpr):
        if UE8M0:
            return code - 127
        e = (code >> 3) & 15
        return tl.where(e > 0, e - 7, -6)                         # subnormal ue4m3 anchored at emin

    @triton.jit
    def _ue4m3_value(code):
        e = (code >> 3) & 15
        m = code & 7
        return tl.where(e > 0, m + 8, m).to(tl.float32) * _pow2(tl.where(e > 0, e - 10, -9))

    @triton.jit
    def _eton_fused_kernel_impl(
        a_ptr, b_ptr,                  # int8 2*e2m1 values [M, K], [N, K]
        ia_ptr, ib_ptr,                # int32 per group: nonzero mask (bits 0-15) | scale code << 16   [M, G], [N, G]
        out_ptr, alpha_ptr,
        M, N, K, M_BUCKET, N_BUCKET, K_BUCKET,                  # buckets: autotune key only
        stride_am, stride_bn, stride_iam, stride_ibn, stride_om,
        F: tl.constexpr, UE8M0: tl.constexpr, TERM_FLOOR: tl.constexpr, C_FLOOR: tl.constexpr,
        OVERFLOW_INF: tl.constexpr, OUT_BF16: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):
        NEG: tl.constexpr = -100000
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mmask = rm < M
        nmask = rn < N
        rk = tl.arange(0, 32)
        a_rows = a_ptr + rm.to(tl.int64)[:, None] * stride_am + rk[None, :]          # [BM, 32]
        b_cols = b_ptr + rn.to(tl.int64)[None, :] * stride_bn + rk[:, None]          # [32, BN]
        first_half = rk[:, None] < 16
        ia_rows = ia_ptr + rm * stride_iam
        ib_cols = ib_ptr + rn * stride_ibn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for step in range(0, K // 64):
            # ---- emax over C and the active groups of this k64 instruction
            ex = (acc.to(tl.int32, bitcast=True) >> 23) & 0xFF
            c_lead = ex - 127                                      # exact for normal C (subnormal C: below C_FLOOR)
            if C_FLOOR > NEG:
                c_lead = tl.maximum(c_lead, C_FLOOR)
            emax = tl.where(acc == 0, NEG, c_lead)
            for j in tl.static_range(4):
                ia = tl.load(ia_rows + step * 4 + j, mask=mmask, other=0)
                ib = tl.load(ib_cols + step * 4 + j, mask=nmask, other=0)
                anchor = _scale_anchor(ia >> 16, UE8M0)[:, None] + _scale_anchor(ib >> 16, UE8M0)[None, :]
                if TERM_FLOOR > NEG:
                    anchor = tl.maximum(anchor, TERM_FLOOR)
                active = ((ia & 0xFFFF)[:, None] & (ib & 0xFFFF)[None, :]) != 0
                emax = tl.maximum(emax, tl.where(active, anchor, NEG))
            q = tl.where(emax > NEG, emax - F, 0)
            # ---- exact sum on the 2^q grid, every operand truncated toward zero
            if UE8M0:
                h1 = (-q) >> 1
                hi, lo = _limbs(acc * _pow2(h1) * _pow2(-q - h1))   # -q up to 174: two exact scalings
            else:
                hi, lo = _limbs(acc * _pow2(-q))
            nan_hit = acc != acc
            for jj in tl.static_range(2):
                a = tl.load(a_rows + step * 64 + jj * 32, mask=mmask[:, None], other=0)
                b = tl.load(b_cols + step * 64 + jj * 32, mask=nmask[None, :], other=0)
                zero = tl.zeros_like(b)
                for h in tl.static_range(2):
                    j = jj * 2 + h
                    if h == 0:
                        p = tl.dot(a, tl.where(first_half, b, zero)).to(tl.float32)   # 4 x exact group sum
                    else:
                        p = tl.dot(a, tl.where(first_half, zero, b)).to(tl.float32)
                    ca = tl.load(ia_rows + step * 4 + j, mask=mmask, other=0) >> 16
                    cb = tl.load(ib_cols + step * 4 + j, mask=nmask, other=0) >> 16
                    if UE8M0:
                        sh = _scale_anchor(ca, True)[:, None] + _scale_anchor(cb, True)[None, :] - q - 2
                        t = p * _pow2(tl.minimum(tl.maximum(sh, -126), 127))     # below 2^-126: truncates to 0
                    else:
                        t = (p * (_ue4m3_value(ca)[:, None] * _ue4m3_value(cb)[None, :])) * _pow2(-q - 2)
                        nan_hit = nan_hit | ((ca == 0x7F)[:, None] | (cb == 0x7F)[None, :])   # NaN scale -> NaN
                    th, tlo = _limbs(t)
                    hi += th
                    lo += tlo
            # ---- RZ to fp32 (subnormal results truncated onto the 2^-149 grid: one rounding)
            c = lo >> 24
            lo_n = lo & 0xFFFFFF
            hi_n = hi + c
            neg = hi_n < 0
            borrow = neg & (lo_n != 0)
            am = tl.where(neg, tl.where(borrow, -hi_n - 1, -hi_n), hi_n)      # |S| = am * 2^24 + bm
            bm = tl.where(borrow, 16777216 - lo_n, lo_n)
            L = tl.where(am > 0, 56 - libdevice.clz(am), 32 - libdevice.clz(bm))  # bit length of |S|
            d = tl.minimum(tl.maximum(tl.maximum(L - 24, -149 - q), 0), 56)
            mant = tl.where(d >= 24, am >> tl.minimum(tl.maximum(d - 24, 0), 31),
                            (am << tl.minimum(tl.maximum(24 - d, 0), 31)) | (bm >> tl.minimum(d, 31)))
            mant = tl.where(d >= 48, 0, mant).to(tl.float32)
            e2 = d + q
            if UE8M0:
                e2h = e2 >> 1
                r = mant * _pow2(tl.minimum(tl.maximum(e2h, -126), 127)) * _pow2(tl.minimum(tl.maximum(e2 - e2h, -126), 127))
            else:
                r = mant * _pow2(tl.minimum(tl.maximum(e2, -126), 127))
            if OVERFLOW_INF:
                big = tl.full((BM, BN), float("inf"), tl.float32)
            else:
                big = tl.full((BM, BN), 3.4028234663852886e38, tl.float32)      # IEEE RZ overflow
            r = tl.where((L > 0) & (L + q > 128), big, r)
            r = tl.where(neg, -r, r)
            if OVERFLOW_INF:
                r = tl.where(tl.abs(acc) == float("inf"), acc, r)           # inf C stays inf
            else:
                r = tl.where(nan_hit, float("nan"), r)
            acc = r
        alpha = tl.load(alpha_ptr)
        o = acc * alpha
        optr = out_ptr + rm.to(tl.int64)[:, None] * stride_om + rn[None, :]
        omask = mmask[:, None] & nmask[None, :]
        if OUT_BF16:
            tl.store(optr, o.to(tl.bfloat16), mask=omask)
        else:
            tl.store(optr, o.to(tl.float16), mask=omask)

    _eton_fused_kernel = triton.autotune(configs=_configs(), key=["M_BUCKET", "N_BUCKET", "K_BUCKET"])(_eton_fused_kernel_impl)


if TRITON_ETON_FUSED_AVAILABLE:
    @triton.jit
    def _prep_kernel(p_ptr, sf_ptr, v_ptr, info_ptr, R, G, stride_p, stride_v, stride_info, sf_ktiles,
                     GPS: tl.constexpr, UE4M3: tl.constexpr, BR: tl.constexpr, BG: tl.constexpr):
        """packed e2m1 [R, K/2] + 128x4-swizzled scale bytes -> int8 2*e2m1 [R, K] and int32 group info [R, G]."""
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        g = tl.program_id(1) * BG + tl.arange(0, BG)
        rmask = r < R
        m2 = rmask[:, None] & (g < G)[None, :]
        r64 = r.to(tl.int64)
        k8 = tl.arange(0, 8)
        byte = tl.load(p_ptr + r64[:, None, None] * stride_p + g[None, :, None] * 8 + k8[None, None, :],
                       mask=m2[:, :, None], other=0).to(tl.int32)                     # [BR, BG, 8]
        lo = byte & 0xF
        hi = (byte >> 4) & 0xF
        # 2 * e2m1: code e|m (e = 2 bits) -> e == 0 ? m : (2 + m) << (e - 1); sign = bit 3
        def_lo = tl.where(((lo >> 1) & 3) == 0, lo & 1, (2 + (lo & 1)) << (((lo >> 1) & 3) - 1))
        def_hi = tl.where(((hi >> 1) & 3) == 0, hi & 1, (2 + (hi & 1)) << (((hi >> 1) & 3) - 1))
        v_lo = tl.where((lo & 8) != 0, -def_lo, def_lo)
        v_hi = tl.where((hi & 8) != 0, -def_hi, def_hi)
        v = tl.interleave(v_lo, v_hi)                                                  # [BR, BG, 16]
        k16 = tl.arange(0, 16)
        tl.store(v_ptr + r64[:, None, None] * stride_v + g[None, :, None] * 16 + k16[None, None, :],
                 v.to(tl.int8), mask=m2[:, :, None])
        mask = tl.sum((v != 0).to(tl.int32) << k16[None, None, :], axis=2)            # [BR, BG]
        sidx = g // GPS
        off = (((r // 128) * sf_ktiles)[:, None] + (sidx // 4)[None, :]) * 512 + ((r % 32) * 16 + ((r % 128) // 32) * 4)[:, None] \
            + (sidx % 4)[None, :]
        code = tl.load(sf_ptr + off, mask=m2, other=0).to(tl.int32)
        if UE4M3:
            code = code & 0x7F                                                         # scale_msb: ignore (probed)
            mask = tl.where(code == 0, 0, mask)                                        # zero scale: never active
        tl.store(info_ptr + r[:, None] * stride_info + g[None, :], mask | (code << 16), mask=m2)


def prepare_operand(packed: torch.Tensor, sf_swizzled: torch.Tensor, R: int, K: int, tmpl: "FusedTemplate"):
    """One operand of eton_fused_mm: (int8 [R, K] = 2 * e2m1 values, int32 [R, K/16] = nonzero mask | scale code << 16).
    Static weights can be prepared once and passed as b_prepared."""
    G = K // 16
    v = torch.empty((R, K), dtype=torch.int8, device=packed.device)
    info = torch.empty((R, G), dtype=torch.int32, device=packed.device)
    p = packed.view(torch.uint8)
    sf = sf_swizzled.view(torch.uint8).contiguous()
    BR, BG = 32, 8
    _prep_kernel[(triton.cdiv(R, BR), triton.cdiv(G, BG))](
        p, sf, v, info, R, G, p.stride(0), v.stride(0), info.stride(0), sf.shape[1] // 4,
        GPS=tmpl.scale_block // 16, UE4M3=tmpl.scale_fmt == "ue4m3", BR=BR, BG=BG)
    return v, info


def _bucket(x: int) -> int:
    return min(1 << max(x - 1, 0).bit_length(), 16384)


def _fixed_config(M: int):
    """Tile config by M (best or within ~5% of best in the RTX 5090 sweep). Used unless ETON_FUSED_AUTOTUNE=1:
    autotuning costs ~1.2 s per new (M, N, K) bucket, which would land inside end-to-end timings. The output does not
    depend on the config (each output element is reduced by one program over K in a fixed order)."""
    if M <= 64:
        return dict(BM=16, BN=32, num_warps=4, num_stages=3)
    if M <= 192:
        return dict(BM=32, BN=32, num_warps=2, num_stages=3)
    return dict(BM=64, BN=16, num_warps=4, num_stages=3)


def eton_fused_mm(a_packed, b_packed, sf_a, sf_b, alpha, M, N, K, tmpl: FusedTemplate = NVFP4_SM120,
                  out_dtype=torch.float16, b_prepared=None):
    """a_packed [M,K/2], b_packed [N,K/2] packed e2m1; sf_a / sf_b: 128x4-swizzled scale bytes (the CUTLASS operand
    layout). b_prepared: optional cached prepare_operand(b) for a static weight. alpha: 1-element fp32 tensor."""
    assert K % 64 == 0
    a8, ia = prepare_operand(a_packed, sf_a, M, K, tmpl)
    b8, ib = prepare_operand(b_packed, sf_b, N, K, tmpl) if b_prepared is None else b_prepared
    out = torch.empty((M, N), device=a8.device, dtype=out_dtype)
    alpha = alpha.reshape(1).float().contiguous()
    if os.environ.get("ETON_FUSED_AUTOTUNE", "0") == "1":
        kern, meta = _eton_fused_kernel, {}
    else:
        kern, meta = _eton_fused_kernel_impl, _fixed_config(M)
    grid = lambda m: (triton.cdiv(M, m["BM"]), triton.cdiv(N, m["BN"]))  # noqa: E731
    kern[grid](
        a8, b8, ia, ib, out, alpha, M, N, K, _bucket(M), _bucket(N), _bucket(K),
        a8.stride(0), b8.stride(0), ia.stride(0), ib.stride(0), out.stride(0),
        F=tmpl.F, UE8M0=tmpl.scale_fmt == "ue8m0",
        TERM_FLOOR=tmpl.term_floor if tmpl.term_floor is not None else _NEG,
        C_FLOOR=tmpl.c_floor if tmpl.c_floor is not None else _NEG,
        OVERFLOW_INF=tmpl.overflow_inf, OUT_BF16=out_dtype == torch.bfloat16, **meta)
    return out
