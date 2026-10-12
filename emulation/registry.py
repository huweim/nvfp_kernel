"""The one table of NVFP4 GEMM implementations.

Both dispatch points use it:
  * EmulationKernel (QuantLinear mode="emulation", CLI --emulation-impl)
  * nvfp._ops_emulation.cutlass_scaled_fp4_mm (QuantLinear mode="real" with NVFP_GEMM_BACKEND=emulation, env
    NVFP_EMULATION_IMPL)

Every entry has the signature fn(a, b, scale_a, scale_b, alpha, M, N, K, accum_bits, out_dtype) -> [M, N] tensor, with
the CUTLASS operand layout (packed e2m1 uint8, 128x4-swizzled ue4m3 scales, 1-element fp32 alpha).
accum_bits = significand bits of the inter-group accumulator (= F + 1; 36 on sm_120a).
"""
import torch

from . import gemm_hist

DEFAULT_IMPL = "beam_eton_fused"
HW_ACCUM_BITS = 36


def _eton_fused(a, b, sa, sb, al, M, N, K, accum_bits, out_dtype, **_):
    from .core import MMAEngine
    from .triton_eton_fused import NVFP4_SM120, FusedTemplate
    tmpl = NVFP4_SM120 if accum_bits == HW_ACCUM_BITS else FusedTemplate(F=accum_bits - 1)
    return MMAEngine.emulation_scaled_fp4_mm_eton_fused(a, b, sa, sb, al, M, N, K, template=tmpl, out_dtype=out_dtype)


def _fusednode(a, b, sa, sb, al, M, N, K, accum_bits, out_dtype, **_):
    from .core import MMAEngine
    return MMAEngine.emulation_scaled_fp4_mm_fusednode(a, b, sa, sb, al, M, N, K, F=accum_bits - 1).to(out_dtype)


def _base(a, b, sa, sb, al, M, N, K, accum_bits, out_dtype, m_chunk_size=128, **_):
    # m_chunk_size: rows per chunk (speed / memory only). 128 next to a vLLM KV cache; Tab. IV "B." used 512.
    from .core import MMAEngine
    return MMAEngine.emulation_scaled_fp4_mm(a, b, sa, sb, al, M, N, K, W_stage3=accum_bits, W_stage4=accum_bits,
                                             m_chunk_size=m_chunk_size).to(out_dtype)


def _mmasim(a, b, sa, sb, al, M, N, K, accum_bits, out_dtype, **_):
    if accum_bits != HW_ACCUM_BITS:
        raise ValueError("mmasim_v2 models the probed sm_120a instruction only (accum_bits=36)")
    from .mmasim_backend import mmasim_scaled_fp4_mm
    return mmasim_scaled_fp4_mm(a, b, sa, sb, al, M, N, K, out_dtype)


IMPLS = {
    "beam_eton_fused": _eton_fused,   # ETON-Fused: one Triton kernel (default)
    "beam_fusednode": _fusednode,     # same numerics, staged; test oracle for ETON-Fused
    "beam_base": _base,               # ETON-Base (Tab. IV "B."): unfused staged PyTorch pipeline
    "mmasim_v2": _mmasim,             # MMA-Sim v2 reference model (needs `pip install mmasim`)
}


def run(impl, a, b, scale_a, scale_b, alpha, M, N, K, accum_bits=HW_ACCUM_BITS, out_dtype=torch.float16, **opts):
    """opts: implementation-specific speed knobs that never change the result (e.g. m_chunk_size for beam_base)."""
    if impl not in IMPLS:
        raise ValueError(f"unknown emulation impl {impl!r}; expected one of {sorted(IMPLS)}")
    gemm_hist.record(M, N, K)
    return IMPLS[impl](a, b, scale_a, scale_b, alpha, M, N, K, accum_bits, out_dtype, **opts)
