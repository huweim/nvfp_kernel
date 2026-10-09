"""MXFP4 (OCP MX v1.0: e2m1 elements, ue8m0 scale per 32 along K) quantizer and ETON-MXFP4 GEMM.

mxfp4_quantize(x [R,K] fp16/bf16/fp32) -> (packed uint8 [R, K/2] (low nibble = even element), ue8m0 scale bytes
    swizzled 128x4 [round_up(R,128), round_up(K/32,4)]) -- the operand layout of kernel_mxfp4 (CUTLASS sm_120a).
    Shared exponent = floor(log2 amax) - 2 (e2m1 emax), clamped to [-127, 127]; elements RNE (ties-to-even) to the e2m1
    grid with saturation at +-6. Every step is exact float arithmetic, so all GPUs produce identical operands.
emulation_scaled_mxfp4_mm_fusednode(...): probed sm_120a MXFP4 template (see triton_fusednode.py), fp16/bf16 output.
"""
import torch

from nvfp import pseudo_quant

_LOWER_TIE = torch.tensor([0.25, 1.25, 2.5, 5.0])    # ties go to the lower (even) code
_UPPER_TIE = torch.tensor([0.75, 1.75, 3.5])         # ties go to the upper (even) code
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4_quantize(x: torch.Tensor):
    R, K = x.shape
    assert K % 32 == 0
    xf = x.float().view(R, K // 32, 32)
    amax = xf.abs().amax(-1)
    _, e1 = torch.frexp(amax)                         # amax = m * 2^e1, m in [0.5, 1)
    sexp = torch.where(amax > 0, e1 - 1 - 2, torch.full_like(e1, -127)).clamp(-127, 127)
    q = torch.ldexp(xf, (-sexp).unsqueeze(-1).float())
    a = q.abs()
    lo, up = _LOWER_TIE.to(x.device), _UPPER_TIE.to(x.device)
    code = (a.unsqueeze(-1) > lo).sum(-1) + (a.unsqueeze(-1) >= up).sum(-1)
    code = code.clamp(max=7).to(torch.uint8)
    code = torch.where((q < 0) & (code > 0), code | 0x8, code).view(R, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    sf = (sexp + 127).to(torch.uint8)
    # pad to (128-row, 4-column) tiles with ZEROS before swizzling: CUTLASS reads the padded scale columns when K is
    # not a multiple of its 128-element K tile, and uninitialized bytes (e.g. 0xFF = E8M0 NaN) would poison outputs
    rp, cp = -(-R // 128) * 128, -(-(K // 32) // 4) * 4
    sfp = torch.zeros((rp, cp), dtype=torch.uint8, device=x.device)
    sfp[:R, :K // 32] = sf
    return packed, pseudo_quant.linear_to_swizzled_128_4(sfp).contiguous()


def mxfp4_dequant(packed, sf_swz, R, K):
    lin = pseudo_quant.swizzled_to_linear_128_4(sf_swz, R, K // 32)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).view(R, K).long()
    v = _E2M1.to(packed.device)[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return v.double() * torch.pow(2.0, lin.double() - 127).repeat_interleave(32, 1)


def emulation_scaled_mxfp4_mm_fusednode(a, b, sa, sb, alpha, M, N, K, out_dtype=torch.float16, F=35,
                                        m_chunk_size=256):
    from .triton_fusednode import stage234_fusednode_mx_rz_triton
    assert K % 64 == 0
    G = K // 16
    e_a = pseudo_quant.swizzled_to_linear_128_4(sa, M, K // 32).to(torch.int32) - 127
    e_b = pseudo_quant.swizzled_to_linear_128_4(sb, N, K // 32).to(torch.int32) - 127
    tab = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=a.device, dtype=torch.float16)
    unpack = lambda p, R: tab[torch.stack([p & 0xF, p >> 4], -1).view(R, K).long()]  # noqa: E731
    va, vb = unpack(a, M), unpack(b, N)
    b_g = vb.view(N, G, 16).permute(1, 2, 0)
    b_ind = (vb != 0).half().view(N, G, 16).permute(1, 2, 0)
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    for m0 in range(0, M, m_chunk_size):
        m1 = min(m0 + m_chunk_size, M)
        a_c = va[m0:m1]
        ps1 = torch.bmm(a_c.view(m1 - m0, G, 16).permute(1, 0, 2), b_g, out_dtype=torch.float32)   # exact
        cnt = torch.bmm((a_c != 0).half().view(m1 - m0, G, 16).permute(1, 0, 2), b_ind)
        out[m0:m1] = stage234_fusednode_mx_rz_triton(ps1.permute(1, 2, 0), cnt.permute(1, 2, 0), e_a[m0:m1], e_b, F=F)
        del ps1, cnt
    return (out * alpha.item()).to(out_dtype)


def emulation_scaled_mxfp4_mm_eton_fused(a, b, sa, sb, alpha, M, N, K, out_dtype=torch.float16, template=None):
    """ETON-Fused MXFP4 (one Triton kernel; see triton_eton_fused.py). template: FusedTemplate / probe json / None."""
    from .triton_eton_fused import FusedTemplate, MXFP4_SM120, eton_fused_mm
    tmpl = MXFP4_SM120 if template is None else (
        template if isinstance(template, FusedTemplate) else FusedTemplate.from_probe(template))
    assert tmpl.scale_fmt == "ue8m0" and K % 64 == 0
    return eton_fused_mm(a, b, sa, sb, alpha, M, N, K, tmpl, out_dtype)
