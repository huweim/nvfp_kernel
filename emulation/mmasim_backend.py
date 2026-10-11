"""MMA-Sim v2 (microsoft/MMA-Sim, `pip install mmasim`) as the NVFP4 GEMM backend: opt-in impl `mmasim_v2`.

For the rebuttal's end-to-end cost comparison only (ETON vs a GPU-capable bit-accurate reference model). MMA-Sim models
sm_120a `mma.sync.m16n8k64...block_scale.scale_vec::4X...ue4m3` (G=16, F=35, RZ) and its __call__ is vectorised over
any (m, n), chaining C over K in k64 steps -- the CUTLASS order. One GEMM = MMA-Sim on [M, N] with C chained over K
slices, then the CUTLASS / ETON epilogue fp16(alpha * D). Bit-exact vs real on 5090
(inference/results/rtx5090/mmasim/).

MMA-Sim broadcasts A and B to [m, n, k_slice] before its own k64 loop, so memory grows as m * n * k_slice: K is fed in
slices and rows in blocks (rows are independent) to stay under a fixed element budget; neither changes the result.
"""
from __future__ import annotations

import torch

from nvfp import pseudo_quant

QUALIFIER = "m16n8k64.block16.f32.e2m1.e2m1.f32.ue4m3"
_ELEM_BUDGET = 1.1e9     # m * n * k_slice per MMA-Sim call (measured to fit a 32 GB RTX 5090 next to the model)
_SIM = None


def _sim():
    global _SIM
    if _SIM is None:
        from mmasim.nv_ptx import MMABlockScale    # imported lazily: only this impl needs the package
        _SIM = MMABlockScale("RTX Blackwell", QUALIFIER)
    return _SIM


def mmasim_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, M, N, K, out_dtype=torch.float16):
    sim = _sim()
    G = K // 16
    dev = a_fp4.device
    s_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G)
    s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G)
    A = a_fp4.view(torch.uint8)
    B = b_fp4.view(torch.uint8).T
    SB = s_b.T
    kc = 512 if M * N * 512 <= _ELEM_BUDGET else 64
    kc = min(kc, K)
    mc = max(16, int(_ELEM_BUDGET // (N * kc)))
    out = torch.empty((M, N), dtype=torch.float32, device=dev)
    for m0 in range(0, M, mc):
        m1 = min(m0 + mc, M)
        C = torch.zeros((m1 - m0, N), dtype=torch.float32, device=dev)
        for k0 in range(0, K, kc):
            C = sim(A[m0:m1, k0 // 2:(k0 + kc) // 2], B[k0 // 2:(k0 + kc) // 2], C,
                    s_a[m0:m1, k0 // 16:(k0 + kc) // 16], SB[k0 // 16:(k0 + kc) // 16])
        out[m0:m1] = C
    return (out * alpha.reshape(1).to(dev)).to(out_dtype)
