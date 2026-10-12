"""Bit-exactness of every registered NVFP4 GEMM implementation (+ ETON-Fused MXFP4).

    pytest -q tests/test_impls.py            (from inference/quant/nvfp_kernel)

Oracle: on the RTX 5090 (scaled_fp4_ops built with the real kernels) the CUTLASS NVFP4 GEMM; elsewhere the staged
fused-node implementation `beam_fusednode`, which has the same numerics as the default `beam_eton_fused`.
`beam_base` (ETON-Base) implements the older two-step W3/W4 model: bit-exact on ordinary inputs (checked here on
Gaussian data), not under large inter-group range with cancellation.
"""
import pytest
import torch

from emulation import registry
from nvfp import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SHAPES = [(1, 256, 512), (77, 384, 1024), (300, 512, 2048)]


def _real_available():
    try:
        ops._load_scaled_fp4_ops(("cutlass_scaled_fp4_mm",))
        return True
    except RuntimeError:
        return False


def _operands(M, N, K, seed, dist):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", generator=g, dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float16)
    if dist == "block_range":   # per-16-block scales over a wide range: exercises the fused node's alignment
        a = a * torch.pow(2.0, torch.randint(-10, 10, (M, K // 16), device="cuda", generator=g)
                          .repeat_interleave(16, 1).to(torch.float16))
    ga = torch.tensor([448.0 * 6.0 / a.abs().max().float().item()], device="cuda")
    gb = torch.tensor([448.0 * 6.0 / b.abs().max().float().item()], device="cuda")
    a4, sa = ops.scaled_fp4_quant(a, ga)
    b4, sb = ops.scaled_fp4_quant(b, gb)
    return a4, b4, sa, sb, torch.tensor([1.0 / (ga.item() * gb.item())], device="cuda")


def _oracle(a4, b4, sa, sb, al, M, N, K):
    if _real_available():
        return ops.cutlass_scaled_fp4_mm(a4, b4, sa, sb, al, torch.float16)
    return registry.run("beam_fusednode", a4, b4, sa, sb, al, M, N, K)


def _eq(x, y):
    return torch.equal(x.view(torch.int16), y.view(torch.int16))


@pytest.mark.parametrize("M,N,K", SHAPES)
@pytest.mark.parametrize("dist", ["gaussian", "block_range"])
@pytest.mark.parametrize("impl", ["beam_eton_fused", "beam_fusednode", "beam_base"])
def test_nvfp4_impl_bitexact(impl, dist, M, N, K):
    if impl == "beam_base" and dist != "gaussian":
        pytest.skip("ETON-Base is the two-step model (see module docstring)")
    a4, b4, sa, sb, al = _operands(M, N, K, seed=M * 7 + N, dist=dist)
    assert _eq(registry.run(impl, a4, b4, sa, sb, al, M, N, K), _oracle(a4, b4, sa, sb, al, M, N, K))


def test_mmasim_v2_if_installed():
    pytest.importorskip("mmasim")
    M, N, K = SHAPES[1]
    a4, b4, sa, sb, al = _operands(M, N, K, seed=1, dist="block_range")
    assert _eq(registry.run("mmasim_v2", a4, b4, sa, sb, al, M, N, K), _oracle(a4, b4, sa, sb, al, M, N, K))


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_mxfp4_eton_fused_vs_fusednode(M, N, K):
    from emulation import mxfp4
    g = torch.Generator(device="cuda").manual_seed(K)
    A = torch.randn(M, K, device="cuda", generator=g, dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", generator=g, dtype=torch.float16)
    a4, sa = mxfp4.mxfp4_quantize(A)
    b4, sb = mxfp4.mxfp4_quantize(B)
    al = torch.tensor([1.0], device="cuda")
    assert _eq(mxfp4.emulation_scaled_mxfp4_mm_eton_fused(a4, b4, sa, sb, al, M, N, K),
               mxfp4.emulation_scaled_mxfp4_mm_fusednode(a4, b4, sa, sb, al, M, N, K))
