"""EmulationKernel: drop-in replacement for ops.cutlass_scaled_fp4_mm (same arguments, layouts and output)."""
import torch

from . import registry


class EmulationKernel:
    """kernel(a, b, block_scale_a, block_scale_b, alpha, out_dtype) -> [M, N], bit-exact with the RTX 5090 real GEMM.

    emulation_impl: a name in emulation.registry.IMPLS (default ETON-Fused).
    accum_bits:     inter-group accumulator significand bits (36 = probed sm_120a; other values are what-if studies).
    """

    def __init__(self, emulation_impl: str = registry.DEFAULT_IMPL, accum_bits: int = registry.HW_ACCUM_BITS):
        if emulation_impl not in registry.IMPLS:
            raise ValueError(f"unknown emulation impl {emulation_impl!r}; expected one of {sorted(registry.IMPLS)}")
        self.emulation_impl = emulation_impl
        self.accum_bits = accum_bits

    @classmethod
    def for_rtx_5090(cls, emulation_impl: str = registry.DEFAULT_IMPL) -> "EmulationKernel":
        """The probed RTX 5090 template (36-bit accumulator, RZ)."""
        return cls(emulation_impl=emulation_impl, accum_bits=registry.HW_ACCUM_BITS)

    def __call__(self, a, b, block_scale_a, block_scale_b, alpha, out_dtype=torch.float16):
        self._validate_inputs(a, b, block_scale_a, block_scale_b, alpha, out_dtype)
        M, N, K = a.shape[0], b.shape[0], a.shape[1] * 2
        out = registry.run(self.emulation_impl, a, b, block_scale_a, block_scale_b, alpha, M, N, K,
                           accum_bits=self.accum_bits, out_dtype=out_dtype)
        return out.to(out_dtype)

    forward = __call__

    @staticmethod
    def _round_up(x: int, y: int) -> int:
        return (x + y - 1) // y * y

    def _validate_inputs(self, a, b, block_scale_a, block_scale_b, alpha, out_dtype) -> None:
        # same contract as ops.cutlass_scaled_fp4_mm
        assert a.is_cuda and b.is_cuda, "a and b must be CUDA tensors"
        assert block_scale_a.is_cuda and block_scale_b.is_cuda, "block scales must be CUDA tensors"
        assert alpha.is_cuda, "alpha must be a CUDA tensor"
        assert a.ndim == 2 and b.ndim == 2, "a and b must be 2D tensors"
        assert a.dtype == torch.uint8 and b.dtype == torch.uint8, "a and b must be uint8 packed FP4"
        assert out_dtype in (torch.float16, torch.bfloat16), "out_dtype must be fp16 or bf16"
        assert a.shape[1] == b.shape[1], f"k mismatch: a.shape={a.shape}, b.shape={b.shape}"
        M, N, K = a.shape[0], b.shape[0], a.shape[1] * 2
        G = K // 16
        exp_a = (self._round_up(M, 128), self._round_up(G, 4))
        exp_b = (self._round_up(N, 128), self._round_up(G, 4))
        assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2, "block scales must be 2D tensors"
        assert block_scale_a.shape == exp_a, f"scale_a must be padded+swizzled to {exp_a}, got {tuple(block_scale_a.shape)}"
        assert block_scale_b.shape == exp_b, f"scale_b must be padded+swizzled to {exp_b}, got {tuple(block_scale_b.shape)}"
