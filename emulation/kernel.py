"""
Deterministic NVFP4 Emulation Kernel API
Provides a drop-in replacement for ops.cutlass_scaled_fp4_mm with identical signature.
"""
import torch
from typing import Optional

from .core import MMAEngine, RoundStrategy
from .config import HardwareConfig


class EmulationKernel:
    """
    Deterministic NVFP4 MMA Emulation Kernel.
    
    This class provides a hardware-accurate emulation of the NVFP4 GEMM operation
    with fixed configuration parameters. Use this as a drop-in replacement for
    ops.cutlass_scaled_fp4_mm when you need deterministic, bit-accurate emulation.
    
    Example:
        >>> from emulation.kernel import EmulationKernel
        >>> kernel = EmulationKernel.for_rtx_5090()
        >>> output = kernel(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)
        
        # Or use the functional API directly:
        >>> from emulation.kernel import emulated_fp4_mm
        >>> output = emulated_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)
    """
    
    def __init__(
        self,
        config: Optional[HardwareConfig] = None,
        w_stage3: int = 36,
        # w_stage4: int = 28,
        w_stage4: int = 36,
        stage3_rounding: RoundStrategy = RoundStrategy.RZ,
        stage4_rounding: RoundStrategy = RoundStrategy.RZ,
        m_chunk_size: int = 128,
        use_triton: bool = True,
        triton_block_size: int = 256,
        triton_use_stage3: bool = True,
        triton_fuse_stage34: bool = False,
        triton_verify_stage3: bool = False,
        triton_verify_stage4: bool = False,
        emulation_impl: str = "beam_234fusion",
    ):
        """
        Initialize emulation kernel with fixed configuration.
        
        Args:
            config: HardwareConfig object (if provided, overrides other params)
            w_stage3: Bit width for Stage 3 (intra-block reduction)
            w_stage4: Bit width for Stage 4 (inter-block accumulation)
            stage3_rounding: Rounding strategy for Stage 3
            stage4_rounding: Rounding strategy for Stage 4
            m_chunk_size: Chunk size for M dimension to avoid OOM (default: 32)
        """
        if config is not None:
            self.w_stage3 = config.w_stage3
            self.w_stage4 = config.w_stage4
            self.stage3_rounding = config.stage3_rounding
            self.stage4_rounding = config.stage4_rounding
        else:
            self.w_stage3 = w_stage3
            self.w_stage4 = w_stage4
            self.stage3_rounding = stage3_rounding
            self.stage4_rounding = stage4_rounding
        self.m_chunk_size = m_chunk_size
        self.use_triton = use_triton
        self.triton_block_size = triton_block_size
        self.triton_use_stage3 = triton_use_stage3
        self.triton_fuse_stage34 = triton_fuse_stage34
        self.triton_verify_stage3 = triton_verify_stage3
        self.triton_verify_stage4 = triton_verify_stage4
        self.emulation_impl = emulation_impl
    
    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Execute emulated NVFP4 GEMM.
        
        Args:
            a: FP4 quantized activation tensor [M, K/2] (uint8 packed)
            b: FP4 quantized weight tensor [N, K/2] (uint8 packed)
            block_scale_a: Padded+swizzled block scales for activation
                [round_up(M, 128), round_up(K/16, 4)] (float8_e4m3fn)
            block_scale_b: Padded+swizzled block scales for weight
                [round_up(N, 128), round_up(K/16, 4)] (float8_e4m3fn)
            alpha: Global scale factor [1] (float32)
            out_dtype: Output dtype (float16 or bfloat16)
            
        Returns:
            Output tensor [M, N] with dtype=out_dtype
        """
        return self.forward(a, b, block_scale_a, block_scale_b, alpha, out_dtype)
    
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Execute emulated NVFP4 GEMM (same as __call__)."""
        self._validate_inputs(a, b, block_scale_a, block_scale_b, alpha, out_dtype)

        # Infer dimensions from input tensors
        M = a.shape[0]
        N = b.shape[0]
        K = a.shape[1] * 2  # FP4 is packed 2 values per byte
        
        if self.use_triton:
            if self.emulation_impl == "beam_naive_triton":
                result = MMAEngine.emulation_scaled_fp4_mm_triton(
                    a_fp4=a,
                    b_fp4=b,
                    scale_a=block_scale_a,
                    scale_b=block_scale_b,
                    alpha_tensor=alpha,
                    M=M,
                    N=N,
                    K=K,
                    W_stage3=self.w_stage3,
                    W_stage4=self.w_stage4,
                    stage3_rounding=self.stage3_rounding,
                    stage4_rounding=self.stage4_rounding,
                    m_chunk_size=self.m_chunk_size,
                    triton_block_size=self.triton_block_size,
                    triton_use_stage3=self.triton_use_stage3,
                    triton_fuse_stage34=self.triton_fuse_stage34,
                    verify_stage3=self.triton_verify_stage3,
                    verify_stage4=self.triton_verify_stage4,
                )
            elif self.emulation_impl == "beam_234fusion":
                result = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused(
                    a_fp4=a,
                    b_fp4=b,
                    scale_a=block_scale_a,
                    scale_b=block_scale_b,
                    alpha_tensor=alpha,
                    M=M,
                    N=N,
                    K=K,
                    W_stage3=self.w_stage3,
                    W_stage4=self.w_stage4,
                    stage3_rounding=self.stage3_rounding,
                    stage4_rounding=self.stage4_rounding,
                    m_chunk_size=self.m_chunk_size,
                    triton_block_size=self.triton_block_size,
                )
            elif self.emulation_impl == "beam_234fusion_bmm":
                result = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused_bmm(
                    a_fp4=a,
                    b_fp4=b,
                    scale_a=block_scale_a,
                    scale_b=block_scale_b,
                    alpha_tensor=alpha,
                    M=M,
                    N=N,
                    K=K,
                    W_stage3=self.w_stage3,
                    W_stage4=self.w_stage4,
                    stage3_rounding=self.stage3_rounding,
                    stage4_rounding=self.stage4_rounding,
                    m_chunk_size=self.m_chunk_size,
                    triton_block_size=self.triton_block_size,
                )
            else:
                raise ValueError(
                    f"Unsupported emulation_impl={self.emulation_impl!r}. "
                    "Expected 'beam_naive_triton', 'beam_234fusion', or 'beam_234fusion_bmm'."
                )
        else:
            # Run emulation with chunking to avoid OOM
            result = MMAEngine.emulation_scaled_fp4_mm(
                a_fp4=a,
                b_fp4=b,
                scale_a=block_scale_a,
                scale_b=block_scale_b,
                alpha_tensor=alpha,
                M=M,
                N=N,
                K=K,
                W_stage3=self.w_stage3,
                W_stage4=self.w_stage4,
                stage3_rounding=self.stage3_rounding,
                stage4_rounding=self.stage4_rounding,
                m_chunk_size=self.m_chunk_size,
            )
        
        # Cast to requested output dtype
        return result.to(out_dtype)

    @staticmethod
    def _round_up(x: int, y: int) -> int:
        return (x + y - 1) // y * y

    def _validate_inputs(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> None:
        # Keep emulation path aligned with ops.cutlass_scaled_fp4_mm contracts.
        assert a.is_cuda and b.is_cuda, "a and b must be CUDA tensors"
        assert block_scale_a.is_cuda and block_scale_b.is_cuda, "block scales must be CUDA tensors"
        assert alpha.is_cuda, "alpha must be a CUDA tensor"
        assert a.ndim == 2 and b.ndim == 2, "a and b must be 2D tensors"
        assert a.dtype == torch.uint8 and b.dtype == torch.uint8, "a and b must be uint8 packed FP4"
        assert out_dtype in (torch.float16, torch.bfloat16), "out_dtype must be fp16 or bf16"
        assert a.shape[1] == b.shape[1], f"k mismatch: a.shape={a.shape}, b.shape={b.shape}"

        M = a.shape[0]
        N = b.shape[0]
        K = a.shape[1] * 2
        G = K // 16
        expected_scale_a = (self._round_up(M, 128), self._round_up(G, 4))
        expected_scale_b = (self._round_up(N, 128), self._round_up(G, 4))

        assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2, "block scales must be 2D tensors"
        assert block_scale_a.shape == expected_scale_a, (
            f"scale_a must be padded+swizzled to {expected_scale_a}, got {tuple(block_scale_a.shape)}"
        )
        assert block_scale_b.shape == expected_scale_b, (
            f"scale_b must be padded+swizzled to {expected_scale_b}, got {tuple(block_scale_b.shape)}"
        )
    
    @classmethod
    def for_rtx_5090(cls, **kwargs) -> "EmulationKernel":
        """
        Preset constructor for RTX 5090: calls __init__ with W3=36, W4=36, RZ rounding.
        Any kwarg (e.g. emulation_impl, m_chunk_size) overrides the preset defaults.
        """
        return cls(
            w_stage3=36,
            # w_stage4=28,
            w_stage4=36,
            stage3_rounding=RoundStrategy.RZ,
            stage4_rounding=RoundStrategy.RZ,
            **kwargs,
        )
    
    @classmethod
    def from_config(cls, config: HardwareConfig) -> "EmulationKernel":
        """Create kernel from HardwareConfig."""
        return cls(config=config)


# =============================================================================
# Functional API (easier drop-in replacement)
# =============================================================================

def emulated_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
    config: Optional[HardwareConfig] = None,
) -> torch.Tensor:
    """
    Functional API for emulated NVFP4 GEMM.
    
    This function has the EXACT same signature as ops.cutlass_scaled_fp4_mm,
    making it a perfect drop-in replacement.
    
    Args:
        a: FP4 quantized activation tensor [M, K/2]
        b: FP4 quantized weight tensor [N, K/2]
        block_scale_a: Padded+swizzled block scales for activation
            [round_up(M, 128), round_up(K/16, 4)]
        block_scale_b: Padded+swizzled block scales for weight
            [round_up(N, 128), round_up(K/16, 4)]
        alpha: Global scale factor [1]
        out_dtype: Output dtype (default: float16)
        config: Optional HardwareConfig (uses RTX 5090 defaults if None)
        
    Returns:
        Output tensor [M, N] with dtype=out_dtype
        
    Example:
        >>> # Direct replacement of hardware kernel
        >>> output = emulated_fp4_mm(a_fp4, b_fp4, s_a, s_b, alpha, torch.float16)
        >>> 
        >>> # Or with custom config
        >>> from emulation.config import HardwareConfig, RoundStrategy
        >>> config = HardwareConfig(
        ...     w_stage3=35, w_stage4=30,
        ...     stage3_rounding=RoundStrategy.RNE,
        ...     stage4_rounding=RoundStrategy.RNE
        ... )
        >>> output = emulated_fp4_mm(a_fp4, b_fp4, s_a, s_b, alpha, torch.float16, config)
    """
    kernel = EmulationKernel.from_config(config) if config else EmulationKernel.for_rtx_5090()
    return kernel(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


# Convenience alias with same name as hardware kernel for easy swapping
emulated_scaled_fp4_mm = emulated_fp4_mm
