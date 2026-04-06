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
        w_stage3: int = 34,
        w_stage4: int = 28,
        stage3_rounding: RoundStrategy = RoundStrategy.RZ,
        stage4_rounding: RoundStrategy = RoundStrategy.RZ,
        # m_chunk_size: int = 128,
        m_chunk_size: int = 2048,
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
            block_scale_a: Block scales for activation [M, K/16] (float8_e4m3fn)
            block_scale_b: Block scales for weight [N, K/16] (float8_e4m3fn)
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
        # Infer dimensions from input tensors
        if a.ndim != 2:
            a = a.reshape(-1, a.shape[-1])
        if b.ndim != 2:
            b = b.reshape(-1, b.shape[-1])
        
        a = a.view(torch.uint8)
        b = b.view(torch.uint8)

        M = a.shape[0]
        N = b.shape[0]
        K = a.shape[1] * 2  # FP4 is packed 2 values per byte
        
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
    
    @classmethod
    def for_rtx_5090(cls) -> "EmulationKernel":
        """
        Create kernel with optimal configuration for RTX 5090.
        
        Configuration: W3=34, W4=28, RZ rounding for both stages.
        This is the recommended configuration for RTX 5090 emulation.
        """
        return cls(
            w_stage3=34,
            w_stage4=28,
            stage3_rounding=RoundStrategy.RZ,
            stage4_rounding=RoundStrategy.RZ,
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
        block_scale_a: Block scales for activation [M, K/16]
        block_scale_b: Block scales for weight [N, K/16]
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
