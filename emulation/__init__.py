"""NVFP4 / MXFP4 GEMM emulation (ETON).

    >>> from emulation import EmulationKernel
    >>> out = EmulationKernel.for_rtx_5090()(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)

Implementations: emulation.registry.IMPLS (default ETON-Fused, `beam_eton_fused`).
"""
from .core import HardwareCore, MMAEngine
from .utils import NVFP4Utils, DataGenerator
from .rounding import RoundStrategy
from .kernel import EmulationKernel
from .registry import IMPLS, DEFAULT_IMPL

__all__ = ["HardwareCore", "MMAEngine", "NVFP4Utils", "DataGenerator", "RoundStrategy", "EmulationKernel", "IMPLS",
           "DEFAULT_IMPL"]
