"""
Emulation package for NVFP4 MMA accuracy modeling.

Quick Start:
    >>> from emulation import emulated_fp4_mm
    >>> output = emulated_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)
    
    >>> # Or use kernel class directly
    >>> from emulation import EmulationKernel
    >>> kernel = EmulationKernel.for_rtx_5090()
    >>> output = kernel(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)
"""
from .core import HardwareCore, MMAEngine
from .utils import NVFP4Utils, DataGenerator, DataGenerator_Abs
from .search import ParameterSearch, SearchResult
from .rounding import RoundStrategy, RoundingRegistry
from .config import HardwareConfig, CONFIG_RTX_5090
from .kernel import EmulationKernel, emulated_fp4_mm, emulated_scaled_fp4_mm

__all__ = [
    # Core classes
    'HardwareCore',
    'MMAEngine',
    
    # Utils
    'NVFP4Utils',
    'DataGenerator',
    'DataGenerator_Abs',
    
    # Search
    'ParameterSearch',
    'SearchResult',
    
    # Rounding
    'RoundStrategy',
    'RoundingRegistry',
    
    # Config
    'HardwareConfig',
    'CONFIG_RTX_5090',
    
    # Kernel API
    'EmulationKernel',
    'emulated_fp4_mm',
    'emulated_scaled_fp4_mm',
]
