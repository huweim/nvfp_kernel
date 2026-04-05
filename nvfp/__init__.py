from typing import TYPE_CHECKING

from .ops import (
    cutlass_scaled_fp4_mm,
    reciprocal_approximate_ftz_tensor,
    scaled_fp4_quant,
)
from .pseudo_quant import (
    nvfp4_pseudo_quantize,
    pytorch_nvfp4_quantize,
    simple_fp4_pseudo_quantize,
)

if TYPE_CHECKING:
    from emulation.core import MMAEngine

__all__ = [
    "MMAEngine",
    "scaled_fp4_quant",
    "cutlass_scaled_fp4_mm",
    "reciprocal_approximate_ftz_tensor",
    "pytorch_nvfp4_quantize",
    "nvfp4_pseudo_quantize",
    "simple_fp4_pseudo_quantize",
]


def __getattr__(name: str):
    if name == "MMAEngine":
        from emulation.core import MMAEngine as _MMAEngine

        return _MMAEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
