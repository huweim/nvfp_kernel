"""
NVFP package exports.

`scaled_fp4_ops` (compiled CUDA extension) is optional for pure-emulation flows.
When it is unavailable, pseudo-quantization helpers remain importable.
"""

from .pseudo_quant import (
    linear_to_swizzled_128_4,
    nvfp4_pseudo_quantize,
    pytorch_nvfp4_quantize,
    swizzled_to_linear_128_4,
)

_OPS_IMPORT_ERROR = None
try:
    from .ops import cutlass_scaled_fp4_mm, reciprocal_approximate_ftz_tensor, scaled_fp4_quant
except ModuleNotFoundError as exc:
    _OPS_IMPORT_ERROR = exc

    def _missing_scaled_fp4_ops(*args, **kwargs):
        raise ModuleNotFoundError(
            "scaled_fp4_ops is not available. Build/install the CUDA extension to use nvfp.ops APIs."
        ) from _OPS_IMPORT_ERROR

    scaled_fp4_quant = _missing_scaled_fp4_ops
    cutlass_scaled_fp4_mm = _missing_scaled_fp4_ops
    reciprocal_approximate_ftz_tensor = _missing_scaled_fp4_ops


__all__ = [
    "scaled_fp4_quant",
    "cutlass_scaled_fp4_mm",
    "reciprocal_approximate_ftz_tensor",
    "pytorch_nvfp4_quantize",
    "linear_to_swizzled_128_4",
    "swizzled_to_linear_128_4",
    "nvfp4_pseudo_quantize",
]
