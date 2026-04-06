import importlib
import os

import torch

_VALID_BACKENDS = {"auto", "real", "emulation"}


def _load_scaled_fp4_ops(required_symbols: tuple[str, ...]):
    try:
        module = importlib.import_module("scaled_fp4_ops")
    except ImportError as exc:
        raise RuntimeError(
            "scaled_fp4_ops extension is not available. "
            "Build with NVFP_BUILD_PROFILE=emulation_only or real."
        ) from exc

    missing = [name for name in required_symbols if not hasattr(module, name)]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "scaled_fp4_ops is missing required symbols: "
            f"{missing_csv}. Rebuild with a compatible NVFP_BUILD_PROFILE."
        )
    return module


def _configured_backend() -> str:
    backend = os.getenv("NVFP_GEMM_BACKEND", "auto").strip().lower()
    if backend not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(
            f"Invalid NVFP_GEMM_BACKEND={backend!r}; expected one of: {valid}"
        )
    return backend


def _real_backend_available() -> bool:
    try:
        _load_scaled_fp4_ops(("scaled_fp4_quant_sm1xxa", "cutlass_scaled_fp4_mm"))
    except RuntimeError:
        return False
    return True


def get_active_backend() -> str:
    backend = _configured_backend()
    if backend == "real" and not _real_backend_available():
        raise RuntimeError(
            "NVFP_GEMM_BACKEND=real selected, but real GEMM symbols are not available."
        )
    if backend != "auto":
        return backend
    return "real" if _real_backend_available() else "emulation"


def _get_backend_impl():
    active_backend = get_active_backend()
    if active_backend == "real":
        from . import _ops_real as impl
    elif active_backend == "emulation":
        from . import _ops_emulation as impl
    else:
        raise RuntimeError(f"Unhandled backend selection: {active_backend!r}")
    return impl, active_backend


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    impl, _ = _get_backend_impl()
    return impl.scaled_fp4_quant(input, input_global_scale)


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    impl, _ = _get_backend_impl()
    return impl.cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)


def reciprocal_approximate_ftz_tensor(x: torch.Tensor):
    y = torch.zeros_like(x)
    assert x.dtype == torch.float
    ext = _load_scaled_fp4_ops(("reciprocal_approximate_ftz_tensor",))
    ext.reciprocal_approximate_ftz_tensor(x, y)
    return y
