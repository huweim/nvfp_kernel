"""Loader of the compiled scaled_fp4_ops extension (setup.py; NVFP_BUILD_PROFILE=real | emulation_only)."""
import importlib


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
