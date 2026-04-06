"""Re-export package API for scripts that ``import ops`` from the repo root (e.g. speed_test)."""
from nvfp.ops import (  # noqa: F401
    cutlass_scaled_fp4_mm,
    get_active_backend,
    reciprocal_approximate_ftz_tensor,
    scaled_fp4_quant,
)
