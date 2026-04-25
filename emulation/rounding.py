"""
Rounding strategy enum for NVFP4 emulation.

Actual rounding logic lives in HardwareCore.to_float32_with_rounding
(core.py) so that it can be JIT-friendly and inlined.
"""
from enum import Enum


class RoundStrategy(Enum):
    """Supported rounding strategies."""
    RZ = "round_toward_zero"           # Implemented in HardwareCore
    RNE = "round_to_nearest_even"      # Implemented in HardwareCore
    RU = "round_up"                    # Not implemented
    RD = "round_down"                  # Not implemented
    RNA = "round_to_nearest_away"      # Not implemented
