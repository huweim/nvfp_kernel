"""
Hardware Configuration for NVFP4 Emulation
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .rounding import RoundStrategy


@dataclass
class HardwareConfig:
    """
    Hardware-specific configuration for NVFP4 emulation.
    
    Attributes:
        w_stage3: Bit width for Stage 3 (intra-block 4-to-1 reduction)
        w_stage4: Bit width for Stage 4 (inter-block accumulation)
        stage3_rounding: Rounding strategy for Stage 3
        stage4_rounding: Rounding strategy for Stage 4
        name: Optional name for this configuration
        description: Optional description
    """
    w_stage3: int = 36
    w_stage4: int = 36
    stage3_rounding: RoundStrategy = RoundStrategy.RZ
    stage4_rounding: RoundStrategy = RoundStrategy.RZ
    name: str = "custom"
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        if self.w_stage3 < 20 or self.w_stage3 > 50:
            raise ValueError(f"w_stage3 should be in [20, 50], got {self.w_stage3}")
        if self.w_stage4 < 20 or self.w_stage4 > 50:
            raise ValueError(f"w_stage4 should be in [20, 50], got {self.w_stage4}")
    
    @classmethod
    def for_rtx_5090(cls) -> "HardwareConfig":
        """
        Optimal configuration for NVIDIA RTX 5090.
        
        Verified with 10,000+ test iterations across various matrix sizes
        and data distributions.
        """
        return cls(
            w_stage3=36,
            w_stage4=36,
            stage3_rounding=RoundStrategy.RZ,
            stage4_rounding=RoundStrategy.RZ,
            name="RTX_5090",
            description="NVIDIA RTX 5090 with NVFP4 input, FP16 output"
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "HardwareConfig":
        """Create from dictionary."""
        # Convert string rounding to enum
        if "stage3_rounding" in d and isinstance(d["stage3_rounding"], str):
            d["stage3_rounding"] = RoundStrategy[d["stage3_rounding"]]
        if "stage4_rounding" in d and isinstance(d["stage4_rounding"], str):
            d["stage4_rounding"] = RoundStrategy[d["stage4_rounding"]]
        return cls(**d)

    @classmethod
    def from_probe_report(
        cls,
        path: Union[str, Path],
        *,
        w_stage4: Optional[int] = None,
        stage4_rounding: Optional[RoundStrategy] = None,
        name: Optional[str] = None,
    ) -> "HardwareConfig":
        """Build a HardwareConfig from a §5.2 ProbeReport JSON
        (see numerical_attribute_modeling/probe_report.py).

        The probe characterizes Stage 3 = intra-instruction inter-group
        reduction (paper §5.2). Stage 4 = cross-instruction GEMM accumulation
        is *not* directly probed by §5.2, so callers may override w_stage4 /
        stage4_rounding; defaults mirror Stage 3.
        """
        with open(path, "r") as f:
            d = json.load(f)

        sv = d.get("schema_version")
        if sv != "1":
            raise ValueError(f"unsupported probe-report schema_version {sv!r}; "
                             f"this build of HardwareConfig understands '1'")

        inter = d.get("inter_group")
        if not inter:
            raise ValueError("probe report has no inter_group section — "
                             "Stage 3 cannot be configured from this report")

        bits = inter.get("accum_bits")
        rounding_str = inter.get("rounding")
        if bits is None or rounding_str is None:
            raise ValueError("inter_group must specify accum_bits and rounding "
                             "to drive Stage 3 of the emulation pipeline")

        try:
            stage3_rounding = RoundStrategy[rounding_str]
        except KeyError:
            raise ValueError(f"inter_group.rounding {rounding_str!r} not in "
                             f"RoundStrategy enum")

        return cls(
            w_stage3=int(bits),
            w_stage4=int(w_stage4 if w_stage4 is not None else bits),
            stage3_rounding=stage3_rounding,
            stage4_rounding=stage4_rounding if stage4_rounding is not None else stage3_rounding,
            name=name or d.get("device", {}).get("gpu_name", "from_probe_report"),
            description=(
                f"loaded from {path}; "
                f"format={d.get('instruction', {}).get('format', '?')}; "
                f"arch={d.get('device', {}).get('arch', '?')}"
            ),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "w_stage3": self.w_stage3,
            "w_stage4": self.w_stage4,
            "stage3_rounding": self.stage3_rounding.name,
            "stage4_rounding": self.stage4_rounding.name,
            "name": self.name,
            "description": self.description,
        }


# Predefined configurations
CONFIG_RTX_5090 = HardwareConfig.for_rtx_5090()
