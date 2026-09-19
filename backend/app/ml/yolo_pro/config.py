"""
yolo_pro/config.py
──────────────────
Size-variant configuration dataclass for YOLO-Pro.

Each variant defines:
  - depths   : number of CSP-InvertedResidual blocks per stage (6 stages)
  - channels : output channel count per stage (6 stages)
  - reg_max  : DFL distribution bins (fixed at 16 across all sizes)
  - name     : human-readable tag

Stage index mapping (matches backbone Section 2.2):
  Stage 0 → Stem output refinement  (stride 1,  H/2)
  Stage 1 → First downsample        (stride 2,  H/4)
  Stage 2 → P3 feature              (stride 2,  H/8)   ← FPN tap
  Stage 3 → Mid refinement          (stride 1,  H/8)
  Stage 4 → P4 feature              (stride 2,  H/16)  ← FPN tap
  Stage 5 → P5 feature              (stride 2,  H/32)  ← FPN tap
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class YoloProConfig:
    name: str
    depths: List[int]    # len == 6, one entry per stage
    channels: List[int]  # len == 6, output channels per stage
    reg_max: int = 16    # DFL bins — fixed across all size variants

    def __post_init__(self):
        assert len(self.depths)   == 6, "depths must have exactly 6 entries"
        assert len(self.channels) == 6, "channels must have exactly 6 entries"
        assert all(d >= 1 for d in self.depths),   "each depth must be >= 1"
        assert all(c > 0  for c in self.channels), "channels must be positive"
        assert all(c % 2  == 0 for c in self.channels), \
            "all channel counts must be even (required by CSP half-split)"


# ── Registered size variants ─────────────────────────────────────────────────

CONFIGS = {
    "nano": YoloProConfig(
        name     = "nano",
        depths   = [1, 1, 2, 2, 3, 2],
        channels = [16, 24, 48, 64, 112, 192],
    ),
    "tiny": YoloProConfig(
        name     = "tiny",
        depths   = [1, 2, 3, 3, 4, 2],
        channels = [24, 40, 80, 112, 192, 320],
    ),
    "small": YoloProConfig(
        name     = "small",
        depths   = [3, 4, 6, 6, 7, 4],
        channels = [48, 64, 128, 192, 320, 512],
    ),
    "medium": YoloProConfig(
        name     = "medium",
        depths   = [4, 5, 8, 8, 9, 5],
        channels = [72, 96, 192, 288, 480, 768],
    ),
    "large": YoloProConfig(
        name     = "large",
        depths   = [6, 8, 12, 12, 14, 8],
        channels = [108, 144, 288, 432, 720, 1152],
    ),
}


def get_config(size: str) -> YoloProConfig:
    """
    Retrieve a config by name.

    Args:
        size: one of 'nano' | 'tiny' | 'small' | 'medium' | 'large'

    Returns:
        YoloProConfig (frozen dataclass)

    Raises:
        ValueError with a helpful message listing valid options.
    """
    size = size.lower().strip()
    if size not in CONFIGS:
        valid = ", ".join(f"'{k}'" for k in CONFIGS)
        raise ValueError(f"Unknown size '{size}'. Valid options: {valid}")
    return CONFIGS[size]
