"""
yolo_pro/stage.py
─────────────────
YOLO-Pro Stage Builder — stacks N CSP-IRBs into one backbone stage.

Backbone stage map (spec Section 2.2):
───────────────────────────────────────────────────────────────────────────
  Stage  │ n_blocks │ stride │ expand_t │ out_ch (nano) │ Output HW (320px)
  ───────┼──────────┼────────┼──────────┼───────────────┼──────────────────
    0    │    1     │   1    │    1     │      24       │    H/2  (160)
    1    │    2     │   2    │    6     │      40       │    H/4  ( 80)
    2    │    3     │   2    │    6     │      80       │    H/8  ( 40) ← P3
    3    │    3     │   1    │    6     │     112       │    H/8  ( 40)
    4    │    4     │   2    │    6     │     192       │    H/16 ( 20) ← P4
    5    │    2     │   2    │    6     │     320       │    H/32 ( 10) ← P5
───────────────────────────────────────────────────────────────────────────

Tensor flow through one stage (n_blocks=3, stride=2 example):

  Input  (B, H,   W,   C_in)
     │
     ▼ Block 0 — stride=2, expand_t (may downsample + change channels)
  (B, H/2, W/2, out_ch)
     │
     ▼ Block 1 — stride=1, expand_t, residual ON (same ch)
  (B, H/2, W/2, out_ch)
     │
     ▼ Block 2 — stride=1, expand_t, residual ON (same ch)
  (B, H/2, W/2, out_ch)
     │
  Output (B, H/2, W/2, out_ch)

Only Block 0 ever changes spatial resolution or may change channel width.
Blocks 1..N-1 are always same-shape → residual connections are always ON.
"""

import tensorflow as tf
from .csp_irb import csp_irb

# Expand ratio used for Stage 0 (t=1, no expansion) vs all other stages (t=6)
_STAGE0_EXPAND = 1
_DEFAULT_EXPAND = 6


def make_stage(
    x: tf.Tensor,
    out_ch: int,
    n_blocks: int,
    stride: int,
    name: str = "stage",
    expand_ratio: int = _DEFAULT_EXPAND,
) -> tf.Tensor:
    """
    Stack `n_blocks` CSP-InvertedResidual blocks into one backbone stage.

    Block 0  — uses `stride` and `out_ch`; may downsample and/or change channels.
    Blocks 1+ — stride=1, out_ch unchanged; residual add is always active
                (channels match after Block 0 normalises them).

    Args:
        x            : input tensor  (B, H, W, C_in)
        out_ch       : output channel count for every block in this stage.
                       Must be even (CSP half-split requirement).
        n_blocks     : number of CSP-IRB blocks to stack (≥ 1).
        stride       : spatial stride applied only in Block 0.
                       1 → no spatial change.  2 → halve H and W.
        name         : layer-name prefix, e.g. "stage0", "stage3".
        expand_ratio : MV2 inner expansion factor t.
                       Pass 1 for Stage 0 (no expansion), 6 elsewhere.

    Returns:
        Tensor  (B, H/stride, W/stride, out_ch)

    Residual behaviour per block:
        Block 0  — residual ON  iff  stride==1  AND  C_in==out_ch
        Block 1+ — residual always ON  (stride=1, channels already == out_ch)
    """
    assert n_blocks >= 1, "n_blocks must be at least 1"

    # ── Block 0 : may downsample / change channel width ──────────────────────
    x = csp_irb(
        x,
        filters_out  = out_ch,
        expand_ratio = expand_ratio,
        strides      = stride,
        name         = f"{name}_b0",
    )
    # shape → (B, H/stride, W/stride, out_ch)

    # ── Blocks 1 … n_blocks-1 : same resolution, same channels ──────────────
    # csp_irb auto-enables residual add because stride=1 and C_in==C_out.
    for i in range(1, n_blocks):
        x = csp_irb(
            x,
            filters_out  = out_ch,
            expand_ratio = expand_ratio,
            strides      = 1,
            name         = f"{name}_b{i}",
        )
    # shape unchanged → (B, H/stride, W/stride, out_ch)

    return x
