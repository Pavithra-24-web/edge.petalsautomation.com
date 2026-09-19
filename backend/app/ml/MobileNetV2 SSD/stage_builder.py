# stage_builder.py
# Step 4 — MobileNetV2 Stage Builder
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from primitives import CFG, make_divisible
from inverted_residual import inverted_residual_block


# ============================================================================
# Stage Builder
# ============================================================================

def make_stage(
    x: tf.Tensor,
    t: int,
    c: int,
    n: int,
    s: int,
    name: str = "stage",
) -> tf.Tensor:
    """
    Build one MobileNetV2 stage: a sequence of `n` inverted residual blocks.

    Convention matches the reference paper Table 2:
        t — expansion factor  (applied to every block in this stage)
        c — output channels   (same for every block in this stage)
        n — number of blocks
        s — stride            (applied to block 0 only; blocks 1..n-1 use s=1)

    Stride rule:
        Block 0  →  stride = s   (may spatially downsample)
        Blocks 1…n-1  →  stride = 1  (residual-eligible if C_in == C_out)

    Because block 0 changes spatial resolution (and often channel count),
    it never carries a residual connection.  Blocks 1…n-1 operate at the
    same resolution and channel width, so the residual fires automatically
    inside `inverted_residual_block` when conditions are met.

    Shape flow  (generic, stride s, C_in channels in, c channels out):
        Input        : (B, H,    W,    C_in)
        After block 0: (B, H/s,  W/s,  c)      stride=s, no residual
        After block 1: (B, H/s,  W/s,  c)      stride=1, residual ✓
        …
        After block n: (B, H/s,  W/s,  c)      stride=1, residual ✓

    Args:
        x:    Input tensor from the previous stage or stem.
        t:    Expansion factor for every IRB in this stage.
        c:    Output channel count (made divisible by 8 for hardware alignment).
        n:    Number of inverted residual blocks.
        s:    Stride for the first block only.
        name: Name prefix; sub-blocks are named <name>/block_0 … block_n-1.

    Returns:
        Output tensor of shape (B, H//s, W//s, c).
    """
    out_channels = make_divisible(c, divisor=8)   # hardware-friendly alignment

    for i in range(n):
        stride = s if i == 0 else 1               # only block 0 may downsample
        x = inverted_residual_block(
            x,
            expansion=t,
            out_channels=out_channels,
            strides=stride,
            name=f"{name}_block_{i}",
        )

    return x                                      # (B, H//s, W//s, c)


# ============================================================================
# Smoke test  (python stage_builder.py)
# ============================================================================

if __name__ == "__main__":

    # Reference MobileNetV2 stage table (paper Table 2, post-stem):
    #
    #  stage | t |  c  | n | s  | input spatial | output spatial
    #  ------+---+-----+---+----+---------------+---------------
    #    1   | 1 |  16 | 1 | 1  |   160×160     |   160×160
    #    2   | 6 |  24 | 2 | 2  |   160×160     |    80×80
    #    3   | 6 |  32 | 3 | 2  |    80×80      |    40×40   ← C3 tap
    #    4   | 6 |  64 | 4 | 2  |    40×40      |    20×20
    #    5   | 6 |  96 | 3 | 1  |    20×20      |    20×20   ← C4 tap
    #    6   | 6 | 160 | 3 | 2  |    20×20      |    10×10
    #    7   | 6 | 320 | 1 | 1  |    10×10      |    10×10   ← C5 tap

    STAGE_CFG = [
        # ( t,   c,  n, s,  in_spatial, in_ch,  stage_name )
        (  1,   16,  1, 1,       160,     32,   "stage_1" ),
        (  6,   24,  2, 2,       160,     32,   "stage_2" ),
        (  6,   32,  3, 2,        80,     24,   "stage_3" ),
        (  6,   64,  4, 2,        40,     32,   "stage_4" ),
        (  6,   96,  3, 1,        20,     64,   "stage_5" ),
        (  6,  160,  3, 2,        20,     96,   "stage_6" ),
        (  6,  320,  1, 1,        10,    160,   "stage_7" ),
    ]

    print("=== Stage Builder — shape tests ===\n")
    print(f"  {'stage':<10} {'config':<22} {'input':>14}  →  {'output':<20}  params")

    for t, c, n, s, sp, in_ch, sname in STAGE_CFG:
        inp    = keras.Input(shape=(sp, sp, in_ch))
        out    = make_stage(inp, t=t, c=c, n=n, s=s, name=sname)
        model  = keras.Model(inp, out, name=sname)

        dummy  = np.random.uniform(-1, 1, (1, sp, sp, in_ch)).astype("float32")
        result = model(dummy, training=False).numpy()

        out_sp = sp // s
        out_ch = make_divisible(c)
        assert result.shape == (1, out_sp, out_sp, out_ch), \
            f"[{sname}] Expected (1,{out_sp},{out_sp},{out_ch}), got {result.shape}"

        cfg_str   = f"t={t} c={c} n={n} s={s}"
        in_str    = f"({sp}×{sp}×{in_ch})"
        out_str   = f"({out_sp}×{out_sp}×{out_ch})"
        print(f"  {sname:<10} {cfg_str:<22} {in_str:>14}  →  {out_str:<20}  {model.count_params():,}")

    print("\n✓ All stage builder tests passed")
