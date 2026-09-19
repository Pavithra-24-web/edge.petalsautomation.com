# anchor_generator.py
# Step 8 — SSD Anchor Generator
# MobileNetV2 SSD FPN-Lite 320x320

from __future__ import annotations

import numpy as np
import tensorflow as tf

from primitives import CFG


# ============================================================================
# Anchor specification
# ============================================================================
#
# The 6-level FPN-Lite grid with ANCHORS_PER_LOC = (3, 6, 6, 6, 6, 6) produces
# exactly 8,028 anchors — the true base grid for this architecture:
#
#   F3: 40×40×3  =  4,800
#   F4: 20×20×6  =  2,400
#   F5: 10×10×6  =    600
#   F6:  5× 5×6  =    150
#   F7:  3× 3×6  =     54
#   F8:  2× 2×6  =     24
#                Σ =  8,028
#
# Scale schedule: linear interpolation from min_scale=0.20 to max_scale=0.95
# across 6 levels, matching the TF OD API SSDLite MobileNetV2 pipeline.config:
#
#   s_i = min_scale + (max_scale - min_scale) * i / (num_layers - 1)
#   half-step scale: sqrt(s_i * s_{i+1})
#
# Aspect ratios:
#   Level 0 (F3, 40×40): [1.0, 2.0, 0.5]                — 3 anchors/loc
#                          reduce_boxes_in_lowest_layer=True, no half-step
#   Levels 1–5 (F4–F8):  [1.0, 2.0, 3.0, 0.5, 1/3]     — 5 ARs + 1 half-step = 6/loc

_NUM_LAYERS           = 6
_MIN_SCALE            = 0.20
_MAX_SCALE            = 0.95
_ASPECT_RATIOS_BASE   = [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0]   # levels 1-5
_ASPECT_RATIOS_LEVEL0 = [1.0, 2.0, 0.5]                     # level 0 only
_GRID_SIZES           = list(CFG.FEATURE_MAP_SIZES)          # [40, 20, 10, 5, 3, 2]


# ============================================================================
# Scale computation
# ============================================================================

def _compute_scales(
    min_scale: float = _MIN_SCALE,
    max_scale: float = _MAX_SCALE,
    num_layers: int  = _NUM_LAYERS,
) -> list[float]:
    """
    Compute anchor scales for each layer using linear interpolation.

    scale_i = min_scale + (max_scale - min_scale) * i / (num_layers - 1)

    Returns list of length num_layers + 1 (includes endpoint for half-step).
    """
    return [
        min_scale + (max_scale - min_scale) * i / (num_layers - 1)
        for i in range(num_layers + 1)
    ]


# ============================================================================
# Per-level anchor generation
# ============================================================================

def _anchors_for_level(
    grid_size: int,
    scale: float,
    scale_next: float,
    aspect_ratios: list[float],
    add_half_step: bool,
) -> np.ndarray:
    """
    Generate all anchors for one feature map level.

    For each cell centre (cy, cx) in the grid, generates:
      - One anchor per aspect_ratio:
            h = scale / sqrt(ar)
            w = scale * sqrt(ar)
      - Optionally one half-step anchor (ar=1.0 at interpolated scale):
            h = w = sqrt(scale * scale_next)

    Cell centres:
        cx = (i + 0.5) / grid_size  for i in range(grid_size)
        cy = (j + 0.5) / grid_size  for j in range(grid_size)

    All values normalised to [0, 1] (fraction of input image side).

    Args:
        grid_size:     Spatial side length of this feature map.
        scale:         Anchor scale at this level.
        scale_next:    Anchor scale at next level (for half-step).
        aspect_ratios: List of w/h aspect ratios.
        add_half_step: Whether to append the sqrt(s * s_next) anchor.

    Returns:
        np.ndarray of shape (grid_size² × anchors_per_loc, 4)
        in [cy, cx, h, w] format, values in (0, 1].
    """
    # Build anchor (h, w) templates for this level
    templates: list[tuple[float, float]] = []
    for ar in aspect_ratios:
        h = scale / (ar ** 0.5)
        w = scale * (ar ** 0.5)
        templates.append((h, w))

    if add_half_step:
        hs = (scale * scale_next) ** 0.5
        templates.append((hs, hs))

    # Grid cell centres
    offsets = (np.arange(grid_size) + 0.5) / grid_size   # (grid_size,)

    anchors = []
    for cy in offsets:
        for cx in offsets:
            for (h, w) in templates:
                anchors.append([cy, cx, h, w])

    return np.array(anchors, dtype=np.float32)


# ============================================================================
# Main generator
# ============================================================================

def generate_anchors(
    grid_sizes:         list[int]   = _GRID_SIZES,
    min_scale:          float       = _MIN_SCALE,
    max_scale:          float       = _MAX_SCALE,
    aspect_ratios_base: list[float] = _ASPECT_RATIOS_BASE,
    aspect_ratios_l0:   list[float] = _ASPECT_RATIOS_LEVEL0,
    target_count:       int         = CFG.TOTAL_ANCHORS,
) -> tf.Tensor:
    """
    Generate all SSD prior boxes for the MobileNetV2 SSD FPN-Lite 320×320 model.

    Implements the TF OD API MultipleGridAnchorGenerator behaviour:
      - Linear scale schedule from min_scale to max_scale
      - 5 aspect ratios + 1 half-step for levels 1-5  (6 anchors/loc)
      - 3 aspect ratios, no half-step for level 0      (3 anchors/loc)
        [reduce_boxes_in_lowest_layer = True]

    Anchor count per level:
        40×40×3 = 4,800
        20×20×6 = 2,400
        10×10×6 =   600
         5× 5×6 =   150
         3× 3×6 =    54
         2× 2×6 =    24
         Σ      = 8,028  (== CFG.TOTAL_ANCHORS)

    Args:
        grid_sizes:         Spatial sizes per level [40, 20, 10, 5, 3, 2].
        min_scale:          Smallest anchor scale (fraction of input size).
        max_scale:          Largest anchor scale.
        aspect_ratios_base: ARs for levels 1-5.
        aspect_ratios_l0:   ARs for level 0 (fewest anchors/loc).
        target_count:       Expected total (8,028). AssertionError if mismatched.

    Returns:
        tf.Tensor of shape (8028, 4) in [cy, cx, h, w] format,
        dtype float32, all values in (0, 2] (centres in (0,1);
        large-scale h/w may slightly exceed 1.0, which is valid for SSD).
    """
    scales = _compute_scales(min_scale, max_scale, len(grid_sizes))
    # scales has len(grid_sizes) + 1 entries; scales[i] and scales[i+1]
    # bound the half-step for level i.

    all_anchors: list[np.ndarray] = []

    for i, gs in enumerate(grid_sizes):
        s_curr = scales[i]
        s_next = scales[i + 1]

        if i == 0:
            # Lowest level: reduce_boxes_in_lowest_layer
            # → 3 fixed ARs, no half-step → 3 anchors/loc
            level_anchors = _anchors_for_level(
                grid_size=gs,
                scale=s_curr,
                scale_next=s_next,
                aspect_ratios=aspect_ratios_l0,
                add_half_step=False,
            )
        else:
            # Levels 1-5: 5 ARs + 1 half-step → 6 anchors/loc
            level_anchors = _anchors_for_level(
                grid_size=gs,
                scale=s_curr,
                scale_next=s_next,
                aspect_ratios=aspect_ratios_base,
                add_half_step=True,
            )

        all_anchors.append(level_anchors)

    anchors_np = np.concatenate(all_anchors, axis=0)

    # Verify exact count — this must match CFG.TOTAL_ANCHORS = 8,028
    assert anchors_np.shape[0] == target_count, (
        f"Anchor count mismatch: generated {anchors_np.shape[0]}, "
        f"expected {target_count}. Check grid_sizes and ANCHORS_PER_LOC."
    )

    # Clip centres to valid (0, 1); allow h/w up to 2.0 for large-scale anchors
    anchors_np = np.clip(anchors_np, 0.0, 2.0)
    # Re-ensure centres stay strictly inside (0, 1)
    anchors_np[:, 0] = np.clip(anchors_np[:, 0], 1e-6, 1.0 - 1e-6)
    anchors_np[:, 1] = np.clip(anchors_np[:, 1], 1e-6, 1.0 - 1e-6)

    return tf.constant(anchors_np, dtype=tf.float32)   # (8028, 4)


# ============================================================================
# Sanity checks
# ============================================================================

def check_anchors(anchors: tf.Tensor, target_count: int = CFG.TOTAL_ANCHORS) -> None:
    """
    Run a suite of sanity checks on the generated anchor tensor.

    Checks:
      1. Shape is (target_count, 4)
      2. dtype is float32
      3. Centres (cy, cx) are within (0, 1)
      4. Sizes (h, w) are positive
      5. No NaN / Inf values

    Args:
        anchors:      Anchor tensor from generate_anchors().
        target_count: Expected number of anchors (default CFG.TOTAL_ANCHORS = 8,028).

    Raises:
        AssertionError on any failure.
    """
    a = anchors.numpy() if isinstance(anchors, tf.Tensor) else anchors

    assert a.shape == (target_count, 4), \
        f"Shape mismatch: expected ({target_count}, 4), got {a.shape}"
    assert a.dtype == np.float32, \
        f"dtype mismatch: expected float32, got {a.dtype}"

    cy, cx, h, w = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    assert np.all(cy > 0.0) and np.all(cy < 1.0), \
        f"cy out of range: min={cy.min():.4f} max={cy.max():.4f}"
    assert np.all(cx > 0.0) and np.all(cx < 1.0), \
        f"cx out of range: min={cx.min():.4f} max={cx.max():.4f}"
    assert np.all(h > 0.0), f"Non-positive h: min={h.min():.4f}"
    assert np.all(w > 0.0), f"Non-positive w: min={w.min():.4f}"
    assert not np.any(np.isnan(a)), "NaN values in anchors"
    assert not np.any(np.isinf(a)), "Inf values in anchors"

    print(f"  ✓ count  : {a.shape[0]:,}  (target {target_count:,})")
    print(f"  ✓ dtype  : {a.dtype}")
    print(f"  ✓ cy     : min={cy.min():.4f}  max={cy.max():.4f}")
    print(f"  ✓ cx     : min={cx.min():.4f}  max={cx.max():.4f}")
    print(f"  ✓ h      : min={h.min():.4f}  max={h.max():.4f}")
    print(f"  ✓ w      : min={w.min():.4f}  max={w.max():.4f}")
    print(f"  ✓ no NaN / Inf")


# ============================================================================
# Smoke test  (python anchor_generator.py)
# ============================================================================

if __name__ == "__main__":

    print("=== Anchor Generator — sanity checks ===\n")

    scales = _compute_scales()
    print("Scale schedule:")
    for i, s in enumerate(scales[:-1]):
        s_next = scales[i + 1]
        s_half = (s * s_next) ** 0.5
        print(f"  level {i}:  s={s:.4f}  s_next={s_next:.4f}  s_half={s_half:.4f}")

    print("\nPer-level anchor counts:")
    grid_sizes = _GRID_SIZES
    base_counts = []
    for i, gs in enumerate(grid_sizes):
        a_per_loc = len(_ASPECT_RATIOS_LEVEL0) if i == 0 else len(_ASPECT_RATIOS_BASE) + 1
        count = gs * gs * a_per_loc
        base_counts.append(count)
        print(f"  F{i+3} ({gs:>2}×{gs:<2}): {a_per_loc} anchors/loc  →  {count:>5,}")
    total = sum(base_counts)
    print(f"  {'Total':>12}: {total:>5,}  (CFG.TOTAL_ANCHORS = {CFG.TOTAL_ANCHORS:,})")
    assert total == CFG.TOTAL_ANCHORS, f"Count mismatch: {total} != {CFG.TOTAL_ANCHORS}"

    print("\nGenerating anchors …")
    anchors = generate_anchors()

    print(f"\nAnchor tensor shape: {anchors.shape}")
    print("\nSanity checks:")
    check_anchors(anchors)

    print("\nSample anchors (first 3, last 3):")
    a = anchors.numpy()
    for idx in list(range(3)) + list(range(-3, 0)):
        cy, cx, h, w = a[idx]
        print(f"  [{idx:>6}]  cy={cy:.4f}  cx={cx:.4f}  h={h:.4f}  w={w:.4f}")

    print("\n✓ Anchor generator complete")
