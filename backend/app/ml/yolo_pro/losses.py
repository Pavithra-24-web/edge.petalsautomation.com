# """
# yolo_pro/losses.py
# ──────────────────
# YOLO-Pro Training Losses.

#     ┌──────────────────────────────────────────────────────────────────────────┐
#     │  Loss          Target signal          Branch         Weight              │
#     ├──────────────────────────────────────────────────────────────────────────┤
#     │  Varifocal     IoU-weighted cls label  cls_out        1.0  ← was 0.5    │
#     │  CIoU          Ground-truth box (xyxy)  reg_out (dec)  2.0               │
#     │  DFL           Soft bin label for ltrb  reg_out (raw)  1.5               │
#     └──────────────────────────────────────────────────────────────────────────┘

# Loss weight rationale (localization + classification balance)
# ─────────────────────────────────────────────────────────────
#   OG Edge Impulse YOLO-Pro uses cls=1.0, box=2.0, dfl=1.5.  With these
#   weights VFL dominates the gradient budget (~88 % of loss) — intentional:
#   BOX_WEIGHT_FLOOR=0.10 independently guards box gradient when VFL quality
#   is near zero, removing the need for a higher box weight.  Reverting from
#   box=7.5 (YOLOv8 convention) restores cls to ~22 % of the gradient budget
#   (vs ~10 % at 7.5) and closes the mAP gap vs OG.
#   However, lowering cls to 0.5 then gave the classification head only ~5 % of
#   the gradient budget (0.5 / (0.5+7.5+1.5) = 5 %).  On small datasets with
#   short training budgets (~1 920 steps) this proved insufficient: the cls head
#   converged too slowly, producing tp_score_mean < fp_score_mean and collapsed
#   mAP despite good recall and IoU.

#   Root cause of the ranking failure:
#     • With cls=0.5 the head barely discriminates between classes by the time
#       box regression plateaus and val_loss stops improving (epoch ~57).
#     • Near-object background anchors (never assigned as positives) retain
#       uninformative cls outputs that can exceed the correct-class output of
#       true-positive anchors.
#     • Per-class NMS does not suppress these cross-class FPs, so they
#       outscore TPs in the AP sort → precision collapses.

#   Conservative fix — cls=1.0:
#     • Doubles the cls gradient share from 5 % to ~10 % (1.0/10.0).
#     • Box regression still dominates (7.5/10.0 = 75 %) — localization is
#       preserved.
#     • The ~10 % cls share is sufficient for the head to reach meaningful
#       score separation within the available training budget.
#     • Gradient clipping (global_norm=10.0) prevents instability.

# Normalization
# ─────────────
#   All losses are normalized by max(num_positives, 1) — the number of
#   foreground cells in the batch.  VFL is summed (not meaned over all N
#   cells) so its gradient magnitude scales with positive count just like
#   the box / DFL terms.  This is the YOLOv8 / RT-DETR convention and
#   prevents the overwhelming majority of negatives from crushing the
#   positive signal.
# """

# import tensorflow as tf


# WEIGHT_CLS = 1.0
# WEIGHT_BOX = 2.0
# WEIGHT_DFL = 1.5

# # Minimum per-anchor weight for CIoU and DFL losses.
# # When VFL quality (= pred-box IoU) is near 0 early in training, raw
# # box_weights collapse to near-zero and stall box convergence.  This floor
# # is the sole guard: TAL no longer applies a quality floor to cls_targets,
# # so BOX_WEIGHT_FLOOR is what keeps box gradient alive while localization
# # bootstraps.
# BOX_WEIGHT_FLOOR: float = 0.10

# # Minimum per-cell weight for the VFL *negative* branch (target == 0).
# #
# # The unmodified VFL negative weight is ``alpha * pred^gamma`` — a focal
# # down-weighting that collapses quadratically as pred approaches the prior.
# # This is the right behaviour for confident background cells (pred near 0
# # already → no need to push further) but is too soft for cells whose pred
# # has drifted *upward* but is still well below 1.0:
# #
# #   pred = 0.01  →  weight = 0.75 · 1e-4  ≈ 7.5e-5   (negligible)
# #   pred = 0.10  →  weight = 0.75 · 1e-2  ≈ 7.5e-3   (still soft)
# #   pred = 0.15  →  weight = 0.75 · 2.25e-2 ≈ 1.7e-2
# #
# # Observed failure mode (EP20→EP31): fp_score_mean drifted 0.093→0.1496
# # while tp_score_mean > fp_score_mean and preds_per_image stayed at 15–17.
# # Ranking is intact and prediction volume is controlled, but background
# # scores keep climbing because the focal factor leaves almost no
# # restoring force in the 0.05–0.15 drift band.
# #
# # A small additive floor restores a baseline push-down in that band while
# # preserving:
# #   • the positive branch (target > 0)   — TP gradient is unchanged,
# #   • confidence-aware focusing — the original alpha * pred^gamma term
# #     remains present, with a small baseline added for negatives,
# #   • the cls-deferral behaviour at the prior — the floor is two orders
# #     of magnitude below the positive-branch gradient at any reasonable
# #     IoU target, so TAL-positive anchors still dominate the loss.
# #
# # Calibrated against the drift band, not the flood band: the floor only
# # adds gradient pressure where the focal factor is too small to act and
# # the existing Layer 1 / Layer 2 gates still own the flood-recovery path.
# VFL_NEG_WEIGHT_FLOOR: float = 0.02

# # Probability clamp for the Varifocal BCE.  MUST be large enough that
# # ``1.0 - VFL_PROB_EPS`` is a DIFFERENT float32 from 1.0.
# #
# # This is not a style preference — it is the fix for a scale-invariant NaN that
# # killed every mixed-precision run.  The previous value, 1e-9, is smaller than
# # half of float32's spacing at 1.0 (5.96e-08), so ``1.0 - 1e-9`` rounds back to
# # exactly 1.0 and ``tf.clip_by_value(p, eps, 1.0 - eps)`` had NO upper bound at
# # all.  The chain that followed:
# #
# #   1. Under ``mixed_float16`` the cls head's Sigmoid emits float16, which
# #      saturates to EXACTLY 1.0 at a logit of 7.623 — measured, and lower
# #      than the value alone suggests because fp16's spacing ABOVE 1.0 is
# #      2**-10, so the `1 + exp(-x)` denominator itself rounds to 1.0.  float32
# #      needs a logit of 17.327, which is why the fp32/CPU path almost never
# #      tripped this and the MP path trips it routinely.
# #   2. p == 1.0 passes the no-op clip untouched.
# #   3. ``tf.math.log(1.0 - p)`` = log(0) = -inf in the FORWARD pass.
# #   4. The ``tf.where(is_finite(...))`` sanitiser below masks that -inf out of
# #      the reported loss, so ``total_loss`` stays finite and the raw head
# #      outputs stay finite — the two signals the training loop trusts.
# #   5. The Log gradient is ``1/(1 - p)`` = 1/0 = inf, and the sanitiser's
# #      ``where`` routes a 0 upstream gradient into it: 0 * inf = NaN.
# #
# # A NaN is invariant under multiplication, so the LossScaleOptimizer's dynamic
# # scale could never clear it: every step overflowed, the scale halved on every
# # step from 2**15 down through the subnormals to 0, and steps kept being
# # skipped forever with a scale of 0.
# #
# # 1e-7 is float32-representable as an offset from 1.0 (giving 0.99999988) and
# # matches ``keras.backend.epsilon()``.  It also bounds the Log gradient: a
# # float16 head can only emit 1.0 or <= 0.99951 (fp16 spacing below 1.0 is
# # 4.88e-04), and the 1.0 case is now genuinely clipped — ``ClipByValue`` routes
# # a zero gradient for out-of-range inputs — so the largest ``1/(1-p)`` that can
# # reach the head is ~2.0e3, not inf.
# VFL_PROB_EPS: float = 1e-7

# # Fail loudly at import if the clamp ever regresses to a value float32 cannot
# # distinguish from 1.0 — the failure it guards against is silent by
# # construction (finite loss, finite head outputs, NaN gradients only).
# if float(tf.constant(1.0, tf.float32) - tf.constant(VFL_PROB_EPS, tf.float32)) >= 1.0:
#     raise ValueError(
#         f"VFL_PROB_EPS={VFL_PROB_EPS!r} is below float32 resolution at 1.0: "
#         f"`1.0 - eps` rounds back to 1.0, leaving log(1 - p) able to evaluate "
#         f"log(0) and emit inf/NaN gradients. Use >= 5.97e-08."
#     )

# # ── Dual-path weight ───────────────────────────────────────────────────────────
# # The o2o (one-to-one) loss block is scaled by this factor before being added
# # to the o2m (one-to-many) loss.  Both paths share the same head outputs; only
# # the assignment targets differ.
# #
# # Normalization behaviour (important):
# #   All three loss components (VFL, CIoU, DFL) normalize by their own
# #   positive count.  Once the model has suppressed background cells (which
# #   happens by the mid-point of training), the VFL numerator is dominated by
# #   foreground terms and the normalization cancels:
# #       loss ≈ Σ_fg(cell_loss) / num_pos  →  avg_fg_cell_loss
# #   Both paths use the same head predictions, so their per-cell foreground
# #   loss is approximately equal late in training, making |o2m| ≈ |o2o| in
# #   absolute terms despite o2m having ~8× more positives.
# #
# #   Setting WEIGHT_O2O_PATH = 1.0 therefore nearly doubles the total gradient
# #   and causes the optimizer to fight itself (dense o2m vs sparse o2o).
# #
# #   WEIGHT_O2O_PATH = 0.25 keeps o2m as the primary supervisor (~80 % of
# #   gradient budget) while o2o provides a light calibration signal (~20 %).
# #   This matches the practical guidance from YOLOv10-style dual-path training.
# #
# # Why o2o helps with cls ranking:
# #   o2m assigns topk=10 anchors per GT, all receiving the same class target.
# #   o2o assigns exactly ONE anchor (the highest TAL-score anchor) with a
# #   concentrated, unambiguous signal.  This single-winner pressure prevents
# #   the cls head from spreading probability mass across all topk anchors and
# #   helps ensure one detection per GT scores clearly above background.
# #   The effect is conservative at weight=0.25 — it nudges score separation
# #   without destabilising the o2m-dominated localization gradient.
# WEIGHT_O2O_PATH = 0.25


# # ─────────────────────────────────────────────────────────────────────────────
# # 1.  Varifocal Loss  (classification)
# # ─────────────────────────────────────────────────────────────────────────────
# def varifocal_loss(
#     pred_scores:   tf.Tensor,
#     target_scores: tf.Tensor,
#     fg_mask:       tf.Tensor,
#     alpha: float = 0.75,
#     gamma: float = 2.0,
#     eps:   float = VFL_PROB_EPS,
#     neg_weight_floor: float = VFL_NEG_WEIGHT_FLOOR,
#     sanitize: bool = True,
# ) -> tf.Tensor:
#     """
#     Varifocal Loss — IoU-aware classification loss.

#     Unlike standard focal loss VFL applies asymmetric focusing:
#       - Negative cells  (target=0) : focal down-weighting
#                                      weight = alpha * pred^gamma
#                                               + neg_weight_floor
#       - Positive cells  (target>0) : NO focal down-weighting
#                                      weight = target  (IoU quality score)

#     The ``neg_weight_floor`` term restores a small constant push-down on
#     background cells whose score has drifted upward but is still well
#     below the prediction the model would make for a confident negative.
#     See ``VFL_NEG_WEIGHT_FLOOR`` for the calibration argument.

#     Normalized by max(num_positives, 1), not by total cell count.

#     Args:
#         pred_scores      : (B, N, C)  sigmoid class probabilities
#         target_scores    : (B, N, C)  IoU-weighted label (iou for pos, 0 for neg)
#         fg_mask          : (B, N)     bool foreground mask — used for normalization
#         alpha, gamma     : focal parameters
#         eps              : numerical guard
#         neg_weight_floor : minimum weight applied to the negative branch
#                            (target == 0).  Defaults to ``VFL_NEG_WEIGHT_FLOOR``.
#                            Set to 0.0 to recover the original VFL behaviour.
#         sanitize         : when True (default) non-finite intermediates are
#                            replaced with neutral values so one bad element
#                            cannot poison the whole batch.  Pass False ONLY
#                            from the first-NaN diagnostic probe, which needs
#                            the UNMASKED value to identify where the
#                            non-finiteness actually originates.
#     """
#     pred_scores   = tf.cast(pred_scores,   tf.float32)
#     target_scores = tf.cast(target_scores, tf.float32)
#     fg_mask       = tf.cast(fg_mask,       tf.float32)   # (B, N)

#     # Sanitize any non-finite head outputs before the clip — clip_by_value
#     # propagates NaN unchanged.
#     if sanitize:
#         pred_scores = tf.where(
#             tf.math.is_finite(pred_scores),
#             pred_scores,
#             tf.fill(tf.shape(pred_scores), tf.constant(0.5, dtype=tf.float32)),
#         )
#     # Upper bound computed in float64 then materialised as float32 so the
#     # `assert` below is checking the value the op will actually use.  Both
#     # bounds must be strictly inside (0, 1): `log(p)` and `log(1 - p)` are
#     # evaluated for EVERY cell, and either endpoint yields -inf with a 1/0
#     # gradient.  See VFL_PROB_EPS for the failure this prevents.
#     _lo = tf.constant(float(eps), dtype=tf.float32)
#     _hi = tf.constant(1.0 - float(eps), dtype=tf.float32)
#     pred_scores = tf.clip_by_value(pred_scores, _lo, _hi)

#     is_positive = tf.cast(target_scores > 0.0, tf.float32)
#     neg_weight = (
#         alpha * (pred_scores ** gamma)
#         + tf.constant(float(neg_weight_floor), dtype=tf.float32)
#     )
#     weight = (
#         is_positive * target_scores
#         + (1.0 - is_positive) * neg_weight
#     )

#     bce = -(
#         target_scores * tf.math.log(pred_scores)
#         + (1.0 - target_scores) * tf.math.log(1.0 - pred_scores)
#     )

#     # Sum over classes → (B, N), then normalize by num_pos (not N)
#     #
#     # WARNING — this sanitiser hides forward evidence but cannot fix a bad
#     # gradient.  `tf.where` routes a ZERO upstream gradient into the masked
#     # branch, and 0 * inf = NaN, so any non-finite gradient inside the branch
#     # is converted into a NaN that reaches the weights while `loss_per_cell`
#     # still reads finite.  It is safe here ONLY because the clamp above makes
#     # `bce` and its gradient finite by construction; never rely on it to
#     # neutralise an unbounded op.
#     loss_per_cell = tf.reduce_sum(weight * bce, axis=-1)  # (B, N)
#     if sanitize:
#         loss_per_cell = tf.where(
#             tf.math.is_finite(loss_per_cell),
#             loss_per_cell,
#             tf.zeros_like(loss_per_cell),
#         )
#     num_pos = tf.reduce_sum(fg_mask) + 1.0
#     return tf.reduce_sum(loss_per_cell) / num_pos


# def varifocal_loss_components(
#     pred_scores:   tf.Tensor,
#     target_scores: tf.Tensor,
#     fg_mask:       tf.Tensor,
#     alpha: float = 0.75,
#     gamma: float = 2.0,
#     eps:   float = VFL_PROB_EPS,
#     neg_weight_floor: float = VFL_NEG_WEIGHT_FLOOR,
#     sanitize: bool = True,
# ) -> tuple[tf.Tensor, tf.Tensor]:
#     """
#     Decomposition of ``varifocal_loss`` into positive-branch and
#     negative-branch contributions.  Diagnostic helper only — training
#     continues to call ``varifocal_loss``.

#     The returned scalars satisfy ``loss_pos + loss_neg == varifocal_loss``
#     up to floating-point rounding because the per-cell loss is split into
#     two disjoint sums by the same ``is_positive`` mask the canonical
#     loss uses internally.  Normalisation by ``max(num_positives, 1)``
#     matches the canonical loss exactly.

#     Used to separate the failure modes that the merged ``val_cls``
#     scalar conflates:
#       • ``loss_pos`` shrinking and ``loss_neg`` flat → positive branch
#         is improving (TP scores rising); calibration likely healthy.
#       • ``loss_pos`` flat and ``loss_neg`` rising → background scores
#         are drifting upward (FP calibration drift band — the failure
#         mode that ``VFL_NEG_WEIGHT_FLOOR`` was added to suppress).
#       • Both rising → classifier is broadly losing separation; likely
#         the regime where ``_maybe_adapt_cls_calibration`` should fire.

#     Returns
#     -------
#     (loss_pos, loss_neg) : tuple of two scalar tf.Tensor.
#         Both are divided by ``max(num_positives, 1)`` so they sum to
#         the same value the canonical loss returns.
#     """
#     pred_scores   = tf.cast(pred_scores,   tf.float32)
#     target_scores = tf.cast(target_scores, tf.float32)
#     fg_mask       = tf.cast(fg_mask,       tf.float32)

#     if sanitize:
#         pred_scores = tf.where(
#             tf.math.is_finite(pred_scores),
#             pred_scores,
#             tf.fill(tf.shape(pred_scores), tf.constant(0.5, dtype=tf.float32)),
#         )
#     # Same float32-representable clamp as `varifocal_loss` — see VFL_PROB_EPS.
#     pred_scores = tf.clip_by_value(
#         pred_scores,
#         tf.constant(float(eps), dtype=tf.float32),
#         tf.constant(1.0 - float(eps), dtype=tf.float32),
#     )

#     is_positive = tf.cast(target_scores > 0.0, tf.float32)
#     neg_weight = (
#         alpha * (pred_scores ** gamma)
#         + tf.constant(float(neg_weight_floor), dtype=tf.float32)
#     )
#     pos_weight_term = is_positive * target_scores
#     neg_weight_term = (1.0 - is_positive) * neg_weight

#     bce = -(
#         target_scores * tf.math.log(pred_scores)
#         + (1.0 - target_scores) * tf.math.log(1.0 - pred_scores)
#     )

#     pos_per_cell = tf.reduce_sum(pos_weight_term * bce, axis=-1)  # (B, N)
#     neg_per_cell = tf.reduce_sum(neg_weight_term * bce, axis=-1)  # (B, N)
#     if sanitize:
#         pos_per_cell = tf.where(
#             tf.math.is_finite(pos_per_cell), pos_per_cell, tf.zeros_like(pos_per_cell)
#         )
#         neg_per_cell = tf.where(
#             tf.math.is_finite(neg_per_cell), neg_per_cell, tf.zeros_like(neg_per_cell)
#         )
#     num_pos = tf.reduce_sum(fg_mask) + 1.0
#     return (
#         tf.reduce_sum(pos_per_cell) / num_pos,
#         tf.reduce_sum(neg_per_cell) / num_pos,
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # 2.  CIoU Loss  (bounding-box regression)
# # ─────────────────────────────────────────────────────────────────────────────
# def ciou_loss(
#     pred_boxes:   tf.Tensor,
#     target_boxes: tf.Tensor,
#     fg_mask:      tf.Tensor,
#     eps:          float = 1e-7,
#     sanitize:     bool  = True,
# ) -> tf.Tensor:
#     """
#     Complete IoU Loss (CIoU), averaged over foreground cells only.

#     Numerical-safety design (EP1 NaN fix)
#     ─────────────────────────────────────
#     Earlier revisions computed CIoU over ALL cells then multiplied by
#     ``fg_mask`` at the final reduction.  This leaks two NaN paths through
#     autograd at initialization:

#       1. ``0 * NaN = NaN``.  Even if the masked cell's contribution should
#          be zero, a single non-finite ``ciou`` element anywhere in the
#          batch poisons the sum.
#       2. The CIoU ``v`` term contains ``tf.atan(p_w / p_h)``.  For
#          degenerate predictions at init (when ``p_h`` is at the ``eps``
#          clamp) the gradient ``1/p_h`` is ~1e7 and combines with the
#          atan chain rule in a way that can produce non-finite gradients
#          for cells that are then masked to zero.

#     Fix: substitute a canonical safe box (``[0,0,1,1]``) into BOTH
#     ``pred_boxes`` and ``target_boxes`` for every non-positive cell
#     *before* the unstable ops run.  For those cells the CIoU value is
#     deterministically 0 with bounded gradient, and ``tf.where``'s
#     per-branch gradient routing keeps the real predictions' gradients
#     confined to genuine positive cells.

#     Args:
#         pred_boxes   : (B, N, 4) predicted boxes  (x1, y1, x2, y2) normalised
#         target_boxes : (B, N, 4) GT boxes         (x1, y1, x2, y2) normalised
#         fg_mask      : (B, N)    float, > 0 for positive (assigned) cells
#                        (carries the per-cell CIoU weight; both the indicator
#                        and the magnitude come from the same tensor)
#         eps          : numerical guard
#         sanitize     : when True (default) non-finite intermediates are
#                        replaced with neutral values so one bad element cannot
#                        poison the whole batch.  Pass False ONLY from the
#                        first-NaN diagnostic probe, which needs the UNMASKED
#                        value to identify where the non-finiteness originates.
#     """
#     pred_boxes   = tf.cast(pred_boxes,   tf.float32)
#     target_boxes = tf.cast(target_boxes, tf.float32)
#     fg_mask      = tf.cast(fg_mask,      tf.float32)

#     # Defensive sanitization: a single non-finite predicted coordinate
#     # anywhere in the batch would otherwise poison the entire scale's
#     # gradient.  Replace with 0 — the cell is masked out via fg_mask
#     # below so the substitution does not bias positive-cell learning.
#     if sanitize:
#         pred_boxes = tf.where(
#             tf.math.is_finite(pred_boxes), pred_boxes, tf.zeros_like(pred_boxes)
#         )

#     # Substitute a safe canonical box for non-positive cells.  With both
#     # pred and target equal to (0,0,1,1), every CIoU term evaluates to
#     # zero with finite, well-conditioned gradients — and tf.where's
#     # gradient routing then zeroes the contribution to real pred_boxes.
#     is_fg   = fg_mask > 0.0
#     is_fg_4 = tf.broadcast_to(is_fg[..., None], tf.shape(pred_boxes))
#     safe_box = tf.stop_gradient(
#         tf.broadcast_to(
#             tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32),
#             tf.shape(pred_boxes),
#         )
#     )
#     pred_boxes   = tf.where(is_fg_4, pred_boxes,   safe_box)
#     target_boxes = tf.where(is_fg_4, target_boxes, safe_box)

#     p_x1, p_y1, p_x2, p_y2 = (pred_boxes[..., i] for i in range(4))
#     g_x1, g_y1, g_x2, g_y2 = (target_boxes[..., i] for i in range(4))

#     inter_x1 = tf.maximum(p_x1, g_x1)
#     inter_y1 = tf.maximum(p_y1, g_y1)
#     inter_x2 = tf.minimum(p_x2, g_x2)
#     inter_y2 = tf.minimum(p_y2, g_y2)
#     inter_w  = tf.maximum(inter_x2 - inter_x1, 0.0)
#     inter_h  = tf.maximum(inter_y2 - inter_y1, 0.0)
#     inter    = inter_w * inter_h

#     # Guard against degenerate boxes producing negative areas
#     p_area = tf.maximum(p_x2 - p_x1, 0.0) * tf.maximum(p_y2 - p_y1, 0.0)
#     g_area = tf.maximum(g_x2 - g_x1, 0.0) * tf.maximum(g_y2 - g_y1, 0.0)
#     union  = p_area + g_area - inter + eps
#     iou    = inter / union

#     enc_x1 = tf.minimum(p_x1, g_x1)
#     enc_y1 = tf.minimum(p_y1, g_y1)
#     enc_x2 = tf.maximum(p_x2, g_x2)
#     enc_y2 = tf.maximum(p_y2, g_y2)
#     c2     = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

#     p_cx = (p_x1 + p_x2) / 2.0;  p_cy = (p_y1 + p_y2) / 2.0
#     g_cx = (g_x1 + g_x2) / 2.0;  g_cy = (g_y1 + g_y2) / 2.0
#     rho2 = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2

#     # Use a larger clamp for the aspect-ratio term: ``eps=1e-7`` produces
#     # 1/p_h ≈ 1e7 in the atan gradient when a predicted side collapses
#     # at init, which is enough to overflow when combined with the rest
#     # of the chain rule in float32.  ``1e-4`` keeps the gradient bounded
#     # at ~1e4 while leaving the value of ``v`` virtually unchanged for
#     # any realistic box.
#     ar_eps = 1e-4
#     p_w = tf.maximum(p_x2 - p_x1, ar_eps);  p_h = tf.maximum(p_y2 - p_y1, ar_eps)
#     g_w = tf.maximum(g_x2 - g_x1, ar_eps);  g_h = tf.maximum(g_y2 - g_y1, ar_eps)

#     pi  = tf.constant(3.14159265358979, dtype=tf.float32)
#     v   = (4.0 / (pi ** 2)) * (tf.atan(g_w / g_h) - tf.atan(p_w / p_h)) ** 2
#     # ``1 - iou`` is guaranteed in [0, 1] but can underflow to numerical
#     # zero when iou ≈ 1.  Clamp the denominator so alpha_v stays finite.
#     alpha_v = tf.stop_gradient(v / tf.maximum(1.0 - iou + v, eps))

#     ciou = 1.0 - iou + rho2 / c2 + alpha_v * v

#     # Defense in depth: any residual non-finite value is now safely zeroed
#     # because the per-cell ciou tensor itself is finite for safe-box cells.
#     if sanitize:
#         ciou = tf.where(tf.math.is_finite(ciou), ciou, tf.zeros_like(ciou))

#     fg_sum = tf.reduce_sum(tf.cast(is_fg, tf.float32)) + eps
#     return tf.reduce_sum(ciou * fg_mask) / fg_sum


# # ─────────────────────────────────────────────────────────────────────────────
# # 3.  DFL Loss  (distribution focal loss)
# # ─────────────────────────────────────────────────────────────────────────────
# def dfl_loss(
#     pred_dist:    tf.Tensor,
#     ltrb_targets: tf.Tensor,
#     fg_mask:      tf.Tensor,
#     reg_max:      int   = 16,
#     eps:          float = 1e-7,
#     sanitize:     bool  = True,
# ) -> tf.Tensor:
#     """
#     Distribution Focal Loss — soft cross-entropy over reg_max bins.

#     Args:
#         pred_dist    : (B, N, 4 * reg_max)  raw DFL logits
#         ltrb_targets : (B, N, 4)            ltrb distances in grid units [0, reg_max-1]
#         fg_mask      : (B, N)               bool, True for positive cells
#         reg_max      : number of bins  (default 16)
#         eps          : numerical guard
#         sanitize     : when True (default) non-finite intermediates are
#                        replaced with neutral values so one bad element cannot
#                        poison the whole batch.  Pass False ONLY from the
#                        first-NaN diagnostic probe, which needs the UNMASKED
#                        value to identify where the non-finiteness originates.
#     """
#     pred_dist    = tf.cast(pred_dist,    tf.float32)
#     ltrb_targets = tf.cast(ltrb_targets, tf.float32)
#     fg_mask      = tf.cast(fg_mask,      tf.float32)

#     # Sanitize the raw DFL logits — a single non-finite logit anywhere in
#     # the batch would propagate through log_softmax → NaN gradients
#     # everywhere on this scale.  Replace with 0 (uniform softmax for that
#     # cell); the cell is masked out via fg_mask below.
#     if sanitize:
#         pred_dist = tf.where(
#             tf.math.is_finite(pred_dist), pred_dist, tf.zeros_like(pred_dist)
#         )

#     B  = tf.shape(pred_dist)[0]
#     N  = tf.shape(pred_dist)[1]

#     pred_dist = tf.reshape(pred_dist, [B, N, 4, reg_max])
#     log_prob  = tf.nn.log_softmax(pred_dist, axis=-1)   # (B, N, 4, reg_max)

#     ltrb_targets = tf.clip_by_value(ltrb_targets, 0.0, float(reg_max - 1) - eps)
#     # Non-finite ltrb targets are theoretically impossible (the assigner
#     # produces float32 in a fixed range) but cheap defense-in-depth.
#     if sanitize:
#         ltrb_targets = tf.where(
#             tf.math.is_finite(ltrb_targets),
#             ltrb_targets,
#             tf.zeros_like(ltrb_targets),
#         )

#     tgt_floor = tf.floor(ltrb_targets)          # (B, N, 4)
#     w_ceil    = ltrb_targets - tgt_floor
#     w_floor   = 1.0 - w_ceil

#     idx_floor = tf.cast(tgt_floor, tf.int32)
#     idx_ceil  = tf.minimum(idx_floor + 1, reg_max - 1)

#     def _gather_bin(log_p, idx):
#         """Batched gather: log_p (B,N,4,R), idx (B,N,4) → (B,N,4)."""
#         B_ = tf.shape(log_p)[0];  N_ = tf.shape(log_p)[1]
#         b_idx = tf.tile(tf.reshape(tf.range(B_), [B_, 1, 1, 1]), [1, N_, 4, 1])
#         n_idx = tf.tile(tf.reshape(tf.range(N_), [1, N_, 1, 1]), [B_, 1, 4, 1])
#         d_idx = tf.tile(tf.reshape(tf.range(4),  [1, 1, 4, 1]), [B_, N_, 1, 1])
#         r_idx = tf.expand_dims(idx, axis=-1)
#         full_idx = tf.concat([b_idx, n_idx, d_idx, r_idx], axis=-1)
#         return tf.gather_nd(log_p, full_idx)    # (B,N,4)

#     lp_floor = _gather_bin(log_prob, idx_floor)
#     lp_ceil  = _gather_bin(log_prob, idx_ceil)

#     per_dir_loss  = -(w_floor * lp_floor + w_ceil * lp_ceil)  # (B, N, 4)
#     per_cell_loss = tf.reduce_sum(per_dir_loss, axis=-1)       # (B, N)

#     # Per-cell masking before reduction prevents 0 * NaN poisoning if
#     # log_softmax somehow produced a non-finite value on a masked cell.
#     is_fg = fg_mask > 0.0
#     per_cell_loss = tf.where(is_fg, per_cell_loss, tf.zeros_like(per_cell_loss))
#     if sanitize:
#         per_cell_loss = tf.where(
#             tf.math.is_finite(per_cell_loss),
#             per_cell_loss,
#             tf.zeros_like(per_cell_loss),
#         )

#     fg_sum = tf.reduce_sum(tf.cast(is_fg, tf.float32)) + eps
#     return tf.reduce_sum(per_cell_loss * fg_mask) / fg_sum


# # ─────────────────────────────────────────────────────────────────────────────
# # 4.  Combined loss
# # ─────────────────────────────────────────────────────────────────────────────
# def yolo_pro_loss(
#     cls_pred:     tf.Tensor,
#     reg_pred:     tf.Tensor,
#     cls_targets:  tf.Tensor,
#     box_targets:  tf.Tensor,
#     ltrb_targets: tf.Tensor,
#     fg_mask:      tf.Tensor,
#     pred_boxes:   tf.Tensor,
#     reg_max:      int   = 16,
#     w_cls:        float = WEIGHT_CLS,
#     w_box:        float = WEIGHT_BOX,
#     w_dfl:        float = WEIGHT_DFL,
#     sanitize:     bool  = True,
# ) -> dict:
#     """
#     Combined YOLO-Pro loss for one scale.  Call once per scale (P3/P4/P5).

#     Args:
#         cls_pred     : (B, N, C)          sigmoid class probabilities from head
#         reg_pred     : (B, N, 4*reg_max)  raw DFL logits from head
#         cls_targets  : (B, N, C)          IoU-weighted varifocal labels
#         box_targets  : (B, N, 4)          GT boxes (x1,y1,x2,y2) normalised
#         ltrb_targets : (B, N, 4)          GT ltrb distances in grid units
#         fg_mask      : (B, N)             bool foreground indicator
#         pred_boxes   : (B, N, 4)          decoded predicted boxes (x1,y1,x2,y2)
#         reg_max      : DFL bins  (default 16)
#         sanitize     : forwarded to the three component losses.  ``True``
#                        (default) is the training path — non-finite
#                        intermediates are neutralised so a single bad element
#                        cannot poison the batch.  ``False`` is the first-NaN
#                        diagnostic path: the returned scalars are then allowed
#                        to go non-finite, which is precisely the signal the
#                        probe is looking for.

#     Returns:
#         dict: 'loss_cls', 'loss_box', 'loss_dfl', 'loss_total'
#     """
#     # Per-anchor weight for box/DFL — IoU quality of each positive anchor.
#     # This prioritizes tightening already-good boxes and drives mAP75.
#     # BOX_WEIGHT_FLOOR prevents collapse to near-zero early in training when
#     # VFL quality is still at its floor: without it, effective CIoU/DFL
#     # gradient collapses to near-zero without the floor.
#     fg_float    = tf.cast(fg_mask, tf.float32)
#     raw_quality = tf.reduce_sum(cls_targets, axis=-1)
#     box_weights = tf.maximum(raw_quality, BOX_WEIGHT_FLOOR * fg_float)
#     l_cls = varifocal_loss(cls_pred, cls_targets, fg_mask, sanitize=sanitize)
#     l_box = ciou_loss(pred_boxes, box_targets, box_weights, sanitize=sanitize)
#     l_dfl = dfl_loss(
#         reg_pred, ltrb_targets, box_weights, reg_max=reg_max, sanitize=sanitize,
#     )

#     total = w_cls * l_cls + w_box * l_box + w_dfl * l_dfl
#     return {
#         "loss_cls":   l_cls,
#         "loss_box":   l_box,
#         "loss_dfl":   l_dfl,
#         "loss_total": total,
#     } 


"""
yolo_pro/losses.py
──────────────────
YOLO-Pro Training Losses.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Loss          Target signal          Branch         Weight              │
    ├──────────────────────────────────────────────────────────────────────────┤
    │  Varifocal     IoU-weighted cls label  cls_out        1.0  ← was 0.5    │
    │  CIoU          Ground-truth box (xyxy)  reg_out (dec)  2.0               │
    │  DFL           Soft bin label for ltrb  reg_out (raw)  1.5               │
    └──────────────────────────────────────────────────────────────────────────┘

Loss weight rationale (localization + classification balance)
─────────────────────────────────────────────────────────────
  OG Edge Impulse YOLO-Pro uses cls=1.0, box=2.0, dfl=1.5.  With these
  weights VFL dominates the gradient budget (~88 % of loss) — intentional:
  BOX_WEIGHT_FLOOR=0.10 independently guards box gradient when VFL quality
  is near zero, removing the need for a higher box weight.  Reverting from
  box=7.5 (YOLOv8 convention) restores cls to ~22 % of the gradient budget
  (vs ~10 % at 7.5) and closes the mAP gap vs OG.
  However, lowering cls to 0.5 then gave the classification head only ~5 % of
  the gradient budget (0.5 / (0.5+7.5+1.5) = 5 %).  On small datasets with
  short training budgets (~1 920 steps) this proved insufficient: the cls head
  converged too slowly, producing tp_score_mean < fp_score_mean and collapsed
  mAP despite good recall and IoU.

  Root cause of the ranking failure:
    • With cls=0.5 the head barely discriminates between classes by the time
      box regression plateaus and val_loss stops improving (epoch ~57).
    • Near-object background anchors (never assigned as positives) retain
      uninformative cls outputs that can exceed the correct-class output of
      true-positive anchors.
    • Per-class NMS does not suppress these cross-class FPs, so they
      outscore TPs in the AP sort → precision collapses.

  Conservative fix — cls=1.0:
    • Doubles the cls gradient share from 5 % to ~10 % (1.0/10.0).
    • Box regression still dominates (7.5/10.0 = 75 %) — localization is
      preserved.
    • The ~10 % cls share is sufficient for the head to reach meaningful
      score separation within the available training budget.
    • Gradient clipping (global_norm=10.0) prevents instability.

Normalization
─────────────
  All losses are normalized by max(num_positives, 1) — the number of
  foreground cells in the batch.  VFL is summed (not meaned over all N
  cells) so its gradient magnitude scales with positive count just like
  the box / DFL terms.  This is the YOLOv8 / RT-DETR convention and
  prevents the overwhelming majority of negatives from crushing the
  positive signal.
"""

import tensorflow as tf

# DEBUG INSTRUMENTATION ONLY — no-op unless YOLO_PRO_NAN_PROBE=1 (see
# yolo_pro/_debug_probe.py).  Never referenced by the training math.
from ._debug_probe import probe as _dbg, probe_scalar as _dbgs


WEIGHT_CLS = 1.0
WEIGHT_BOX = 2.0
WEIGHT_DFL = 1.5

# Minimum per-anchor weight for CIoU and DFL losses.
# When VFL quality (= pred-box IoU) is near 0 early in training, raw
# box_weights collapse to near-zero and stall box convergence.  This floor
# is the sole guard: TAL no longer applies a quality floor to cls_targets,
# so BOX_WEIGHT_FLOOR is what keeps box gradient alive while localization
# bootstraps.
BOX_WEIGHT_FLOOR: float = 0.10

# Minimum per-cell weight for the VFL *negative* branch (target == 0).
#
# The unmodified VFL negative weight is ``alpha * pred^gamma`` — a focal
# down-weighting that collapses quadratically as pred approaches the prior.
# This is the right behaviour for confident background cells (pred near 0
# already → no need to push further) but is too soft for cells whose pred
# has drifted *upward* but is still well below 1.0:
#
#   pred = 0.01  →  weight = 0.75 · 1e-4  ≈ 7.5e-5   (negligible)
#   pred = 0.10  →  weight = 0.75 · 1e-2  ≈ 7.5e-3   (still soft)
#   pred = 0.15  →  weight = 0.75 · 2.25e-2 ≈ 1.7e-2
#
# Observed failure mode (EP20→EP31): fp_score_mean drifted 0.093→0.1496
# while tp_score_mean > fp_score_mean and preds_per_image stayed at 15–17.
# Ranking is intact and prediction volume is controlled, but background
# scores keep climbing because the focal factor leaves almost no
# restoring force in the 0.05–0.15 drift band.
#
# A small additive floor restores a baseline push-down in that band while
# preserving:
#   • the positive branch (target > 0)   — TP gradient is unchanged,
#   • confidence-aware focusing — the original alpha * pred^gamma term
#     remains present, with a small baseline added for negatives,
#   • the cls-deferral behaviour at the prior — the floor is two orders
#     of magnitude below the positive-branch gradient at any reasonable
#     IoU target, so TAL-positive anchors still dominate the loss.
#
# Calibrated against the drift band, not the flood band: the floor only
# adds gradient pressure where the focal factor is too small to act and
# the existing Layer 1 / Layer 2 gates still own the flood-recovery path.
VFL_NEG_WEIGHT_FLOOR: float = 0.02

# ── Dual-path weight ───────────────────────────────────────────────────────────
# The o2o (one-to-one) loss block is scaled by this factor before being added
# to the o2m (one-to-many) loss.  Both paths share the same head outputs; only
# the assignment targets differ.
#
# Normalization behaviour (important):
#   All three loss components (VFL, CIoU, DFL) normalize by their own
#   positive count.  Once the model has suppressed background cells (which
#   happens by the mid-point of training), the VFL numerator is dominated by
#   foreground terms and the normalization cancels:
#       loss ≈ Σ_fg(cell_loss) / num_pos  →  avg_fg_cell_loss
#   Both paths use the same head predictions, so their per-cell foreground
#   loss is approximately equal late in training, making |o2m| ≈ |o2o| in
#   absolute terms despite o2m having ~8× more positives.
#
#   Setting WEIGHT_O2O_PATH = 1.0 therefore nearly doubles the total gradient
#   and causes the optimizer to fight itself (dense o2m vs sparse o2o).
#
#   WEIGHT_O2O_PATH = 0.25 keeps o2m as the primary supervisor (~80 % of
#   gradient budget) while o2o provides a light calibration signal (~20 %).
#   This matches the practical guidance from YOLOv10-style dual-path training.
#
# Why o2o helps with cls ranking:
#   o2m assigns topk=10 anchors per GT, all receiving the same class target.
#   o2o assigns exactly ONE anchor (the highest TAL-score anchor) with a
#   concentrated, unambiguous signal.  This single-winner pressure prevents
#   the cls head from spreading probability mass across all topk anchors and
#   helps ensure one detection per GT scores clearly above background.
#   The effect is conservative at weight=0.25 — it nudges score separation
#   without destabilising the o2m-dominated localization gradient.
WEIGHT_O2O_PATH = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Varifocal Loss  (classification)
# ─────────────────────────────────────────────────────────────────────────────
def varifocal_loss(
    pred_scores:   tf.Tensor,
    target_scores: tf.Tensor,
    fg_mask:       tf.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    eps:   float = 1e-9,
    neg_weight_floor: float = VFL_NEG_WEIGHT_FLOOR,
) -> tf.Tensor:
    """
    Varifocal Loss — IoU-aware classification loss.

    Unlike standard focal loss VFL applies asymmetric focusing:
      - Negative cells  (target=0) : focal down-weighting
                                     weight = alpha * pred^gamma
                                              + neg_weight_floor
      - Positive cells  (target>0) : NO focal down-weighting
                                     weight = target  (IoU quality score)

    The ``neg_weight_floor`` term restores a small constant push-down on
    background cells whose score has drifted upward but is still well
    below the prediction the model would make for a confident negative.
    See ``VFL_NEG_WEIGHT_FLOOR`` for the calibration argument.

    Normalized by max(num_positives, 1), not by total cell count.

    Args:
        pred_scores      : (B, N, C)  sigmoid class probabilities
        target_scores    : (B, N, C)  IoU-weighted label (iou for pos, 0 for neg)
        fg_mask          : (B, N)     bool foreground mask — used for normalization
        alpha, gamma     : focal parameters
        eps              : numerical guard
        neg_weight_floor : minimum weight applied to the negative branch
                           (target == 0).  Defaults to ``VFL_NEG_WEIGHT_FLOOR``.
                           Set to 0.0 to recover the original VFL behaviour.
    """
    pred_scores   = tf.cast(pred_scores,   tf.float32)
    target_scores = tf.cast(target_scores, tf.float32)
    fg_mask       = tf.cast(fg_mask,       tf.float32)   # (B, N)

    # Sanitize any non-finite head outputs before the clip — clip_by_value
    # propagates NaN unchanged.
    pred_scores = tf.where(
        tf.math.is_finite(pred_scores),
        pred_scores,
        tf.fill(tf.shape(pred_scores), tf.constant(0.5, dtype=tf.float32)),
    )
    pred_scores = tf.clip_by_value(pred_scores, eps, 1.0 - eps)

    is_positive = tf.cast(target_scores > 0.0, tf.float32)
    neg_weight = (
        alpha * (pred_scores ** gamma)
        + tf.constant(float(neg_weight_floor), dtype=tf.float32)
    )
    weight = (
        is_positive * target_scores
        + (1.0 - is_positive) * neg_weight
    )

    bce = -(
        target_scores * tf.math.log(pred_scores)
        + (1.0 - target_scores) * tf.math.log(1.0 - pred_scores)
    )

    # Sum over classes → (B, N), then normalize by num_pos (not N)
    loss_per_cell = tf.reduce_sum(weight * bce, axis=-1)  # (B, N)
    # DEBUG PROBE (no-op unless YOLO_PRO_NAN_PROBE=1) — VFL internals BEFORE
    # the finiteness sanitiser, so a poisoned value is visible rather than
    # silently zeroed.
    _dbg("06_VFL", "vfl.pred_scores", pred_scores)
    _dbg("06_VFL", "vfl.target_scores", target_scores)
    _dbg("06_VFL", "vfl.weight", weight)
    _dbg("06_VFL", "vfl.bce", bce)
    _dbg("06_VFL", "vfl.loss_per_cell_raw", loss_per_cell)
    loss_per_cell = tf.where(
        tf.math.is_finite(loss_per_cell),
        loss_per_cell,
        tf.zeros_like(loss_per_cell),
    )
    num_pos = tf.reduce_sum(fg_mask) + 1.0
    _out = tf.reduce_sum(loss_per_cell) / num_pos
    _dbg("06_VFL", "vfl.num_pos", num_pos)
    _dbg("06_VFL", "vfl.loss_sum", tf.reduce_sum(loss_per_cell))
    _dbg("06_VFL", "vfl.OUT", _out)
    return _out


def varifocal_loss_components(
    pred_scores:   tf.Tensor,
    target_scores: tf.Tensor,
    fg_mask:       tf.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    eps:   float = 1e-9,
    neg_weight_floor: float = VFL_NEG_WEIGHT_FLOOR,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Decomposition of ``varifocal_loss`` into positive-branch and
    negative-branch contributions.  Diagnostic helper only — training
    continues to call ``varifocal_loss``.

    The returned scalars satisfy ``loss_pos + loss_neg == varifocal_loss``
    up to floating-point rounding because the per-cell loss is split into
    two disjoint sums by the same ``is_positive`` mask the canonical
    loss uses internally.  Normalisation by ``max(num_positives, 1)``
    matches the canonical loss exactly.

    Used to separate the failure modes that the merged ``val_cls``
    scalar conflates:
      • ``loss_pos`` shrinking and ``loss_neg`` flat → positive branch
        is improving (TP scores rising); calibration likely healthy.
      • ``loss_pos`` flat and ``loss_neg`` rising → background scores
        are drifting upward (FP calibration drift band — the failure
        mode that ``VFL_NEG_WEIGHT_FLOOR`` was added to suppress).
      • Both rising → classifier is broadly losing separation; likely
        the regime where ``_maybe_adapt_cls_calibration`` should fire.

    Returns
    -------
    (loss_pos, loss_neg) : tuple of two scalar tf.Tensor.
        Both are divided by ``max(num_positives, 1)`` so they sum to
        the same value the canonical loss returns.
    """
    pred_scores   = tf.cast(pred_scores,   tf.float32)
    target_scores = tf.cast(target_scores, tf.float32)
    fg_mask       = tf.cast(fg_mask,       tf.float32)

    pred_scores = tf.where(
        tf.math.is_finite(pred_scores),
        pred_scores,
        tf.fill(tf.shape(pred_scores), tf.constant(0.5, dtype=tf.float32)),
    )
    pred_scores = tf.clip_by_value(pred_scores, eps, 1.0 - eps)

    is_positive = tf.cast(target_scores > 0.0, tf.float32)
    neg_weight = (
        alpha * (pred_scores ** gamma)
        + tf.constant(float(neg_weight_floor), dtype=tf.float32)
    )
    pos_weight_term = is_positive * target_scores
    neg_weight_term = (1.0 - is_positive) * neg_weight

    bce = -(
        target_scores * tf.math.log(pred_scores)
        + (1.0 - target_scores) * tf.math.log(1.0 - pred_scores)
    )

    pos_per_cell = tf.reduce_sum(pos_weight_term * bce, axis=-1)  # (B, N)
    neg_per_cell = tf.reduce_sum(neg_weight_term * bce, axis=-1)  # (B, N)
    pos_per_cell = tf.where(
        tf.math.is_finite(pos_per_cell), pos_per_cell, tf.zeros_like(pos_per_cell)
    )
    neg_per_cell = tf.where(
        tf.math.is_finite(neg_per_cell), neg_per_cell, tf.zeros_like(neg_per_cell)
    )
    num_pos = tf.reduce_sum(fg_mask) + 1.0
    return (
        tf.reduce_sum(pos_per_cell) / num_pos,
        tf.reduce_sum(neg_per_cell) / num_pos,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CIoU Loss  (bounding-box regression)
# ─────────────────────────────────────────────────────────────────────────────
def ciou_loss(
    pred_boxes:   tf.Tensor,
    target_boxes: tf.Tensor,
    fg_mask:      tf.Tensor,
    eps:          float = 1e-7,
) -> tf.Tensor:
    """
    Complete IoU Loss (CIoU), averaged over foreground cells only.

    Numerical-safety design (EP1 NaN fix)
    ─────────────────────────────────────
    Earlier revisions computed CIoU over ALL cells then multiplied by
    ``fg_mask`` at the final reduction.  This leaks two NaN paths through
    autograd at initialization:

      1. ``0 * NaN = NaN``.  Even if the masked cell's contribution should
         be zero, a single non-finite ``ciou`` element anywhere in the
         batch poisons the sum.
      2. The CIoU ``v`` term contains ``tf.atan(p_w / p_h)``.  For
         degenerate predictions at init (when ``p_h`` is at the ``eps``
         clamp) the gradient ``1/p_h`` is ~1e7 and combines with the
         atan chain rule in a way that can produce non-finite gradients
         for cells that are then masked to zero.

    Fix: substitute a canonical safe box (``[0,0,1,1]``) into BOTH
    ``pred_boxes`` and ``target_boxes`` for every non-positive cell
    *before* the unstable ops run.  For those cells the CIoU value is
    deterministically 0 with bounded gradient, and ``tf.where``'s
    per-branch gradient routing keeps the real predictions' gradients
    confined to genuine positive cells.

    Args:
        pred_boxes   : (B, N, 4) predicted boxes  (x1, y1, x2, y2) normalised
        target_boxes : (B, N, 4) GT boxes         (x1, y1, x2, y2) normalised
        fg_mask      : (B, N)    float, > 0 for positive (assigned) cells
                       (carries the per-cell CIoU weight; both the indicator
                       and the magnitude come from the same tensor)
        eps          : numerical guard
    """
    pred_boxes   = tf.cast(pred_boxes,   tf.float32)
    target_boxes = tf.cast(target_boxes, tf.float32)
    fg_mask      = tf.cast(fg_mask,      tf.float32)

    # Defensive sanitization: a single non-finite predicted coordinate
    # anywhere in the batch would otherwise poison the entire scale's
    # gradient.  Replace with 0 — the cell is masked out via fg_mask
    # below so the substitution does not bias positive-cell learning.
    pred_boxes = tf.where(
        tf.math.is_finite(pred_boxes), pred_boxes, tf.zeros_like(pred_boxes)
    )

    # Substitute a safe canonical box for non-positive cells.  With both
    # pred and target equal to (0,0,1,1), every CIoU term evaluates to
    # zero with finite, well-conditioned gradients — and tf.where's
    # gradient routing then zeroes the contribution to real pred_boxes.
    is_fg   = fg_mask > 0.0
    is_fg_4 = tf.broadcast_to(is_fg[..., None], tf.shape(pred_boxes))
    safe_box = tf.stop_gradient(
        tf.broadcast_to(
            tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32),
            tf.shape(pred_boxes),
        )
    )
    pred_boxes   = tf.where(is_fg_4, pred_boxes,   safe_box)
    target_boxes = tf.where(is_fg_4, target_boxes, safe_box)

    p_x1, p_y1, p_x2, p_y2 = (pred_boxes[..., i] for i in range(4))
    g_x1, g_y1, g_x2, g_y2 = (target_boxes[..., i] for i in range(4))

    inter_x1 = tf.maximum(p_x1, g_x1)
    inter_y1 = tf.maximum(p_y1, g_y1)
    inter_x2 = tf.minimum(p_x2, g_x2)
    inter_y2 = tf.minimum(p_y2, g_y2)
    inter_w  = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h  = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter    = inter_w * inter_h

    # Guard against degenerate boxes producing negative areas
    p_area = tf.maximum(p_x2 - p_x1, 0.0) * tf.maximum(p_y2 - p_y1, 0.0)
    g_area = tf.maximum(g_x2 - g_x1, 0.0) * tf.maximum(g_y2 - g_y1, 0.0)
    union  = p_area + g_area - inter + eps
    iou    = inter / union

    enc_x1 = tf.minimum(p_x1, g_x1)
    enc_y1 = tf.minimum(p_y1, g_y1)
    enc_x2 = tf.maximum(p_x2, g_x2)
    enc_y2 = tf.maximum(p_y2, g_y2)
    c2     = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    p_cx = (p_x1 + p_x2) / 2.0;  p_cy = (p_y1 + p_y2) / 2.0
    g_cx = (g_x1 + g_x2) / 2.0;  g_cy = (g_y1 + g_y2) / 2.0
    rho2 = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2

    # Use a larger clamp for the aspect-ratio term: ``eps=1e-7`` produces
    # 1/p_h ≈ 1e7 in the atan gradient when a predicted side collapses
    # at init, which is enough to overflow when combined with the rest
    # of the chain rule in float32.  ``1e-4`` keeps the gradient bounded
    # at ~1e4 while leaving the value of ``v`` virtually unchanged for
    # any realistic box.
    ar_eps = 1e-4
    p_w = tf.maximum(p_x2 - p_x1, ar_eps);  p_h = tf.maximum(p_y2 - p_y1, ar_eps)
    g_w = tf.maximum(g_x2 - g_x1, ar_eps);  g_h = tf.maximum(g_y2 - g_y1, ar_eps)

    pi  = tf.constant(3.14159265358979, dtype=tf.float32)
    v   = (4.0 / (pi ** 2)) * (tf.atan(g_w / g_h) - tf.atan(p_w / p_h)) ** 2
    # ``1 - iou`` is guaranteed in [0, 1] but can underflow to numerical
    # zero when iou ≈ 1.  Clamp the denominator so alpha_v stays finite.
    alpha_v = tf.stop_gradient(v / tf.maximum(1.0 - iou + v, eps))

    ciou = 1.0 - iou + rho2 / c2 + alpha_v * v

    # DEBUG PROBE (no-op unless YOLO_PRO_NAN_PROBE=1) — every CIoU intermediate
    # BEFORE the finiteness sanitiser below.
    _dbg("06_CIOU", "ciou.pred_boxes", pred_boxes)
    _dbg("06_CIOU", "ciou.target_boxes", target_boxes)
    _dbg("06_CIOU", "ciou.fg_mask", fg_mask)
    _dbg("06_CIOU", "ciou.inter", inter)
    _dbg("06_CIOU", "ciou.union", union)
    _dbg("06_CIOU", "ciou.iou", iou)
    _dbg("06_CIOU", "ciou.c2", c2)
    _dbg("06_CIOU", "ciou.rho2", rho2)
    _dbg("06_CIOU", "ciou.rho2_over_c2", rho2 / c2)
    _dbg("06_CIOU", "ciou.p_w", p_w)
    _dbg("06_CIOU", "ciou.p_h", p_h)
    _dbg("06_CIOU", "ciou.v", v)
    _dbg("06_CIOU", "ciou.alpha_v", alpha_v)
    _dbg("06_CIOU", "ciou.per_cell_raw", ciou)

    # Defense in depth: any residual non-finite value is now safely zeroed
    # because the per-cell ciou tensor itself is finite for safe-box cells.
    ciou = tf.where(tf.math.is_finite(ciou), ciou, tf.zeros_like(ciou))

    fg_sum = tf.reduce_sum(tf.cast(is_fg, tf.float32)) + eps
    _out = tf.reduce_sum(ciou * fg_mask) / fg_sum
    _dbg("06_CIOU", "ciou.fg_sum", fg_sum)
    _dbg("06_CIOU", "ciou.OUT", _out)
    return _out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DFL Loss  (distribution focal loss)
# ─────────────────────────────────────────────────────────────────────────────
def dfl_loss(
    pred_dist:    tf.Tensor,
    ltrb_targets: tf.Tensor,
    fg_mask:      tf.Tensor,
    reg_max:      int   = 16,
    eps:          float = 1e-7,
) -> tf.Tensor:
    """
    Distribution Focal Loss — soft cross-entropy over reg_max bins.

    Args:
        pred_dist    : (B, N, 4 * reg_max)  raw DFL logits
        ltrb_targets : (B, N, 4)            ltrb distances in grid units [0, reg_max-1]
        fg_mask      : (B, N)               bool, True for positive cells
        reg_max      : number of bins  (default 16)
        eps          : numerical guard
    """
    pred_dist    = tf.cast(pred_dist,    tf.float32)
    ltrb_targets = tf.cast(ltrb_targets, tf.float32)
    fg_mask      = tf.cast(fg_mask,      tf.float32)

    # Sanitize the raw DFL logits — a single non-finite logit anywhere in
    # the batch would propagate through log_softmax → NaN gradients
    # everywhere on this scale.  Replace with 0 (uniform softmax for that
    # cell); the cell is masked out via fg_mask below.
    pred_dist = tf.where(
        tf.math.is_finite(pred_dist), pred_dist, tf.zeros_like(pred_dist)
    )

    B  = tf.shape(pred_dist)[0]
    N  = tf.shape(pred_dist)[1]

    pred_dist = tf.reshape(pred_dist, [B, N, 4, reg_max])
    log_prob  = tf.nn.log_softmax(pred_dist, axis=-1)   # (B, N, 4, reg_max)

    ltrb_targets = tf.clip_by_value(ltrb_targets, 0.0, float(reg_max - 1) - eps)
    # Non-finite ltrb targets are theoretically impossible (the assigner
    # produces float32 in a fixed range) but cheap defense-in-depth.
    ltrb_targets = tf.where(
        tf.math.is_finite(ltrb_targets),
        ltrb_targets,
        tf.zeros_like(ltrb_targets),
    )

    tgt_floor = tf.floor(ltrb_targets)          # (B, N, 4)
    w_ceil    = ltrb_targets - tgt_floor
    w_floor   = 1.0 - w_ceil

    idx_floor = tf.cast(tgt_floor, tf.int32)
    idx_ceil  = tf.minimum(idx_floor + 1, reg_max - 1)

    def _gather_bin(log_p, idx):
        """Batched gather: log_p (B,N,4,R), idx (B,N,4) → (B,N,4)."""
        B_ = tf.shape(log_p)[0];  N_ = tf.shape(log_p)[1]
        b_idx = tf.tile(tf.reshape(tf.range(B_), [B_, 1, 1, 1]), [1, N_, 4, 1])
        n_idx = tf.tile(tf.reshape(tf.range(N_), [1, N_, 1, 1]), [B_, 1, 4, 1])
        d_idx = tf.tile(tf.reshape(tf.range(4),  [1, 1, 4, 1]), [B_, N_, 1, 1])
        r_idx = tf.expand_dims(idx, axis=-1)
        full_idx = tf.concat([b_idx, n_idx, d_idx, r_idx], axis=-1)
        return tf.gather_nd(log_p, full_idx)    # (B,N,4)

    lp_floor = _gather_bin(log_prob, idx_floor)
    lp_ceil  = _gather_bin(log_prob, idx_ceil)

    per_dir_loss  = -(w_floor * lp_floor + w_ceil * lp_ceil)  # (B, N, 4)
    per_cell_loss = tf.reduce_sum(per_dir_loss, axis=-1)       # (B, N)

    # DEBUG PROBE (no-op unless YOLO_PRO_NAN_PROBE=1) — DFL internals BEFORE
    # the mask + finiteness sanitisers below.
    _dbg("06_DFL", "dfl.pred_dist", pred_dist)
    _dbg("06_DFL", "dfl.log_prob", log_prob)
    _dbg("06_DFL", "dfl.ltrb_targets_clipped", ltrb_targets)
    _dbg("06_DFL", "dfl.idx_floor", idx_floor)
    _dbg("06_DFL", "dfl.idx_ceil", idx_ceil)
    _dbg("06_DFL", "dfl.lp_floor", lp_floor)
    _dbg("06_DFL", "dfl.lp_ceil", lp_ceil)
    _dbg("06_DFL", "dfl.per_cell_raw", per_cell_loss)

    # Per-cell masking before reduction prevents 0 * NaN poisoning if
    # log_softmax somehow produced a non-finite value on a masked cell.
    is_fg = fg_mask > 0.0
    per_cell_loss = tf.where(is_fg, per_cell_loss, tf.zeros_like(per_cell_loss))
    per_cell_loss = tf.where(
        tf.math.is_finite(per_cell_loss),
        per_cell_loss,
        tf.zeros_like(per_cell_loss),
    )

    fg_sum = tf.reduce_sum(tf.cast(is_fg, tf.float32)) + eps
    _out = tf.reduce_sum(per_cell_loss * fg_mask) / fg_sum
    _dbg("06_DFL", "dfl.fg_sum", fg_sum)
    _dbg("06_DFL", "dfl.OUT", _out)
    return _out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Combined loss
# ─────────────────────────────────────────────────────────────────────────────
def yolo_pro_loss(
    cls_pred:     tf.Tensor,
    reg_pred:     tf.Tensor,
    cls_targets:  tf.Tensor,
    box_targets:  tf.Tensor,
    ltrb_targets: tf.Tensor,
    fg_mask:      tf.Tensor,
    pred_boxes:   tf.Tensor,
    reg_max:      int   = 16,
    w_cls:        float = WEIGHT_CLS,
    w_box:        float = WEIGHT_BOX,
    w_dfl:        float = WEIGHT_DFL,
) -> dict:
    """
    Combined YOLO-Pro loss for one scale.  Call once per scale (P3/P4/P5).

    Args:
        cls_pred     : (B, N, C)          sigmoid class probabilities from head
        reg_pred     : (B, N, 4*reg_max)  raw DFL logits from head
        cls_targets  : (B, N, C)          IoU-weighted varifocal labels
        box_targets  : (B, N, 4)          GT boxes (x1,y1,x2,y2) normalised
        ltrb_targets : (B, N, 4)          GT ltrb distances in grid units
        fg_mask      : (B, N)             bool foreground indicator
        pred_boxes   : (B, N, 4)          decoded predicted boxes (x1,y1,x2,y2)
        reg_max      : DFL bins  (default 16)

    Returns:
        dict: 'loss_cls', 'loss_box', 'loss_dfl', 'loss_total'
    """
    # Per-anchor weight for box/DFL — IoU quality of each positive anchor.
    # This prioritizes tightening already-good boxes and drives mAP75.
    # BOX_WEIGHT_FLOOR prevents collapse to near-zero early in training when
    # VFL quality is still at its floor: without it, effective CIoU/DFL
    # gradient collapses to near-zero without the floor.
    fg_float    = tf.cast(fg_mask, tf.float32)
    raw_quality = tf.reduce_sum(cls_targets, axis=-1)
    box_weights = tf.maximum(raw_quality, BOX_WEIGHT_FLOOR * fg_float)
    # DEBUG PROBE (no-op unless YOLO_PRO_NAN_PROBE=1) — the loss inputs for
    # this (path, scale); the caller stamps the tag via _debug_probe.set_tag.
    _dbg("05_LOSS_IN", "cls_pred", cls_pred)
    _dbg("05_LOSS_IN", "reg_pred", reg_pred)
    _dbg("05_LOSS_IN", "pred_boxes", pred_boxes)
    _dbg("05_LOSS_IN", "cls_targets", cls_targets)
    _dbg("05_LOSS_IN", "box_targets", box_targets)
    _dbg("05_LOSS_IN", "ltrb_targets", ltrb_targets)
    _dbg("05_LOSS_IN", "fg_mask", fg_mask)
    _dbg("05_LOSS_IN", "box_weights", box_weights)
    _dbgs("05_LOSS_IN", "n_positive_cells", tf.reduce_sum(fg_float))
    l_cls = varifocal_loss(cls_pred, cls_targets, fg_mask)
    l_box = ciou_loss(pred_boxes, box_targets, box_weights)
    l_dfl = dfl_loss(reg_pred, ltrb_targets, box_weights, reg_max=reg_max)

    total = w_cls * l_cls + w_box * l_box + w_dfl * l_dfl
    _dbg("07_LOSS_OUT", "l_cls", l_cls)
    _dbg("07_LOSS_OUT", "l_box", l_box)
    _dbg("07_LOSS_OUT", "l_dfl", l_dfl)
    _dbg("07_LOSS_OUT", "loss_total_scale", total)
    return {
        "loss_cls":   l_cls,
        "loss_box":   l_box,
        "loss_dfl":   l_dfl,
        "loss_total": total,
    }

