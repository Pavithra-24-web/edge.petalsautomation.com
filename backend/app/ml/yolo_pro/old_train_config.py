"""
yolo_pro/train_config.py
────────────────────────
YOLO-Pro Training Configuration.

All hyperparameters, schedules, and augmentation policies in one place.
Nothing here executes training — this module only defines config objects
and factory functions that the training loop (Step 12) will consume.

Active augmentations (training batches only)
────────────────────────────────────────────
  Mosaic          : 2×2 composite from 4 images
                    gated by BOTH epoch < mosaic_epochs  AND  rand < mosaic_prob
                    (default mosaic_prob=1.0 → every sample in the window)
  Horizontal flip : random mirror                (prob = hflip_prob)
  HSV color jitter: hue / saturation / value     (prob = color_jitter_prob)

  Validation, evaluation, export calibration, and inference receive
  unmodified images.  No augmentation occurs outside the training loop.

How it plugs into training
──────────────────────────
  cfg  = TrainConfig(size="nano", num_classes=4)

  # Optimizer + schedule
  lr_fn     = cfg.build_lr_schedule()
  optimizer = cfg.build_optimizer(lr_fn)

  # EMA
  ema       = cfg.build_ema(model)

  # Augmentation flags (checked per sample in the data pipeline)
  use_mosaic        = cfg.use_mosaic(current_epoch) and random() < cfg.mosaic_prob
  use_hflip         = random() < cfg.hflip_prob
  use_color_jitter  = random() < cfg.color_jitter_prob

  # Batch size
  batch_size = cfg.batch_size   # auto-selected based on hardware

Architecture constants (fixed across all training runs)
────────────────────────────────────────────────────────
  reg_max  = 16   (DFL bins)
  strides  = [8, 16, 32]  (P3, P4, P5)
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass, field
from typing import Callable

import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """
    All training hyperparameters for one YOLO-Pro run.

    Instantiate with at minimum `size` and `num_classes`.
    All other fields have spec-default values.

    Args:
        size              : model variant  'nano' | 'tiny' | 'small' | 'medium' | 'large'
        num_classes       : number of object classes (no background)
        input_size        : (H, W) — must both be multiples of 32
        epochs            : total training epochs                    (spec: 300)
        warmup_epochs     : linear LR warmup at start               (spec: 3)
        base_lr           : peak learning rate after warmup          (spec: 0.001)
        min_lr_ratio      : cosine decay floor as fraction of base   (spec: 0.01)
        weight_decay      : AdamW L2 regularisation                  (spec: 0.0005)
        ema_decay         : exponential moving average decay         (spec: 0.9999)
        mosaic_epochs     : epochs during which mosaic is active     (spec: 280)
        mosaic_prob       : probability of mosaic per sample         (spec: 1.0)
        hflip_prob        : probability of horizontal flip           (spec: 0.5)
        color_jitter_prob : probability of HSV color jitter          (spec: 0.5)
        reg_max           : DFL distribution bins                    (spec: 16)
        label_smoothing   : classification label smoothing           (spec: 0.0)
        batch_size        : training batch size; caller sets this explicitly
    """

    # ── Required ──────────────────────────────────────────────────────────────
    size:          str
    num_classes:   int

    # ── Input ─────────────────────────────────────────────────────────────────
    input_size:    tuple[int, int] = (640, 640)

    # ── Schedule ──────────────────────────────────────────────────────────────
    epochs:        int   = 300
    warmup_epochs: int   = 3
    base_lr:       float = 1e-3
    min_lr_ratio:  float = 0.01    # cosine floor = base_lr * min_lr_ratio

    # ── Optimizer ─────────────────────────────────────────────────────────────
    weight_decay:  float = 5e-4

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_decay:     float = 0.9995

    # ── Augmentation (training batches only) ─────────────────────────────────
    # Active: mosaic (first mosaic_epochs epochs), hflip, HSV color jitter.
    # Mixup and copy-paste are NOT implemented — omitted from config entirely
    # so this class reflects reality rather than aspirational spec.
    mosaic_epochs:     int   = 240     # mosaic off final ~20% for clean box regression
    mosaic_prob:       float = 1.0
    hflip_prob:        float = 0.5
    color_jitter_prob: float = 0.5    # probability of HSV jitter per sample

    # ── Architecture (fixed) ─────────────────────────────────────────────────
    reg_max:        int   = 16
    label_smoothing: float = 0.0

    # ── Batch size ────────────────────────────────────────────────────────────
    batch_size:     int = 8    # caller should set this; default 8 keeps spe>1 on tiny datasets

    def __post_init__(self):
        # Validate input resolution
        H, W = self.input_size
        assert H % 32 == 0 and W % 32 == 0, (
            f"input_size {self.input_size} must be multiples of 32"
        )
        assert self.size in ("nano", "tiny", "small", "medium", "large"), (
            f"size must be 'nano', 'tiny', 'small', 'medium', or 'large'; got '{self.size}'"
        )
        assert self.mosaic_epochs <= self.epochs, (
            "mosaic_epochs must be <= total epochs"
        )

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def min_lr(self) -> float:
        """Absolute minimum learning rate (cosine floor)."""
        return self.base_lr * self.min_lr_ratio

    @property
    def steps_per_epoch(self) -> int | None:
        """None until dataset is known; set externally by training loop."""
        return None

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Full input shape (H, W, 3) for model construction."""
        return (*self.input_size, 3)

    # ── Augmentation gate ─────────────────────────────────────────────────────
    def use_mosaic(self, epoch: int) -> bool:
        """True when the current epoch is within the mosaic window."""
        return epoch < self.mosaic_epochs

    # ── Schedule factory ──────────────────────────────────────────────────────
    def build_lr_schedule(
        self,
        steps_per_epoch: int,
    ) -> Callable[[int], float]:
        """
        Build a step-level learning rate schedule:

            Phase 1  [0, warmup_steps)         : linear warmup  0 → base_lr
            Phase 2  [warmup_steps, total_steps]: cosine decay   base_lr → min_lr

        Args:
            steps_per_epoch : number of gradient steps per epoch
                              (= ceil(dataset_size / batch_size))

        Returns:
            A callable  f(step: int) → float  compatible with
            tf.keras.optimizers.schedules or used manually.

        Usage in training loop:
            lr_fn   = cfg.build_lr_schedule(steps_per_epoch=len(train_ds))
            for step, batch in enumerate(train_ds):
                optimizer.learning_rate.assign(lr_fn(step))
        """
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps  = self.epochs * steps_per_epoch
        base_lr      = self.base_lr
        min_lr       = self.min_lr

        def schedule(step: int) -> float:
            step = float(step)
            if step < warmup_steps:
                # Linear warmup: 0 → base_lr
                return base_lr * (step / max(warmup_steps, 1))
            else:
                # Cosine annealing: base_lr → min_lr
                progress = (step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr + (base_lr - min_lr) * cosine

        return schedule

    # ── Optimizer factory ────────────────────────────────────────────────────
    def build_optimizer(
        self,
        lr_schedule: float | Callable | tf.keras.optimizers.schedules.LearningRateSchedule,
    ) -> tf.keras.optimizers.Optimizer:
        """
        Build AdamW optimizer.

        Args:
            lr_schedule : initial LR scalar, callable, or Keras schedule.
                          Pass the output of build_lr_schedule() directly,
                          or a fixed float for debugging.

        Returns:
            tf.keras.optimizers.AdamW instance.

        Usage:
            lr_fn     = cfg.build_lr_schedule(steps_per_epoch)
            optimizer = cfg.build_optimizer(lr_fn(0))   # warm-start at step 0
            # then update manually each step:
            optimizer.learning_rate.assign(lr_fn(global_step))
        """
        return tf.keras.optimizers.AdamW(
            learning_rate = lr_schedule,
            weight_decay  = self.weight_decay,
            beta_1        = 0.9,
            beta_2        = 0.999,
            epsilon       = 1e-7,
            name          = "adamw",
        )

    # ── EMA factory ──────────────────────────────────────────────────────────
    def build_ema(
        self,
        model: tf.keras.Model,
    ) -> "ModelEMA":
        """
        Build an EMA shadow of the model weights.

        Args:
            model : the live training model (keras.Model)

        Returns:
            ModelEMA instance.  Call ema.update(model) each step.

        Usage:
            ema = cfg.build_ema(model)
            for step, batch in enumerate(train_ds):
                # ... train step ...
                ema.update(model)
            ema.apply(model)   # swap to EMA weights for evaluation
        """
        return ModelEMA(model, decay=self.ema_decay)

    # ── Summary ──────────────────────────────────────────────────────────────
    def summary(self) -> None:
        """Print a human-readable config table."""
        print("=" * 56)
        print(f"  YOLO-Pro Training Config  [{self.size.upper()}]")
        print("=" * 56)
        rows = [
            ("Model size",      self.size),
            ("Num classes",     self.num_classes),
            ("Input size",      f"{self.input_size[0]}×{self.input_size[1]}×3"),
            ("Batch size",      self.batch_size),
            ("Epochs",          self.epochs),
            ("Warmup epochs",   self.warmup_epochs),
            ("Base LR",         self.base_lr),
            ("Min LR",          f"{self.min_lr:.2e}  (ratio {self.min_lr_ratio})"),
            ("Weight decay",    self.weight_decay),
            ("EMA decay",       self.ema_decay),
            ("Mosaic",          f"p={self.mosaic_prob}  epochs 0–{self.mosaic_epochs-1}"),
            ("H-flip",          f"p={self.hflip_prob}"),
            ("Color jitter",    f"p={self.color_jitter_prob}  (HSV)"),
            ("Label smoothing", self.label_smoothing),
            ("reg_max",         self.reg_max),
        ]
        for label, value in rows:
            print(f"  {label:<20}  {value}")
        print("=" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# Model EMA
# ─────────────────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of every trainable variable:
        ema_var = decay * ema_var + (1 - decay) * live_var

    The shadow weights are NOT used during the forward/backward pass.
    They are applied to the model temporarily for evaluation, then
    restored so training continues from the live weights.

    Args:
        model : Keras model to shadow
        decay : EMA decay factor  (spec: 0.9999)

    Usage:
        ema = ModelEMA(model, decay=0.9999)

        # inside training step, after optimizer.apply_gradients:
        ema.update(model)

        # before evaluation:
        ema.apply(model)
        val_loss = evaluate(model, val_ds)
        ema.restore(model)   # put live weights back for next train step
    """

    def __init__(self, model: tf.keras.Model, decay: float = 0.9999):
        self.decay    = decay
        self._shadows = [tf.Variable(w, trainable=False, dtype=w.dtype)
                         for w in model.trainable_weights]

    def update(self, model: tf.keras.Model) -> None:
        """Update shadow weights: ema = decay*ema + (1-decay)*live."""
        for shadow, live in zip(self._shadows, model.trainable_weights):
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * live)

    def apply(self, model: tf.keras.Model) -> None:
        """Copy shadow weights into model for evaluation."""
        for shadow, live in zip(self._shadows, model.trainable_weights):
            live.assign(shadow)

    def restore(self, model: tf.keras.Model, backup: list) -> None:
        """Restore pre-apply weights from a backup list."""
        for live, bak in zip(model.trainable_weights, backup):
            live.assign(bak)

    def backup_weights(self, model: tf.keras.Model) -> list:
        """Snapshot current live weights before apply()."""
        return [tf.identity(w) for w in model.trainable_weights]
