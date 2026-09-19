# preprocessing.py
# Step 1 — Input Normalization Layer
# MobileNetV2 SSD FPN-Lite 320x320

import tensorflow as tf
from tensorflow import keras


class NormalizationLayer(keras.layers.Layer):
    """
    Casts uint8 RGB input [0, 255] → float32 [-1.0, 1.0].

    Formula: x_norm = (x / 127.5) - 1.0
    This is the standard MobileNetV2 preprocessing contract.

    Args:
        **kwargs: Passed through to the base Layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
            inputs: uint8 tensor of shape (B, 320, 320, 3)

        Returns:
            float32 tensor of shape (B, 320, 320, 3), values in [-1.0, 1.0]
        """
        x = tf.cast(inputs, dtype=tf.float32)  # (B, 320, 320, 3) uint8 → float32
        x = x / 127.5 - 1.0                    # (B, 320, 320, 3) → [-1.0, 1.0]
        return x

    def get_config(self) -> dict:
        return super().get_config()


def build_input_layer(input_shape: tuple = (320, 320, 3)) -> tuple:
    """
    Builds the model input + normalization stage.

    Args:
        input_shape: (H, W, C) — default (320, 320, 3)

    Returns:
        inputs:     keras.Input  — uint8, shape (None, 320, 320, 3)
        normalized: tf.Tensor   — float32, shape (None, 320, 320, 3), range [-1, 1]
    """
    inputs = keras.Input(
        shape=input_shape,
        dtype=tf.uint8,
        name="input_image",
    )
    normalized = NormalizationLayer(name="normalization")(inputs)
    return inputs, normalized


# ---------------------------------------------------------------------------
# Quick verification (run this file directly to test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    inputs, normalized = build_input_layer()
    model = tf.keras.Model(inputs=inputs, outputs=normalized, name="step1_normalization")
    model.summary()

    dummy = np.zeros((2, 320, 320, 3), dtype=np.uint8)
    dummy[..., 0] = 255  # red channel maxed

    out = model(dummy, training=False).numpy()

    assert out.dtype == np.float32,        f"Expected float32, got {out.dtype}"
    assert out.shape == (2, 320, 320, 3),  f"Shape mismatch: {out.shape}"
    assert np.isclose(out[..., 0].max(),  1.0, atol=1e-5), "Max should be ~1.0"
    assert np.isclose(out[..., 1].min(), -1.0, atol=1e-5), "Min should be ~-1.0"

    print("✓ dtype :", out.dtype)
    print("✓ shape :", out.shape)
    print("✓ min   :", out.min(), "  max:", out.max())
