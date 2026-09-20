"""
DSP Processor Unit Tests
"""
import io
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.ml.dsp.processor import DSPProcessor, merge_image_params
from PIL import Image


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_png(w: int, h: int, color=(128, 64, 32)) -> bytes:
    """Create a tiny solid-colour PNG and return its raw bytes."""
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDSPProcessorEdgeCases:
    """Test DSP processor with edge-case inputs."""

    def test_short_signal_spectral(self):
        """Shorter signal than FFT length should be zero-padded."""
        signal = np.random.randn(32).astype(np.float32)
        proc   = DSPProcessor("spectral_analysis", {"fft_length": 128}, 100.0)
        feats  = proc.extract(signal)
        assert feats.shape[0] == 65  # 128//2+1

    def test_multichannel_spectral(self):
        """Multi-channel input should be flattened before spectral."""
        signal = np.random.randn(100, 3).astype(np.float32)
        proc   = DSPProcessor("spectral_analysis", {"fft_length": 64}, 100.0)
        feats  = proc.extract(signal)
        assert feats.ndim == 1

    def test_mfcc_various_lengths(self):
        for n in [64, 128, 256, 512, 1024]:
            signal = np.random.randn(n).astype(np.float32)
            proc   = DSPProcessor("mfcc", {
                "num_coefficients": 13, "frame_length": 32,
                "frame_stride": 16, "fft_length": 32,
            }, 8000.0)
            feats = proc.extract(signal)
            assert feats.shape[0] > 0, f"MFCC empty for n={n}"
            assert not np.any(np.isnan(feats)), f"NaN in MFCC for n={n}"

    def test_raw_normalize_range(self):
        signal = np.array([1.0, 5.0, 3.0, -2.0, 8.0], dtype=np.float32)
        proc   = DSPProcessor("raw", {"normalize": True, "flatten": True}, 100.0)
        feats  = proc.extract(signal)
        assert feats.min() >= -0.001
        assert feats.max() <= 1.001

    def test_flatten_feature_count(self):
        """Check feature count for multi-channel flatten."""
        # 4 channels × 7 features
        signal = np.random.randn(50, 4).astype(np.float32)
        feat_list = ["mean", "std", "rms", "max", "min", "skewness", "kurtosis"]
        proc  = DSPProcessor("flatten", {"features": feat_list}, 100.0)
        feats = proc.extract(signal)
        assert feats.shape[0] == 4 * 7

    def test_spectrogram_shape(self):
        signal = np.random.randn(256).astype(np.float32)
        proc   = DSPProcessor("spectrogram", {
            "frame_length": 64, "frame_stride": 32,
            "fft_length": 64, "num_mel_filters": 32,
        }, 8000.0)
        feats = proc.extract(signal)
        assert feats.ndim == 2
        assert feats.shape[1] == 32

    def test_scale_factor(self):
        signal = np.ones(50, dtype=np.float32)
        proc   = DSPProcessor("raw", {"scale_axes": 2.0, "normalize": False, "flatten": True}, 100.0)
        feats  = proc.extract(signal)
        assert np.allclose(feats, 2.0)

    def test_all_blocks_no_nan(self):
        """All DSP blocks should produce finite output on normal input."""
        signal = np.random.randn(200).astype(np.float32) * 9.81
        for btype in ["spectral_analysis", "mfcc", "spectrogram", "raw", "flatten"]:
            proc  = DSPProcessor(btype, {}, 100.0)
            feats = proc.extract(signal)
            assert np.all(np.isfinite(feats)), f"Non-finite output from {btype}"
            assert len(feats) > 0, f"Empty output from {btype}"

    def test_filter_low_pass(self):
        """Low-pass filtered signal should have less high-frequency energy."""
        t      = np.linspace(0, 1, 1000)
        signal = (np.sin(2 * np.pi * 5 * t) +    # 5 Hz — pass
                  np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # 200 Hz — block
        proc_filtered = DSPProcessor("spectral_analysis", {
            "fft_length": 256, "filter_type": "low", "filter_cutoff": 50.0,
        }, 1000.0)
        proc_raw = DSPProcessor("spectral_analysis", {"fft_length": 256}, 1000.0)
        feats_f = proc_filtered.extract(signal)
        feats_r = proc_raw.extract(signal)
        # High-frequency bins should be smaller after filtering
        high_bins = slice(len(feats_f) // 2, None)
        assert feats_f[high_bins].mean() < feats_r[high_bins].mean()


class TestMelFilterbank:
    def test_filterbank_shape(self):
        proc = DSPProcessor("mfcc", {}, 8000.0)
        fb   = proc._mel_filterbank(40, 512, 8000.0, 300.0, 4000.0)
        assert fb.shape == (40, 257)  # n_filters, fft_len//2+1

    def test_filterbank_non_negative(self):
        proc = DSPProcessor("mfcc", {}, 8000.0)
        fb   = proc._mel_filterbank(20, 256, 8000.0, 0.0, 4000.0)
        assert np.all(fb >= 0)

    def test_hz_mel_roundtrip(self):
        proc = DSPProcessor("raw", {}, 100.0)
        for hz in [100, 500, 1000, 4000, 8000]:
            mel = proc._hz_to_mel(float(hz))
            hz2 = proc._mel_to_hz(mel)
            assert abs(hz - hz2) < 0.01, f"Roundtrip failed for {hz} Hz"


# ─── Image DSP (single block) ─────────────────────────────────────────────────

class TestImageDSP:
    """Tests for the image DSP block: shape, normalization, grayscale."""

    def test_rgb_output_shape(self):
        raw = _make_png(100, 80)
        proc = DSPProcessor("image", {"image_width": 32, "image_height": 32,
                                       "grayscale": False, "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (32, 32, 3), f"Expected (32,32,3), got {feats.shape}"

    def test_grayscale_output_shape(self):
        raw = _make_png(100, 80)
        proc = DSPProcessor("image", {"image_width": 48, "image_height": 48,
                                       "grayscale": True, "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (48, 48), f"Expected (48,48), got {feats.shape}"

    def test_normalized_range(self):
        raw = _make_png(64, 64, color=(200, 100, 50))
        proc = DSPProcessor("image", {"image_width": 16, "image_height": 16,
                                       "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.min() >= 0.0 - 1e-5
        assert feats.max() <= 1.0 + 1e-5

    def test_unnormalized_range(self):
        raw = _make_png(64, 64, color=(200, 100, 50))
        proc = DSPProcessor("image", {"image_width": 16, "image_height": 16,
                                       "normalize": False}, 100.0)
        feats = proc.extract(raw)
        assert feats.max() > 1.0, "Un-normalized features should exceed 1.0"

    def test_finite_output(self):
        raw = _make_png(50, 50)
        proc = DSPProcessor("image", {"image_width": 24, "image_height": 24}, 100.0)
        feats = proc.extract(raw)
        assert np.all(np.isfinite(feats)), "Image DSP produced non-finite values"

    def test_jpeg_bytes_work(self):
        """JPEG-format bytes (the most common camera upload format) must open correctly."""
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()
        proc = DSPProcessor("image", {"image_width": 16, "image_height": 16, "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (16, 16, 3)
        assert feats.min() >= 0.0 - 1e-4
        assert feats.max() <= 1.0 + 1e-4


# ─── Raw pixel bytes fallback ────────────────────────────────────────────────

class TestImageRawBytesFallback:
    """
    DSP image block must accept raw uint8 HWC pixel bytes emitted by camera
    firmware that skips JPEG/PNG encoding.  These bytes cannot be opened by
    PIL's Image.open() so _image() falls back to a direct reshape.
    """

    def test_raw_rgb_bytes_shape(self):
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8).tobytes()
        proc = DSPProcessor("image", {"image_width": 16, "image_height": 16,
                                       "grayscale": False, "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (16, 16, 3)

    def test_raw_rgb_bytes_normalized_range(self):
        rng = np.random.default_rng(1)
        raw = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8).tobytes()
        proc = DSPProcessor("image", {"image_width": 8, "image_height": 8,
                                       "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.min() >= 0.0 - 1e-5
        assert feats.max() <= 1.0 + 1e-5

    def test_raw_grayscale_bytes_shape(self):
        rng = np.random.default_rng(2)
        raw = rng.integers(0, 256, size=(24, 24), dtype=np.uint8).tobytes()
        proc = DSPProcessor("image", {"image_width": 24, "image_height": 24,
                                       "grayscale": True, "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (24, 24)

    def test_raw_bytes_pixel_values_preserved(self):
        """Pixel values must survive the raw-bytes path (normalized correctly)."""
        arr = np.full((4, 4, 3), 128, dtype=np.uint8)
        proc = DSPProcessor("image", {"image_width": 4, "image_height": 4,
                                       "normalize": True}, 100.0)
        feats = proc.extract(arr.tobytes())
        expected = 128.0 / 255.0
        assert abs(float(feats[0, 0, 0]) - expected) < 1e-4

    def test_png_bytes_still_work_alongside_raw(self):
        """Existing PNG samples must continue to work after the fallback is added."""
        raw = _make_png(32, 32, color=(200, 100, 50))
        proc = DSPProcessor("image", {"image_width": 16, "image_height": 16,
                                       "normalize": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (16, 16, 3)
        assert feats.min() >= 0.0 - 1e-5
        assert feats.max() <= 1.0 + 1e-5

    def test_non_image_sensor_paths_unaffected(self):
        """Non-image DSP blocks must not be affected by this change."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 256)).astype(np.float32)
        proc = DSPProcessor("spectral_analysis", {"fft_length": 64}, 100.0)
        feats = proc.extract(signal)
        assert np.all(np.isfinite(feats))
        assert feats.dtype == np.float32


# ─── resize_mode ──────────────────────────────────────────────────────────────

class TestResizeMode:
    """Verify each resize_mode produces the correct output dimensions."""

    def _extract(self, mode: str, src_w=200, src_h=100, target=64) -> np.ndarray:
        raw = _make_png(src_w, src_h)
        proc = DSPProcessor("image", {
            "image_width": target, "image_height": target,
            "resize_mode": mode, "grayscale": False,
        }, 100.0)
        return proc.extract(raw)

    def test_squash_output_shape(self):
        feats = self._extract("Squash")
        assert feats.shape == (64, 64, 3)

    def test_fit_shortest_axis_output_shape(self):
        feats = self._extract("Fit shortest axis")
        assert feats.shape == (64, 64, 3)

    def test_fit_longest_axis_output_shape(self):
        feats = self._extract("Fit longest axis")
        assert feats.shape == (64, 64, 3)

    def test_fit_longest_axis_pads_with_zeros(self):
        """Wide image padded to a square should have zero-value padding rows."""
        raw = _make_png(200, 100, color=(255, 255, 255))  # all-white wide image
        proc = DSPProcessor("image", {
            "image_width": 64, "image_height": 64,
            "resize_mode": "Fit longest axis", "normalize": True,
        }, 100.0)
        feats = proc.extract(raw)
        # Top/bottom padding rows should be all zeros
        assert feats.shape == (64, 64, 3)
        top_row = feats[0, :, :]   # should be padded (black)
        assert top_row.sum() < 0.1, "Expected zero-padding at the top"

    def test_unknown_mode_falls_back_to_squash(self):
        """An unrecognised mode should not raise — treated as Squash."""
        raw = _make_png(200, 100)
        proc = DSPProcessor("image", {
            "image_width": 32, "image_height": 32,
            "resize_mode": "nonexistent_mode",
        }, 100.0)
        feats = proc.extract(raw)
        assert feats.shape == (32, 32, 3)


# ─── merge_image_params ───────────────────────────────────────────────────────

class TestMergeImageParams:
    """Unit tests for the merge_image_params() helper."""

    class _Imp:
        """Minimal impulse-like ORM object."""
        def __init__(self, input_type="image", w=128, h=64, r="Squash"):
            self.input_type  = input_type
            self.image_width  = w
            self.image_height = h
            self.resize_mode  = r

    def test_fills_all_gaps(self):
        result = merge_image_params(self._Imp(), {"type": "image", "params": {}})
        assert result["image_width"]  == 128
        assert result["image_height"] == 64
        assert result["resize_mode"]  == "Squash"  # _Imp() sets resize_mode="Squash" explicitly

    def test_block_params_take_precedence(self):
        result = merge_image_params(
            self._Imp(w=128, h=128),
            {"type": "image", "params": {"image_width": 96, "image_height": 96}},
        )
        assert result["image_width"]  == 96
        assert result["image_height"] == 96

    def test_partial_override(self):
        result = merge_image_params(
            self._Imp(w=128, h=64, r="Fit shortest axis"),
            {"type": "image", "params": {"image_width": 48}},
        )
        assert result["image_width"]  == 48       # block wins
        assert result["image_height"] == 64       # root fills in
        assert result["resize_mode"]  == "Fit shortest axis"  # root fills in

    def test_noop_for_non_image_impulse(self):
        imp = self._Imp(input_type="time-series")
        result = merge_image_params(imp, {"type": "spectral_analysis",
                                          "params": {"fft_length": 256}})
        assert "image_width"  not in result
        assert "image_height" not in result
        assert result["fft_length"] == 256

    def test_works_with_dict_impulse(self):
        imp = {"input_type": "image", "image_width": 32, "image_height": 32,
               "resize_mode": "Fit longest axis"}
        result = merge_image_params(imp, {"type": "image", "params": {}})
        assert result["image_width"]  == 32
        assert result["resize_mode"]  == "Fit longest axis"


# ─── multi-block DSP ──────────────────────────────────────────────────────────

class TestMultiBlockDSP:
    """Feature-concatenation semantics for multi-block pipelines."""

    def test_two_blocks_concatenate(self):
        signal = np.random.randn(256).astype(np.float32)
        p1 = DSPProcessor("spectral_analysis", {"fft_length": 64}, 100.0)
        p2 = DSPProcessor("flatten", {}, 100.0)
        f1 = p1.extract(signal).flatten()
        f2 = p2.extract(signal).flatten()
        combined = np.concatenate([f1, f2])
        assert combined.shape[0] == f1.shape[0] + f2.shape[0]

    def test_single_block_equals_multi_of_one(self):
        """Single-block pipeline and a multi-block-of-one must produce identical output."""
        signal = np.random.randn(200).astype(np.float32)
        proc = DSPProcessor("raw", {"normalize": True}, 100.0)
        single = proc.extract(signal).flatten()

        # Simulate the multi-block concatenation path
        all_feats = [proc.extract(signal).flatten()]
        multi = np.concatenate(all_feats) if len(all_feats) > 1 else all_feats[0]

        np.testing.assert_array_almost_equal(single, multi)

    def test_image_block_in_multi_pipeline(self):
        """An image block inside a multi-block pipeline should still produce correct shape."""
        raw = _make_png(60, 60)
        img_proc  = DSPProcessor("image", {"image_width": 32, "image_height": 32,
                                            "grayscale": False}, 100.0)
        img_feats = img_proc.extract(raw).flatten()
        assert img_feats.shape[0] == 32 * 32 * 3


# ─── preview / input-size parity ─────────────────────────────────────────────

class TestPreviewInputSizeParity:
    """The feature count from preview must equal input-size for the same config."""

    def test_feature_count_matches_flatten(self):
        """Feature count reported by a manual run must match the flattened feature size."""
        raw = _make_png(64, 64)
        W, H = 32, 32
        proc  = DSPProcessor("image", {"image_width": W, "image_height": H,
                                        "grayscale": False}, 100.0)
        feats = proc.extract(raw)
        # Same logic as /dsp/input-size: flatten and count
        flat_count = int(feats.flatten().size)
        expected   = W * H * 3
        assert flat_count == expected, (
            f"Feature count {flat_count} does not match expected {expected}"
        )

    def test_grayscale_feature_count(self):
        raw = _make_png(64, 64)
        proc  = DSPProcessor("image", {"image_width": 24, "image_height": 24,
                                        "grayscale": True}, 100.0)
        feats = proc.extract(raw)
        assert feats.flatten().size == 24 * 24
