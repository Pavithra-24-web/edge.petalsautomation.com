"""
DSP Processor — feature extraction for edge ML pipelines.

Supported blocks:
  - spectral_analysis  FFT-based frequency features
  - mfcc               Mel-frequency cepstral coefficients
  - spectrogram        Log-mel spectrogram
  - raw                Normalized raw samples
  - flatten            Statistical time-domain features
  - image              Image resize / normalise
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from typing import Optional


def _get_impulse_field(obj, attr, default):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _uses_fomo_min_resolution(impulse_like) -> bool:
    """
    Return True if any ML block in this impulse uses a FOMO learning block.

    Recognises two block representations:
      • type="fomo_mobilenetv2_0_1"  — explicit FOMO block (v1 or v2)
      • type="object_detection" + params.model in ("fomo",…)  — generic catalog
        entry whose model param selects FOMO (the common case from the UI block
        picker which stores "object_detection" as the type)

    Returns True for both v1 and v2 — callers that need to distinguish versions
    should call _get_fomo_version() separately.
    """
    ml_blocks = _get_impulse_field(impulse_like, "ml_blocks", []) or []
    for block in ml_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type") or block.get("architecture") or ""
        if block_type == "fomo_mobilenetv2_0_1":
            return True
        if block_type == "object_detection":
            model_param = (block.get("params") or {}).get("model", "fomo")
            if str(model_param).lower() in ("fomo", "fomo_mobilenetv2_0_1"):
                return True
    return False


def _get_fomo_version(impulse_like) -> int:
    """Return fomo_version (1 or 2) from the impulse's FOMO ML block params.

    Old impulses without the field default to 1, keeping all v1 code paths
    unchanged at runtime.
    """
    ml_blocks = _get_impulse_field(impulse_like, "ml_blocks", []) or []
    for block in ml_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type") or block.get("architecture") or ""
        is_fomo_block = (
            block_type == "fomo_mobilenetv2_0_1"
            or (
                block_type == "object_detection"
                and str((block.get("params") or {}).get("model", "fomo")).lower()
                in ("fomo", "fomo_mobilenetv2_0_1")
            )
        )
        if is_fomo_block:
            return int((block.get("params") or {}).get("fomo_version", 1))
    return 1


def merge_image_params(impulse_like, block_cfg: dict) -> dict:
    """Merge impulse root-level image settings into a DSP block's params.

    For image-type impulses the processor needs image_width, image_height, and
    resize_mode.  These values live on the Impulse root *and* may (after a save)
    also be present inside block_cfg["params"].  Block-level values take
    precedence; impulse-root values fill any gaps.

    Works with SQLAlchemy ORM objects (attribute access) or plain dicts.

    Args:
        impulse_like: Impulse ORM object or dict with image_width / image_height /
                      resize_mode / input_type fields.
        block_cfg:    DSP block config dict, e.g. {"type": "image", "params": {...}}.

    Returns:
        A new params dict with all three image fields guaranteed for image-type
        impulses; unchanged params dict for non-image impulses.
    """
    input_type = _get_impulse_field(impulse_like, "input_type", "")
    params = dict(block_cfg.get("params", {}))

    if input_type != "image":
        return params

    root_w = _get_impulse_field(impulse_like, "image_width",  96) or 96
    root_h = _get_impulse_field(impulse_like, "image_height", 96) or 96
    root_r = _get_impulse_field(impulse_like, "resize_mode", "Fit shortest axis") or "Fit shortest axis"

    if not params.get("image_width"):
        params["image_width"] = root_w
    if not params.get("image_height"):
        params["image_height"] = root_h
    if not params.get("resize_mode"):
        params["resize_mode"] = root_r

    # v1: enforce 96×96 floor so the stride-16 grid is at least 6×6.
    # v2: no minimum floor — allow any resolution divisible by 8.
    # Guard: v2 logic is only active when fomo_version == 2.
    if _uses_fomo_min_resolution(impulse_like):
        _fver = _get_fomo_version(impulse_like)
        if _fver == 2:
            # v2: snap to multiple of 8 (downsampling factor), no 96×96 minimum.
            _w = int(params.get("image_width") or 128)
            _h = int(params.get("image_height") or 128)
            params["image_width"]  = _w if _w % 8 == 0 else _w + (8 - _w % 8)
            params["image_height"] = _h if _h % 8 == 0 else _h + (8 - _h % 8)
        else:
            # v1 (unchanged): apply 96×96 minimum and snap to mult-of-8.
            params["image_width"] = max(int(params.get("image_width") or 96), 96)
            params["image_height"] = max(int(params.get("image_height") or 96), 96)

    return params


class DSPProcessor:
    def __init__(self, block_type: str, params: dict, frequency_hz: float = 100.0):
        self.block_type = block_type
        self.params = params
        self.frequency_hz = frequency_hz

    def extract(self, data) -> np.ndarray:
        """Route to the appropriate extraction method."""
        dispatch = {
            "spectral_analysis": self._spectral_analysis,
            "mfcc":               self._mfcc,
            "spectrogram":        self._spectrogram,
            "raw":                self._raw,
            "flatten":            self._flatten,
            "image":              self._image,
        }
        fn = dispatch.get(self.block_type)
        if fn is None:
            raise ValueError(f"Unknown DSP block type: {self.block_type}")

        if self.block_type != "image":
            if isinstance(data, bytes):
                import json
                try:
                    parsed = json.loads(data.decode('utf-8'))
                    data = np.array(parsed.get("values", []), dtype=np.float32)
                except Exception:
                    try:
                        import csv
                        from io import StringIO
                        reader = csv.reader(StringIO(data.decode('utf-8')))
                        data = np.array([list(map(float, r)) for r in reader if r], dtype=np.float32)
                    except Exception as e:
                        raise ValueError(f"Could not parse binary data: {e}")
            else:
                data = np.array(data, dtype=np.float32)

            # Flatten multi-channel input to 1D if needed for audio blocks
            if data.ndim > 1 and self.block_type in ("mfcc", "spectrogram", "spectral_analysis"):
                data = data.flatten()

        return fn(data)

    # ─── Spectral Analysis ────────────────────────────────────────────────────

    def _spectral_analysis(self, data: np.ndarray) -> np.ndarray:
        fft_length   = int(self.params.get("fft_length", 256))
        overlap      = float(self.params.get("overlap", 0.5))
        noise_floor  = float(self.params.get("noise_floor_db", -52.0))
        filter_type  = self.params.get("filter_type", "none")
        filter_cutoff= float(self.params.get("filter_cutoff", 100.0))
        scale        = float(self.params.get("scale_axes", 1.0))

        data = data * scale

        # Optional filter
        if filter_type != "none":
            data = self._apply_filter(data, filter_type, filter_cutoff)

        hop = max(1, int(fft_length * (1 - overlap)))
        frames = self._frame(data, fft_length, hop)

        # Hann window + FFT magnitude
        window = np.hanning(fft_length)
        spectra = np.abs(np.fft.rfft(frames * window, n=fft_length))

        # Convert to dB, clip noise floor
        eps = 1e-10
        spectra_db = 20 * np.log10(spectra + eps)
        spectra_db = np.maximum(spectra_db, noise_floor)

        # Average over time frames → single feature vector
        features = np.mean(spectra_db, axis=0)
        return features.astype(np.float32)

    # ─── MFCC ─────────────────────────────────────────────────────────────────

    def _mfcc(self, data: np.ndarray) -> np.ndarray:
        n_coeff    = int(self.params.get("num_coefficients", 13))
        frame_len  = int(self.params.get("frame_length", 256))
        frame_stride = int(self.params.get("frame_stride", 128))
        n_filters  = int(self.params.get("num_filters", 40))
        fft_len    = int(self.params.get("fft_length", 256))
        low_freq   = float(self.params.get("low_frequency", 300.0))
        high_freq  = float(self.params.get("high_frequency", 8000.0))
        noise_floor= float(self.params.get("noise_floor_db", -52.0))

        sr = self.frequency_hz
        high_freq = min(high_freq, sr / 2.0)

        frames = self._frame(data, frame_len, frame_stride)
        window = np.hanning(frame_len)
        powered = (np.abs(np.fft.rfft(frames * window, n=fft_len)) ** 2)

        mel_fb = self._mel_filterbank(n_filters, fft_len, sr, low_freq, high_freq)
        mel_energy = np.dot(powered, mel_fb.T)

        eps = 1e-10
        log_mel = np.log(mel_energy + eps)
        log_mel = np.maximum(log_mel, noise_floor)

        # DCT-II
        mfcc = self._dct(log_mel)[:, :n_coeff]
        return mfcc.flatten().astype(np.float32)

    # ─── Spectrogram ──────────────────────────────────────────────────────────

    def _spectrogram(self, data: np.ndarray) -> np.ndarray:
        frame_len   = int(self.params.get("frame_length", 256))
        frame_stride= int(self.params.get("frame_stride", 128))
        fft_len     = int(self.params.get("fft_length", 256))
        n_mel       = int(self.params.get("num_mel_filters", 32))
        noise_floor = float(self.params.get("noise_floor_db", -52.0))

        sr = self.frequency_hz
        frames = self._frame(data, frame_len, frame_stride)
        window = np.hanning(frame_len)
        powered = (np.abs(np.fft.rfft(frames * window, n=fft_len)) ** 2)

        mel_fb = self._mel_filterbank(n_mel, fft_len, sr, 0, sr / 2)
        mel_spec = np.dot(powered, mel_fb.T)

        log_mel = 10 * np.log10(mel_spec + 1e-10)
        log_mel = np.maximum(log_mel, noise_floor)

        # Return as 2D (time, mel_bins) — flattened for model input
        return log_mel.astype(np.float32)

    # ─── Raw ──────────────────────────────────────────────────────────────────

    def _raw(self, data: np.ndarray) -> np.ndarray:
        scale     = float(self.params.get("scale_axes", 1.0))
        normalize = bool(self.params.get("normalize", True))
        flatten   = bool(self.params.get("flatten", True))

        data = data * scale
        if normalize:
            rng = data.max() - data.min()
            if rng > 1e-8:
                data = (data - data.min()) / rng
        if flatten:
            data = data.flatten()
        return data.astype(np.float32)

    # ─── Image ────────────────────────────────────────────────────────────────

    def _image(self, data) -> np.ndarray:
        import io
        from PIL import Image

        image_width  = int(self.params.get("image_width")  or 96)
        image_height = int(self.params.get("image_height") or 96)
        grayscale    = bool(self.params.get("grayscale", False))
        normalize    = bool(self.params.get("normalize", True))
        resize_mode  = self.params.get("resize_mode", "Fit shortest axis")

        if isinstance(data, bytes):
            # Try PIL first (handles JPEG, PNG, BMP, and all standard formats).
            # Fall back to raw-pixel-byte interpretation when PIL cannot identify
            # the format — this covers camera firmware that sends unencoded RGB/L
            # frames (e.g. raw uint8 HWC arrays) at exactly the target resolution.
            img = None
            try:
                img = Image.open(io.BytesIO(data))
            except Exception:
                pass

            if img is not None:
                img = img.convert('L' if grayscale else 'RGB')

                orig_w, orig_h = img.size

                if resize_mode == "Fit shortest axis":
                    # Scale so the shortest side fills the target; center-crop the excess.
                    scale = max(image_width / orig_w, image_height / orig_h)
                    new_w = max(image_width,  round(orig_w * scale))
                    new_h = max(image_height, round(orig_h * scale))
                    img   = img.resize((new_w, new_h), Image.LANCZOS)
                    left  = (new_w - image_width)  // 2
                    top   = (new_h - image_height) // 2
                    img   = img.crop((left, top, left + image_width, top + image_height))

                elif resize_mode == "Fit longest axis":
                    # Scale so the longest side fits the target; pad the short side with zeros.
                    scale  = min(image_width / orig_w, image_height / orig_h)
                    new_w  = max(1, round(orig_w * scale))
                    new_h  = max(1, round(orig_h * scale))
                    img    = img.resize((new_w, new_h), Image.LANCZOS)
                    canvas = Image.new(img.mode, (image_width, image_height), 0)
                    canvas.paste(img, ((image_width - new_w) // 2, (image_height - new_h) // 2))
                    img    = canvas

                else:  # "Squash" — plain resize, ignores aspect ratio
                    img = img.resize((image_width, image_height), Image.LANCZOS)

                data = np.array(img, dtype=np.float32)
            else:
                # Raw pixel bytes: uint8 HWC layout at the configured resolution.
                channels = 1 if grayscale else 3
                expected = image_height * image_width * channels
                raw_arr = np.frombuffer(data[:expected], dtype=np.uint8)
                if len(raw_arr) < expected:
                    raw_arr = np.pad(raw_arr, (0, expected - len(raw_arr)))
                if grayscale:
                    data = raw_arr.reshape((image_height, image_width)).astype(np.float32)
                else:
                    data = raw_arr.reshape((image_height, image_width, channels)).astype(np.float32)
        else:
            data = np.array(data, dtype=np.float32)
            if grayscale and data.ndim == 3 and data.shape[-1] == 3:
                data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])

        if normalize:
            data = data / 255.0

        return data.astype(np.float32)

    # ─── Flatten (statistical) ────────────────────────────────────────────────

    def _flatten(self, data: np.ndarray) -> np.ndarray:
        scale    = float(self.params.get("scale_axes", 1.0))
        features = self.params.get("features", ["mean", "std", "rms"])
        data = data * scale

        # Handle multi-channel: compute per-channel then concatenate
        if data.ndim == 1:
            channels = [data]
        else:
            channels = [data[:, i] for i in range(data.shape[1])] if data.ndim == 2 else [data.flatten()]

        result = []
        for ch in channels:
            for feat in features:
                if feat == "mean":
                    result.append(float(np.mean(ch)))
                elif feat == "std":
                    result.append(float(np.std(ch)))
                elif feat == "rms":
                    result.append(float(np.sqrt(np.mean(ch ** 2))))
                elif feat == "max":
                    result.append(float(np.max(ch)))
                elif feat == "min":
                    result.append(float(np.min(ch)))
                elif feat == "skewness":
                    result.append(float(skew(ch)))
                elif feat == "kurtosis":
                    result.append(float(kurtosis(ch)))

        return np.array(result, dtype=np.float32)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _frame(data: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
        """Slice 1D signal into overlapping frames → shape (n_frames, frame_len)."""
        if len(data) < frame_len:
            data = np.pad(data, (0, frame_len - len(data)))
        n_frames = 1 + (len(data) - frame_len) // hop
        idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
        return data[idx]

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_filterbank(
        self, n_filters: int, fft_len: int, sr: float, low_hz: float, high_hz: float
    ) -> np.ndarray:
        n_fft = fft_len // 2 + 1
        low_mel  = self._hz_to_mel(low_hz)
        high_mel = self._hz_to_mel(high_hz)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        hz_points  = np.array([self._mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((fft_len + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_filters, n_fft))
        for m in range(1, n_filters + 1):
            f_m_minus = bin_points[m - 1]
            f_m       = bin_points[m]
            f_m_plus  = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fb

    @staticmethod
    def _dct(x: np.ndarray) -> np.ndarray:
        """DCT-II via FFT."""
        N = x.shape[-1]
        v = np.concatenate([x[:, ::2], x[:, 1::2][:, ::-1]], axis=-1)
        V = np.fft.fft(v, axis=-1)
        k = np.arange(N)
        factor = 2 * np.exp(-1j * np.pi * k / (2 * N))
        return np.real(V * factor)

    def _apply_filter(self, data: np.ndarray, filter_type: str, cutoff: float) -> np.ndarray:
        nyq = self.frequency_hz / 2.0
        norm_cutoff = min(cutoff / nyq, 0.99)
        if filter_type == "low":
            b, a = scipy_signal.butter(4, norm_cutoff, btype="low")
        elif filter_type == "high":
            b, a = scipy_signal.butter(4, norm_cutoff, btype="high")
        else:
            return data
        return scipy_signal.filtfilt(b, a, data).astype(np.float32)
