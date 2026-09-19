from typing import Iterator

import numpy as np


def _window_geometry(freq_hz: float, window_size_ms: float, window_increase_ms: float) -> tuple[int, int]:
    window_length = round(window_size_ms * freq_hz / 1000)
    stride = round(window_increase_ms * freq_hz / 1000)
    return window_length, stride


def iter_windows(
    values: np.ndarray,
    freq_hz: float,
    window_size_ms: float,
    window_increase_ms: float,
    zero_pad: bool,
) -> Iterator[np.ndarray]:
    values = np.asarray(values)
    num_samples = values.shape[0]
    window_length, stride = _window_geometry(freq_hz, window_size_ms, window_increase_ms)

    if num_samples < window_length:
        if zero_pad:
            pad_widths = [(0, window_length - num_samples)] + [(0, 0)] * (values.ndim - 1)
            yield np.pad(values, pad_widths)
        return

    num_windows = 1 + (num_samples - window_length) // stride
    for i in range(num_windows):
        start = i * stride
        yield values[start:start + window_length]


def count_windows(
    num_samples: int,
    freq_hz: float,
    window_size_ms: float,
    window_increase_ms: float,
    zero_pad: bool,
) -> int:
    window_length, stride = _window_geometry(freq_hz, window_size_ms, window_increase_ms)

    if num_samples < window_length:
        return 1 if zero_pad else 0

    return 1 + (num_samples - window_length) // stride
