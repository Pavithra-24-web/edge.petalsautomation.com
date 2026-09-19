"""Dataset-wide window-count preview for the DSP preview endpoints
(motion_phase3.md §3.5, Hooks H11/H12).

Computes window counts purely from ``Sample.num_samples`` /
``Sample.frequency_hz`` metadata via ``count_windows`` -- no object-storage
read, no signal decode.
"""
from __future__ import annotations

from app.motion.dsp.windowing import count_windows


def sample_window_count(sample, impulse):
    """Window count for one sample from metadata alone, or ``"unknown"``.

    Fallback chain (motion_phase3.md §3.5): ``num_samples`` when available,
    else ``duration_ms * frequency_hz``, else ``"unknown"`` -- never a guess.
    """
    freq_hz = sample.frequency_hz or getattr(impulse, "frequency_hz", None)
    if not freq_hz:
        return "unknown"

    num_samples = sample.num_samples
    if not num_samples:
        duration_ms = getattr(sample, "duration_ms", None)
        if not duration_ms:
            return "unknown"
        num_samples = int(round(duration_ms * freq_hz / 1000))

    return count_windows(
        num_samples,
        freq_hz,
        impulse.window_size_ms,
        impulse.window_increase_ms,
        impulse.zero_pad_allowed,
    )


def dataset_window_summary(samples, impulse):
    """Total window count and short-recording skip count across ``samples``.

    A sample with insufficient metadata ("unknown") contributes to neither
    total -- there is no basis to call it either counted or skipped.
    """
    window_count = 0
    skipped_too_short = 0
    for sample in samples:
        count = sample_window_count(sample, impulse)
        if count == "unknown":
            continue
        window_count += count
        if count == 0:
            skipped_too_short += 1
    return window_count, skipped_too_short
