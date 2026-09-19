"""The single integration point between stored sample bytes and the DSP loop.

`iter_payloads` carries the `input_type` gate (motion_phase3.md §3.2): a
non-time-series (e.g. image) sample is returned untouched as the one-element
identity list the worker already produces today, while a time-series sample
is decoded (`signal_decode.decode`, Phase 2.2) and segmented into windows
(`windowing.iter_windows`).
"""
from __future__ import annotations

from app.motion.dsp.windowing import iter_windows
from app.motion.services import signal_decode


def iter_payloads(impulse, sample, raw_bytes: bytes) -> list:
    if impulse.input_type != "time-series":
        return [raw_bytes]

    decoded = signal_decode.decode(raw_bytes, sample)
    freq_hz = sample.frequency_hz or impulse.frequency_hz or 100.0
    return list(iter_windows(
        decoded["values"],
        freq_hz,
        impulse.window_size_ms,
        impulse.window_increase_ms,
        impulse.zero_pad_allowed,
    ))
