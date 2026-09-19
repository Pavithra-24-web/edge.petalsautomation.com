"""
Tests for start-inference-stream command parsing in UnoQRuntime.

Covers:
  1. All three new fields (sensor, frequency, sample_length_ms) parsed and stored.
  2. Missing fields default to None (backward compat with old payloads).
  3. Empty/whitespace sensor string → None.
  4. Non-positive or non-numeric frequency/sample_length_ms → None (safe fallback).
  5. No payload key → no crash.
  6. Repeated start clears previous config.
  7. Instance variables declared in __init__.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _runtime_source() -> str:
    return (pathlib.Path(__file__).parent.parent / "runtime.py").read_text(encoding="utf-8")


def _import_runtime_module():
    rt_dir = str(pathlib.Path(__file__).parent.parent)
    if rt_dir not in sys.path:
        sys.path.insert(0, rt_dir)
    if "runtime" in sys.modules:
        del sys.modules["runtime"]
    import runtime as _mod
    return _mod


def _make_runtime() -> object:
    """Construct a UnoQRuntime with minimal fields for dispatch tests."""
    mod = _import_runtime_module()
    cls = mod.UnoQRuntime
    obj = cls.__new__(cls)

    # Seed fields _dispatch relies on.
    obj._inference_active        = False
    obj._inference_task          = None
    obj._snapshot_active         = False
    obj._snapshot_task           = None
    obj._stream_sensor           = None
    obj._stream_frequency_hz     = None
    obj._stream_sample_length_ms = None
    obj._pkg_dir                 = None
    obj._infer_mod               = None
    obj._interpreter             = None
    obj._infer_tick              = 0
    obj._pxe_proc                = None
    obj._pkg_format              = "pe"
    obj._cv2_warned              = False
    return obj


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_command(payload: dict) -> dict:
    return {"type": "start-inference-stream", "correlationId": "cid-r1", "payload": payload}


# ── 1. Instance vars declared in __init__ ────────────────────────────────────

class TestInitVars:
    def test_stream_sensor_in_source(self):
        assert "_stream_sensor" in _runtime_source()

    def test_stream_frequency_hz_in_source(self):
        assert "_stream_frequency_hz" in _runtime_source()

    def test_stream_sample_length_ms_in_source(self):
        assert "_stream_sample_length_ms" in _runtime_source()

    def test_fresh_runtime_has_none_defaults(self):
        obj = _make_runtime()
        assert obj._stream_sensor           is None
        assert obj._stream_frequency_hz     is None
        assert obj._stream_sample_length_ms is None


# ── 2. Full payload parsed and stored ────────────────────────────────────────

class TestFullPayloadParsed:
    def test_all_fields_stored(self):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command({
                    "sensor": "accelerometer",
                    "frequency": 62.5,
                    "sample_length_ms": 2000,
                }))

        _run(_go())

        assert obj._stream_sensor           == "accelerometer"
        assert obj._stream_frequency_hz     == 62.5
        assert obj._stream_sample_length_ms == 2000

    def test_integer_frequency_cast_to_float(self):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command({"frequency": 100}))

        _run(_go())
        assert obj._stream_frequency_hz == 100.0
        assert isinstance(obj._stream_frequency_hz, float)


# ── 3. Missing fields → None (backward compat) ───────────────────────────────

class TestBackwardCompat:
    def test_empty_payload_all_none(self):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command({}))

        _run(_go())

        assert obj._stream_sensor           is None
        assert obj._stream_frequency_hz     is None
        assert obj._stream_sample_length_ms is None

    def test_no_payload_key_no_crash(self):
        """Command missing the 'payload' key entirely must not crash."""
        obj = _make_runtime()
        fake_ws = MagicMock()
        msg = {"type": "start-inference-stream", "correlationId": "c"}

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, msg)

        _run(_go())

        assert obj._stream_sensor           is None
        assert obj._stream_frequency_hz     is None
        assert obj._stream_sample_length_ms is None

    def test_inference_activates_with_no_config(self):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command({}))

        _run(_go())
        assert obj._inference_active is True


# ── 4. Invalid / edge-case values fall back safely ───────────────────────────

class TestInvalidValueFallback:
    def _dispatch_with(self, payload: dict):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command(payload))

        _run(_go())
        return obj

    def test_empty_string_sensor_is_none(self):
        assert self._dispatch_with({"sensor": ""})._stream_sensor is None

    def test_whitespace_sensor_is_none(self):
        assert self._dispatch_with({"sensor": "   "})._stream_sensor is None

    def test_non_string_sensor_is_none(self):
        assert self._dispatch_with({"sensor": 99})._stream_sensor is None

    def test_zero_frequency_is_none(self):
        assert self._dispatch_with({"frequency": 0})._stream_frequency_hz is None

    def test_negative_frequency_is_none(self):
        assert self._dispatch_with({"frequency": -1.0})._stream_frequency_hz is None

    def test_non_numeric_frequency_is_none(self):
        assert self._dispatch_with({"frequency": "fast"})._stream_frequency_hz is None

    def test_zero_sample_length_is_none(self):
        assert self._dispatch_with({"sample_length_ms": 0})._stream_sample_length_ms is None

    def test_negative_sample_length_is_none(self):
        assert self._dispatch_with({"sample_length_ms": -100})._stream_sample_length_ms is None

    def test_non_numeric_sample_length_is_none(self):
        assert self._dispatch_with({"sample_length_ms": "long"})._stream_sample_length_ms is None

    def test_float_sample_length_truncated_to_int(self):
        obj = self._dispatch_with({"sample_length_ms": 2500.7})
        assert obj._stream_sample_length_ms == 2500
        assert isinstance(obj._stream_sample_length_ms, int)


# ── 5. Repeated start clears previous config ─────────────────────────────────

class TestRestartClearsPreviousConfig:
    def test_second_start_without_config_clears_first(self):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command({
                    "sensor": "microphone", "frequency": 16000.0, "sample_length_ms": 1000,
                }))
                await obj._dispatch(fake_ws, _make_command({}))

        _run(_go())

        assert obj._stream_sensor           is None
        assert obj._stream_frequency_hz     is None
        assert obj._stream_sample_length_ms is None

    def test_second_start_with_different_config_overwrites(self):
        obj = _make_runtime()
        fake_ws = MagicMock()

        async def _go():
            with patch("asyncio.create_task") as mock_ct:
                mock_ct.return_value = MagicMock()
                await obj._dispatch(fake_ws, _make_command({
                    "sensor": "accelerometer", "frequency": 62.5, "sample_length_ms": 2000,
                }))
                await obj._dispatch(fake_ws, _make_command({
                    "sensor": "microphone", "frequency": 16000.0, "sample_length_ms": 500,
                }))

        _run(_go())

        assert obj._stream_sensor           == "microphone"
        assert obj._stream_frequency_hz     == 16000.0
        assert obj._stream_sample_length_ms == 500
