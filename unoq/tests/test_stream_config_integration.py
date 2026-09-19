"""
Integration tests: stream config (sensor/frequency/sample_length_ms) actually
affects inference feature generation and result payloads in UnoQRuntime.

Covered:
  1. _features_from_sensor uses stream frequency when set (overrides env+dsp_config).
  2. _features_from_sensor falls back to env default when stream freq is None.
  3. _features_from_sensor uses dsp_config freq when present and stream freq is None.
  4. inference-result payload includes stream_config metadata when config is set.
  5. inference-result payload has no stream_config when all-None.
  6. sensor and sample_length_ms appear in stream_config metadata.
  7. Source references stream_frequency_hz in _features_from_sensor.
  8. Source references stream_config in _inference_stream_task.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
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
    mod = _import_runtime_module()
    cls = mod.UnoQRuntime
    obj = cls.__new__(cls)
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


# ── 1+2+3. frequency resolution in _features_from_sensor ─────────────────────

class TestFeaturesFromSensorFrequency:
    def _call_features_from_sensor(self, obj, input_shape):
        """Call _features_from_sensor with sine patched, return captured kwargs."""
        import sys
        # Always patch the live module to avoid stale-reference issues from reimports.
        live_mod = sys.modules.get("runtime") or _import_runtime_module()
        captured = {}

        def fake_sine(label, n_samples, freq_hz):
            captured["freq_hz"]   = freq_hz
            captured["n_samples"] = n_samples
            return [0.1] * n_samples

        with patch.object(live_mod, "_sine_accel_values", side_effect=fake_sine):
            try:
                obj._features_from_sensor(input_shape)
            except Exception:
                pass
        return captured

    def test_stream_freq_overrides_env_default(self):
        obj = _make_runtime()
        obj._stream_frequency_hz = 100.0

        captured = self._call_features_from_sensor(obj, [375])
        assert captured.get("freq_hz") == 100.0, (
            f"Expected stream freq 100.0, got {captured.get('freq_hz')}"
        )

    def test_env_default_used_when_stream_freq_none(self):
        obj = _make_runtime()
        obj._stream_frequency_hz = None
        import sys
        live_mod = sys.modules.get("runtime") or _import_runtime_module()
        env_default = live_mod.UNOQ_SAMPLE_FREQ_HZ

        captured = self._call_features_from_sensor(obj, [375])
        assert captured.get("freq_hz") == env_default, (
            f"Expected env default {env_default}, got {captured.get('freq_hz')}"
        )

    def test_stream_freq_beats_dsp_config(self, tmp_path):
        """Stream freq must override dsp_config.json freq."""
        obj = _make_runtime()
        obj._pkg_dir = tmp_path
        obj._stream_frequency_hz = 200.0

        dsp = {"interval_ms": 16, "axes": ["accX", "accY", "accZ"]}  # ~62.5 Hz
        (tmp_path / "dsp_config.json").write_text(json.dumps(dsp))

        captured = self._call_features_from_sensor(obj, [375])
        assert captured.get("freq_hz") == 200.0, (
            "stream_frequency_hz must override dsp_config.json"
        )

    def test_dsp_config_used_when_stream_freq_none(self, tmp_path):
        """dsp_config.json freq used when stream freq is None."""
        obj = _make_runtime()
        obj._pkg_dir = tmp_path
        obj._stream_frequency_hz = None

        dsp = {"interval_ms": 10, "axes": ["accX", "accY", "accZ"]}  # 100 Hz
        (tmp_path / "dsp_config.json").write_text(json.dumps(dsp))

        captured = self._call_features_from_sensor(obj, [300])
        assert captured.get("freq_hz") == pytest.approx(100.0)


# ── 4+5+6. stream_config metadata in inference-result payload ─────────────────

class TestStreamConfigMetadata:
    def _run_one_tick(self, obj, result_dict):
        """Drive _inference_stream_task for one tick, return sent messages."""
        sent = []

        async def _go():
            fake_ws = MagicMock()
            fake_ws.closed = False
            fake_ws.send_json = AsyncMock(side_effect=lambda m: sent.append(m))

            call_count = 0

            def _fake_tick():
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    obj._inference_active = False
                    return None
                return result_dict

            obj._inference_active = True

            with patch.object(obj, "_run_inference_tick", side_effect=_fake_tick), \
                 patch("asyncio.sleep", AsyncMock()):
                task = asyncio.ensure_future(
                    obj._inference_stream_task(fake_ws, "cid-r")
                )
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

        _run(_go())
        return sent

    def test_stream_config_in_payload_when_all_set(self):
        obj = _make_runtime()
        obj._stream_sensor           = "accelerometer"
        obj._stream_frequency_hz     = 62.5
        obj._stream_sample_length_ms = 2000

        msgs = self._run_one_tick(obj, {"label": "walking", "confidence": 0.9})
        inf = [m for m in msgs if m.get("type") == "inference-result"]
        assert inf, "No inference-result sent"
        sc = inf[0]["payload"].get("stream_config")
        assert sc is not None
        assert sc["sensor"]           == "accelerometer"
        assert sc["frequency_hz"]     == 62.5
        assert sc["sample_length_ms"] == 2000

    def test_no_stream_config_when_all_none(self):
        obj = _make_runtime()
        # defaults are all None
        msgs = self._run_one_tick(obj, {"label": "idle", "confidence": 0.8})
        inf = [m for m in msgs if m.get("type") == "inference-result"]
        assert inf, "No inference-result sent"
        assert "stream_config" not in inf[0]["payload"]

    def test_only_sensor_set(self):
        obj = _make_runtime()
        obj._stream_sensor = "camera"

        msgs = self._run_one_tick(obj, {"label": "person", "confidence": 0.75})
        inf = [m for m in msgs if m.get("type") == "inference-result"]
        if inf and "stream_config" in inf[0]["payload"]:
            sc = inf[0]["payload"]["stream_config"]
            assert sc["sensor"] == "camera"
            assert "frequency_hz" not in sc
            assert "sample_length_ms" not in sc

    def test_debug_excluded_with_stream_config(self):
        obj = _make_runtime()
        obj._stream_sensor = "accel"

        msgs = self._run_one_tick(obj, {"label": "run", "confidence": 0.9, "debug": {"x": 1}})
        inf = [m for m in msgs if m.get("type") == "inference-result"]
        if inf:
            assert "debug" not in inf[0]["payload"]

    def test_backward_compat_result_without_config(self):
        """Old result dict without any stream config → no stream_config key."""
        obj = _make_runtime()
        # All None — old behavior
        msgs = self._run_one_tick(obj, {"label": "idle", "confidence": 0.5})
        inf = [m for m in msgs if m.get("type") == "inference-result"]
        if inf:
            payload = inf[0]["payload"]
            assert payload["label"]      == "idle"
            assert payload["confidence"] == 0.5
            assert "stream_config" not in payload


# ── 7+8. Source-level checks ──────────────────────────────────────────────────

class TestSourceLevelChecks:
    def test_features_from_sensor_references_stream_frequency(self):
        import ast
        src  = _runtime_source()
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "_features_from_sensor":
                fn_src = "\n".join(lines[node.lineno - 1:node.end_lineno])
                assert "_stream_frequency_hz" in fn_src, (
                    "_features_from_sensor must reference _stream_frequency_hz"
                )
                return
        pytest.fail("_features_from_sensor not found in runtime.py")

    def test_inference_stream_task_emits_stream_config(self):
        import ast
        src  = _runtime_source()
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "_inference_stream_task":
                fn_src = "\n".join(lines[node.lineno - 1:node.end_lineno])
                assert "stream_config" in fn_src, (
                    "_inference_stream_task must include stream_config in payload"
                )
                return
        pytest.fail("_inference_stream_task not found in runtime.py")
