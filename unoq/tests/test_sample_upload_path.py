"""
Regression tests for P0-A5 (docs/Action/action_fix.md WP-4): the UNO Q
runtime's `_upload_sample` must post to the `path` carried on the
`start-sample` WS command, not to a hardcoded training URL.

`unoq/runtime/protocol.py` is a third, independent implementation of the
same start-sample -> upload contract as `device_client/client.py` and
`device_client/esp32_client.cpp` (it was originally missed when those two
were fixed). Mirrors `device_client/tests/test_sample_upload_path.py`.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest


def _import_runtime_module():
    rt_dir = str(pathlib.Path(__file__).parent.parent)
    if rt_dir not in sys.path:
        sys.path.insert(0, rt_dir)
    if "runtime" in sys.modules:
        del sys.modules["runtime"]
    import runtime as _mod  # noqa: PLC0415
    return _mod


class _FakeResponse:
    def __init__(self, status=201, body=None):
        self.status = status
        self._body = body or {"id": "sample-1"}

    async def text(self):
        return "ok"

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Captures the URL passed to .post() without making a real HTTP call."""

    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse()


def _make_runtime():
    mod = _import_runtime_module()
    cls = mod.UnoQRuntime
    obj = cls.__new__(cls)
    obj.host      = "http://testhost:8010"
    obj.api_key   = "ef_test_key"
    obj.device_id = "dev-1"
    obj._session  = _FakeSession()
    obj._pkg_dir  = None
    return obj, mod


class TestUnoqUploadSampleUsesCommandPath:

    def test_upload_uses_path_from_command_not_hardcoded_training(self):
        obj, _ = _make_runtime()
        asyncio.run(obj._upload_sample(
            body=b"{}", label="l", hmac_key=None, sample_token=None,
            path="/api/v1/ingestion/testing/data",
        ))
        assert len(obj._session.calls) == 1
        assert obj._session.calls[0]["url"] == "http://testhost:8010/api/v1/ingestion/testing/data"

    def test_upload_uses_anomaly_path_from_command(self):
        obj, _ = _make_runtime()
        asyncio.run(obj._upload_sample(
            body=b"{}", label="l", hmac_key=None, sample_token=None,
            path="/api/v1/ingestion/anomaly/data",
        ))
        assert obj._session.calls[0]["url"] == "http://testhost:8010/api/v1/ingestion/anomaly/data"

    def test_upload_falls_back_to_training_path_when_command_omits_path(self, caplog):
        """Older backends that predate the `path` field: fall back, but warn."""
        obj, _ = _make_runtime()
        import logging
        with caplog.at_level(logging.WARNING, logger="unoq"):
            asyncio.run(obj._upload_sample(
                body=b"{}", label="l", hmac_key=None, sample_token=None, path=None,
            ))
        assert obj._session.calls[0]["url"] == "http://testhost:8010/api/v1/ingestion/training/data"
        assert any("path" in rec.message for rec in caplog.records)

    def test_sample_real_extracts_path_from_payload_and_forwards_it(self):
        """The WS start-sample payload's `path` field must reach _upload_sample,
        not just label/length/frequency/hmacKey/sampleToken."""
        obj, _mod = _make_runtime()

        captured = {}

        async def _fake_upload(body, label, hmac_key, sample_token, path=None):
            captured["path"] = path

        obj._upload_sample = _fake_upload

        class _FakeWs:
            async def send_json(self, *a, **kw):
                pass

        payload = {
            "label": "walking",
            "length": 10,
            "frequency": 62.5,
            "path": "/api/v1/ingestion/testing/data",
        }
        asyncio.run(obj._sample_real(_FakeWs(), payload, cid="cid-1"))
        assert captured["path"] == "/api/v1/ingestion/testing/data"

    def test_sample_real_forwards_none_when_payload_has_no_path(self):
        obj, _mod = _make_runtime()

        captured = {"called": False}

        async def _fake_upload(body, label, hmac_key, sample_token, path=None):
            captured["called"] = True
            captured["path"] = path

        obj._upload_sample = _fake_upload

        class _FakeWs:
            async def send_json(self, *a, **kw):
                pass

        payload = {"label": "x", "length": 10, "frequency": 10}
        asyncio.run(obj._sample_real(_FakeWs(), payload, cid="cid-2"))
        assert captured["called"] is True
        assert captured["path"] is None
