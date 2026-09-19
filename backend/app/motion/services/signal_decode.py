"""
Signal decoding — turns stored sample bytes into axis/value arrays.

Extracted from `backend/app/api/v1/endpoints/samples.py` (Phase 2.2's
`GET /samples/{sample_id}/signal`, `get_sample_signal`), where this logic
originally lived inline and private (`_SignalDecodeError`,
`_decode_json_signal`, `_decode_csv_signal`). This module is the public,
reusable home for it — Phase 2's signal endpoint and Phase 3's window
segmentation (`app/motion/dsp/payloads.py`, `iter_payloads`) both call the
same decoder, so the two paths can never disagree on what a stored recording
means.

No behavior changed in the extraction: validation rules, extension/sniffing
behavior, frequency resolution, metadata fallback behavior and error
semantics are all moved verbatim from `samples.py`.

It must decode two stored formats, because the two ways a recording enters
the dataset store different bytes:

- Device recording: the flat Edge-Impulse-style JSON envelope
  (`sensors[]`, `values[][]`, `interval_ms`) written by `process_ingestion`
  (`app/services/ingestion.py`).
- CSV upload: raw CSV bytes, as uploaded via `POST /samples/upload`.

Unrecognised or malformed payloads raise `SignalDecodeError`, never a
partially decoded array.
"""
from __future__ import annotations

import csv
import io
import json

from app.models.user import Sample
from app.services.sample_ingest import _parse_csv_feature_names


class SignalDecodeError(ValueError):
    """Raised when stored sample bytes don't match the format they claim to be."""


def _decode_json_signal(raw: bytes) -> dict:
    """Decode a device-recording (Edge Impulse-style) JSON envelope.

    Mirrors the flat shape `process_ingestion` writes to storage — top-level
    sensors[]/values[][]/interval_ms, not the nested {"payload": {...}} export
    shape some other tools use — since that's the only JSON shape this app's
    own ingestion path actually persists.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignalDecodeError(f"Sample is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SignalDecodeError("JSON sample must be an object")

    sensors = data.get("sensors")
    values = data.get("values")
    if not isinstance(sensors, list) or not sensors:
        raise SignalDecodeError("JSON sample is missing sensors[]")
    if not isinstance(values, list) or not values:
        raise SignalDecodeError("JSON sample is missing values[][]")

    axes: list[str] = []
    for sensor in sensors:
        name = sensor.get("name") if isinstance(sensor, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise SignalDecodeError("JSON sample sensors[] entries must each have a name")
        axes.append(name.strip())

    num_channels = len(axes)
    parsed_values: list[list[float]] = []
    for row in values:
        if not isinstance(row, list) or len(row) != num_channels:
            raise SignalDecodeError("JSON sample values[] rows must match sensors[] length")
        try:
            parsed_values.append([float(v) for v in row])
        except (TypeError, ValueError) as exc:
            raise SignalDecodeError(f"JSON sample values[] contains non-numeric data: {exc}") from exc

    interval_ms = data.get("interval_ms") or 10
    try:
        interval_ms = float(interval_ms)
    except (TypeError, ValueError) as exc:
        raise SignalDecodeError("JSON sample interval_ms must be numeric") from exc
    if interval_ms <= 0:
        raise SignalDecodeError("JSON sample interval_ms must be positive")

    num_samples = len(parsed_values)
    return {
        "axes": axes,
        "values": parsed_values,
        "frequency_hz": 1000.0 / interval_ms,
        "duration_ms": round(num_samples * interval_ms),
        "num_channels": num_channels,
        "num_samples": num_samples,
    }


def _decode_csv_signal(sample: Sample, raw: bytes) -> dict:
    """Decode a raw CSV upload (as accepted by POST /samples/upload).

    Axis names come from the sample's own stored `extra_metadata.feature_names`
    (computed once at upload time by `_parse_csv_feature_names`) when present,
    falling back to re-parsing the header directly for older samples that
    predate that field. Either way the result is validated against the actual
    row data before being trusted.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SignalDecodeError(f"Sample is not valid UTF-8 CSV: {exc}") from exc

    rows = [r for r in csv.reader(io.StringIO(text))]
    if not rows or not any(c.strip() for c in rows[0]):
        raise SignalDecodeError("CSV sample is missing a header row")
    header = rows[0]

    meta = sample.extra_metadata if isinstance(sample.extra_metadata, dict) else {}
    feature_names = meta.get("feature_names")
    if not (isinstance(feature_names, list) and feature_names):
        feature_names = _parse_csv_feature_names(raw)
    if not feature_names:
        raise SignalDecodeError("CSV sample has no parseable feature/axis names")

    axes = [str(name) for name in feature_names]
    num_channels = len(axes)
    # feature_names is always a trailing slice of the header (only a single
    # leading non-feature column, e.g. "timestamp", is ever stripped — see
    # _CSV_NON_FEATURE_COLUMNS), so the offset is just the length difference.
    start_idx = len(header) - num_channels
    if start_idx < 0:
        raise SignalDecodeError("CSV header has fewer columns than its recorded feature names")

    parsed_values: list[list[float]] = []
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        cells = row[start_idx:start_idx + num_channels]
        if len(cells) != num_channels:
            raise SignalDecodeError("CSV row has fewer columns than feature/axis names")
        try:
            parsed_values.append([float(c) for c in cells])
        except ValueError as exc:
            raise SignalDecodeError(f"CSV row contains non-numeric data: {exc}") from exc

    if not parsed_values:
        raise SignalDecodeError("CSV sample has no data rows")

    frequency_hz = sample.frequency_hz or 100.0
    num_samples = len(parsed_values)
    return {
        "axes": axes,
        "values": parsed_values,
        "frequency_hz": frequency_hz,
        "duration_ms": round(num_samples * (1000.0 / frequency_hz)) if frequency_hz else None,
        "num_channels": num_channels,
        "num_samples": num_samples,
    }


def decode(raw_bytes: bytes, sample: Sample) -> dict:
    """Decode a recording's stored bytes into `{axes, values, frequency_hz,
    duration_ms, num_channels, num_samples}`.

    Picks the decoder from the sample's filename extension — matching the
    upload pipeline's own convention — falling back to sniffing the first
    byte of the payload when the extension is missing or unrecognised. Only
    one decoder is ever tried per call, so a malformed payload raises
    `SignalDecodeError` instead of being silently reinterpreted as the other
    format.
    """
    filename = (sample.filename or "").lower()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("json", "csv"):
        # No/unknown extension: sniff the actual bytes rather than guessing —
        # still just one decoder is tried, so a bad payload still raises.
        ext = "json" if raw_bytes.lstrip()[:1] in (b"{", b"[") else "csv"

    if ext == "json":
        return _decode_json_signal(raw_bytes)
    return _decode_csv_signal(sample, raw_bytes)
