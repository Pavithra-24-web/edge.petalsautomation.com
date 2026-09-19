"""
Sample ingest service — the single path that turns raw bytes into a `Sample`
row plus an object in storage.

Extracted from `POST /samples/upload` (`api/v1/endpoints/samples.py`) so any
future producer of sample bytes (e.g. a generation worker) can create samples
that are indistinguishable from uploads to every downstream consumer, without
a second implementation of this logic. `upload_sample()` calls
`create_sample_from_bytes()` too, so there is exactly one function that does
this.

`_resolve_sample_type`, `_extract_metadata` and `_parse_csv_feature_names`
live here rather than in `samples.py` for the same reason — `upload_batch()`
imports them back from here so both write paths share one implementation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.storage import storage
from app.models.user import Label, Sample


def _resolve_sample_type(requested: str, filename: str = "") -> str:
    """
    Resolve the final sample_type for 'automatic' split mode.

    Matches Edge Impulse's behaviour exactly:
      - Compute the CRC-32 hash of the bare filename (case-insensitive).
      - hash_value % 100 < 80  → training  (≈80 % of files)
      - hash_value % 100 >= 80 → testing   (≈20 % of files)

    The same filename always maps to the same category, so re-uploading
    a file never changes its split assignment.
    """
    if requested == "automatic":
        # Use the bare filename only (strip folder prefix) for consistency.
        bare = filename.split("/")[-1].split("\\")[-1].lower()
        hash_val = int(hashlib.md5(bare.encode("utf-8")).hexdigest(), 16)
        return "training" if (hash_val % 100) < 80 else "testing"
    return requested


def _get_or_create_label(name: str, project_id: str, db: Session) -> Optional[str]:
    """Get an existing label by name or create a new one. Returns label_id."""
    if not name or not name.strip():
        return None
    name = name.strip()
    # Reject serialized objects (e.g. str(dict) → "{'type': 'label', ...}") that
    # slip through when annotation files store label as a JSON object instead of a string.
    if name.startswith(("{", "[")):
        return None
    label = db.query(Label).filter(Label.project_id == project_id, Label.name == name).first()
    if not label:
        label = Label(project_id=project_id, name=name)
        db.add(label)
        db.flush()  # get the id without committing
    return label.id


def _extract_metadata(filename: str, content: bytes) -> tuple[str, float, Optional[int]]:
    """
    Post-processing pipeline: Extracts feature metadata purely from the uploaded payload.
    Returns: (sensor_type, frequency_hz, duration_ms)
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    # 1. Image
    if ext in ["jpg", "jpeg", "png", "bmp"]:
        return "camera", 0.0, None

    # 2. Audio (Extract sample rate and duration from WAV header)
    if ext == "wav":
        try:
            # Quick WAV header parse
            if len(content) > 44:
                sample_rate = struct.unpack("<I", content[24:28])[0]
                byte_rate = struct.unpack("<I", content[28:32])[0]
                data_size = struct.unpack("<I", content[40:44])[0]
                duration_ms = int((data_size / byte_rate) * 1000) if byte_rate > 0 else 1000
                return "microphone", float(sample_rate), duration_ms
        except Exception:
            pass
        return "microphone", 16000.0, 1000

    # 3. Edge Impulse JSON format
    if ext == "json":
        try:
            data = json.loads(content.decode("utf-8"))
            payload = data.get("payload", {})
            sensors = payload.get("sensors", [])
            s_name = sensors[0].get("name", "sensor") if sensors else "sensor"

            # Use 'accelerometer' as default shorthand if accX/Y/Z are found
            if any("acc" in s.get("name", "").lower() for s in sensors):
                s_name = "accelerometer"

            freq_hz = float(data.get("signature", {}).get("frequency", 100.0))
            return s_name, freq_hz, None
        except Exception:
            pass

    # Defaults for CSV, Parquet, CBOR, or unknown
    # Usually treated as generic time-series data
    return "accelerometer", 100.0, None


# Leading columns that identify a row rather than a sensed axis — dropped from
# the parsed feature list so a CSV like "timestamp,accX,accY,accZ" reports 3
# feature axes, not 4.
_CSV_NON_FEATURE_COLUMNS = {"timestamp", "time", "time_ms", "t"}


def _parse_csv_feature_names(content: bytes) -> Optional[list[str]]:
    """Read a motion CSV's header row and return its feature/axis names.

    Returns None when the content isn't parseable as CSV or has no header —
    callers treat that as "no feature metadata available" rather than an error.
    """
    try:
        text = content.decode("utf-8", errors="ignore")
        header = next(csv.reader(io.StringIO(text)), None)
        if not header:
            return None
        names = [c.strip() for c in header if c and c.strip()]
        if not names:
            return None
        if names[0].lower() in _CSV_NON_FEATURE_COLUMNS:
            names = names[1:]
        return names or None
    except Exception:
        return None


def create_sample_from_bytes(
    db: Session,
    *,
    project_id: str,
    filename: str,
    content: bytes,
    content_type: Optional[str],
    sample_type: str = "training",
    label_id: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
    uploaded_by: Optional[str] = None,
    commit: bool = True,
) -> Sample:
    """Write `content` into storage and create its `Sample` row.

    The single path from bytes to a `Sample`: every existing consumer of
    `samples` (Dataset page, Annotation Editor, Labeling Queue, AI Labeling,
    DSP, training, export, cloning) works on the resulting row unmodified,
    regardless of what produced the bytes.
    """
    key = storage.sample_key(project_id, filename)
    storage.upload_bytes(content, key, content_type=content_type or "application/octet-stream")

    resolved_type = _resolve_sample_type(sample_type, filename or "")

    sensor_type, freq, duration = _extract_metadata(filename or "", content)

    merged_metadata: dict[str, Any] = dict(extra_metadata or {})
    if (filename or "").lower().endswith(".csv"):
        feature_names = _parse_csv_feature_names(content)
        if feature_names:
            merged_metadata["feature_names"] = feature_names

    sample = Sample(
        project_id=project_id,
        label_id=label_id,
        filename=filename,
        storage_key=key,
        sensor_type=sensor_type,
        frequency_hz=freq,
        duration_ms=duration,
        sample_type=resolved_type,
        file_size_bytes=len(content),
        extra_metadata=merged_metadata,
        uploaded_by=uploaded_by,
    )
    db.add(sample)
    if commit:
        db.commit()
        db.refresh(sample)
    else:
        db.flush()
    return sample
