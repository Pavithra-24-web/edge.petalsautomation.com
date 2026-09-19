"""
Ingestion service — Phase 3.

All business logic for device HTTP ingestion lives here so that the route
functions stay thin and this layer can be tested without FastAPI's request
cycle.

Public surface:
    IngestionRequest      dataclass — carries parsed headers + body
    process_ingestion()   main pipeline entry point
    compute_payload_hash()  SHA-256 dedup hook (Phase 4 can query on this)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.ingestion_auth import build_hmac_body, verify_hmac
from app.core.storage import storage
from app.models.devices import ProjectDeviceKey
from app.models.user import Device, Label, Sample
from app.services.devices import upsert_device_from_hello


# ─── Constants ────────────────────────────────────────────────────────────────

_UNLABELED_NAMES: frozenset[str] = frozenset(
    {"unlabeled", "unlabelled", "unknown", ""}
)

# Anomaly data is stored as SampleType.testing; extra_metadata.anomaly=True
# distinguishes it from ordinary testing samples without a DB enum change.
_CATEGORY_TO_SAMPLE_TYPE: dict[str, str] = {
    "training": "training",
    "testing":  "testing",
    "anomaly":  "testing",
}


# ─── Request context ──────────────────────────────────────────────────────────

@dataclass
class IngestionRequest:
    """
    Carries all data resolved by the route layer before calling the service.
    Keeping this as a plain dataclass decouples the service from FastAPI
    Request/Header objects, which makes unit testing straightforward.
    """
    project_id:   str
    key_record:   ProjectDeviceKey
    category:     str            # "training" | "testing" | "anomaly"
    raw_body:     bytes
    parsed:       Any
    content_type: str
    x_signature:  Optional[str] = None
    x_sample_token: Optional[str] = None
    x_label:      Optional[str] = None
    x_no_label:   bool          = False
    x_file_name:  Optional[str] = None


# ─── Payload hash (dedup hook) ────────────────────────────────────────────────

def compute_payload_hash(data: bytes) -> str:
    """
    SHA-256 of stored payload bytes.

    Stored in extra_metadata.payload_hash for every ingested sample.
    Phase 4: query for an existing Sample with the same project_id +
    payload_hash before writing to detect and reject duplicate uploads.
    """
    return hashlib.sha256(data).hexdigest()


# ─── Device identity ──────────────────────────────────────────────────────────

def _resolve_device_id(payload: dict) -> Optional[str]:
    """
    Deterministically extract a stable hardware identifier from the payload.

    Priority:
      1. device_name  (Edge Impulse canonical field — usually a MAC address)
      2. device_id    (alternate field name used by some firmwares)
      3. None         → caller skips device upsert (no stable ID available)

    Explicit fallbacks are intentional: if neither field is present, we
    cannot create a stable device record so we skip upsert entirely rather
    than generating a non-deterministic identifier.
    """
    raw = payload.get("device_name") or payload.get("device_id")
    if not raw:
        return None
    return str(raw).strip() or None


# ─── Label resolution ─────────────────────────────────────────────────────────

def _label_from_filename(filename: str) -> Optional[str]:
    """
    Infer a label from a filename using Edge Impulse naming conventions.

    Patterns handled:
      "<label>.<timestamp>.json"   →  "<label>"    (most common firmware output)
      "<label>.json"               →  "<label>"

    Only the segment before the first dot is considered.  Underscores within
    that segment are preserved (e.g. "acc_wave" stays "acc_wave").
    Placeholder names (unlabeled, unknown, etc.) are rejected.
    """
    bare = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    # Remove extension(s): take everything before the first dot
    stem = bare.split(".")[0] if "." in bare else bare
    candidate = stem.strip()
    if candidate.lower() in _UNLABELED_NAMES:
        return None
    return candidate or None


def _get_or_create_label(name: str, project_id: str, db: Session) -> Optional[str]:
    if not name or name.strip().lower() in _UNLABELED_NAMES:
        return None
    name = name.strip()
    label = (
        db.query(Label)
        .filter(Label.project_id == project_id, Label.name == name)
        .first()
    )
    if not label:
        label = Label(project_id=project_id, name=name)
        db.add(label)
        db.flush()
    return label.id


def _resolve_label(
    req: IngestionRequest,
    payload: dict,
    filename: str,
    project_id: str,
    db: Session,
) -> Optional[str]:
    """
    Resolve label_id with this explicit priority:

      1. x-no-label header      → caller opts out; return None immediately
      2. x-label header         → use as label name (wins over all inference)
      3. payload.label / label_name field
      4. Filename inference      (Edge Impulse <label>.<timestamp>.json)
    """
    if req.x_no_label:
        return None

    if req.x_label and req.x_label.strip():
        return _get_or_create_label(req.x_label, project_id, db)

    payload_label = payload.get("label") or payload.get("label_name")
    if payload_label and str(payload_label).strip():
        return _get_or_create_label(str(payload_label), project_id, db)

    inferred = _label_from_filename(filename)
    if inferred:
        return _get_or_create_label(inferred, project_id, db)

    return None


# ─── Filename builder ─────────────────────────────────────────────────────────

def _build_filename(req: IngestionRequest, payload: dict) -> str:
    """
    Determine the final sample filename.

      1. x-file-name header      (legacy client / data forwarder)
      2. payload.filename        (some SDK versions include this)
      3. Generated               ingestion_{category}_{device_type}_{ts}.json
    """
    if req.x_file_name and req.x_file_name.strip():
        return req.x_file_name.strip()

    payload_fn = payload.get("filename")
    if payload_fn and str(payload_fn).strip():
        return str(payload_fn).strip()

    device_type = payload.get("device_type") or "device"
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"ingestion_{req.category}_{device_type}_{ts}.json"


# ─── Sensor type inference ────────────────────────────────────────────────────

def _infer_sensor_type(sensors: list, device_type: str) -> str:
    names = [str(s.get("name", "")).lower() for s in sensors if isinstance(s, dict)]
    if any("acc" in n for n in names):
        return "accelerometer"
    if any(tok in n for n in names for tok in ("mic", "audio")):
        return "microphone"
    if any(tok in n for n in names for tok in ("cam", "img", "image")):
        return "camera"
    return (device_type or "sensor").lower()


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_ingestion(req: IngestionRequest, db: Session) -> dict:
    """
    Execute the full ingestion pipeline.  Returns a stable response dict.

    Steps:
      1. HMAC verification
      2. Edge Impulse envelope unwrap
      3. Device identity resolution + upsert/restore via Phase 1 service
      4. Filename resolution (x-file-name > payload.filename > generated)
      5. Label resolution   (x-no-label > x-label > payload.label > filename)
      6. Payload hash       (dedup hook for Phase 4)
      7. Storage upload
      8. Sample persistence with device linkage in extra_metadata
    """
    # 1. HMAC ─────────────────────────────────────────────────────────────────
    if req.x_signature:
        sig = req.x_signature
        hmac_body = req.raw_body
    else:
        sig = req.parsed.get("signature") if isinstance(req.parsed, dict) else None
        hmac_body = build_hmac_body(req.raw_body, req.parsed, req.content_type)

    verify_hmac(hmac_body, req.key_record, sig, sample_token=req.x_sample_token)

    # 2. Unwrap ────────────────────────────────────────────────────────────────
    inner: dict = {}
    if isinstance(req.parsed, dict):
        inner = req.parsed.get("payload", req.parsed)

    # 3. Device identity ───────────────────────────────────────────────────────
    device_id_str = _resolve_device_id(inner)
    device_pk: Optional[str] = None

    if device_id_str:
        dev = upsert_device_from_hello(
            db,
            project_id=req.project_id,
            device_id=device_id_str,
            device_type=inner.get("device_type"),
            sensors=inner.get("sensors"),
        )
        device_pk = dev.id

    # 4. Filename ──────────────────────────────────────────────────────────────
    filename = _build_filename(req, inner)

    # 5. Label ─────────────────────────────────────────────────────────────────
    label_id = _resolve_label(req, inner, filename, req.project_id, db)

    # 6. Payload hash — dedup: return existing sample if same project + hash ────
    data_bytes = json.dumps(inner).encode()
    payload_hash = compute_payload_hash(data_bytes)

    existing = (
        db.query(Sample)
        .filter(
            Sample.project_id == req.project_id,
            Sample.payload_hash == payload_hash,
        )
        .first()
    )
    if existing:
        return {
            "success":         True,
            "duplicate":       True,
            "id":              existing.id,
            "filename":        existing.filename,
            "sample_type":     existing.sample_type,
            "label_id":        existing.label_id,
            "sensor_type":     existing.sensor_type,
            "frequency_hz":    existing.frequency_hz,
            "num_channels":    existing.num_channels,
            "num_samples":     existing.num_samples,
            "file_size_bytes": existing.file_size_bytes,
            "device_id":       (existing.extra_metadata or {}).get("device_id"),
            "payload_hash":    payload_hash,
            "created_at":      existing.created_at.isoformat(),
        }

    # 7. Storage ───────────────────────────────────────────────────────────────
    key = storage.sample_key(req.project_id, filename)
    storage.upload_bytes(data_bytes, key, content_type="application/json")

    # 8. Persist ───────────────────────────────────────────────────────────────
    interval_ms = inner.get("interval_ms") or 10
    sensors: list = inner.get("sensors") or []
    values: list  = inner.get("values") or []
    device_type: str = inner.get("device_type") or "device"
    freq_hz = round(1000.0 / interval_ms, 4) if interval_ms else 100.0
    sensor_type = _infer_sensor_type(sensors, device_type)

    extra_meta: dict = {
        "ingestion_category": req.category,
        "payload_hash": payload_hash,
    }
    if req.category == "anomaly":
        extra_meta["anomaly"] = True
    # Device linkage — stored in metadata since Sample has no device FK yet
    if device_pk:
        extra_meta["device_id"] = device_pk
    if device_id_str:
        extra_meta["device_name"] = device_id_str

    sample = Sample(
        project_id=req.project_id,
        label_id=label_id,
        filename=filename,
        storage_key=key,
        sensor_type=sensor_type,
        frequency_hz=freq_hz,
        num_channels=len(sensors),
        num_samples=len(values),
        file_size_bytes=len(data_bytes),
        payload_hash=payload_hash,
        sample_type=_CATEGORY_TO_SAMPLE_TYPE[req.category],
        extra_metadata=extra_meta,
    )
    db.add(sample)
    try:
        db.commit()
    except IntegrityError:
        # Another request inserted the same project+payload_hash concurrently.
        db.rollback()
        existing = (
            db.query(Sample)
            .filter(
                Sample.project_id == req.project_id,
                Sample.payload_hash == payload_hash,
            )
            .first()
        )
        if not existing:
            raise
        return {
            "success":         True,
            "id":              existing.id,
            "filename":        existing.filename,
            "sample_type":     existing.sample_type,
            "label_id":        existing.label_id,
            "sensor_type":     existing.sensor_type,
            "frequency_hz":    existing.frequency_hz,
            "num_channels":    existing.num_channels,
            "num_samples":     existing.num_samples,
            "file_size_bytes": existing.file_size_bytes,
            "device_id":       (existing.extra_metadata or {}).get("device_id"),
            "payload_hash":    payload_hash,
            "created_at":      existing.created_at.isoformat(),
        }
    db.refresh(sample)

    return {
        "success":       True,
        "id":            sample.id,
        "filename":      sample.filename,
        "sample_type":   sample.sample_type,
        "label_id":      sample.label_id,
        "sensor_type":   sample.sensor_type,
        "frequency_hz":  sample.frequency_hz,
        "num_channels":  sample.num_channels,
        "num_samples":   sample.num_samples,
        "file_size_bytes": sample.file_size_bytes,
        "device_id":     device_pk,
        "payload_hash":  payload_hash,
        "created_at":    sample.created_at.isoformat(),
    }
