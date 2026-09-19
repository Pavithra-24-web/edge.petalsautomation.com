"""
Device ingestion endpoints — Phase 3 (thin route layer).

Routes (mounted under /api/v1/ingestion/):
    POST /training/data   →  SampleType.training
    POST /testing/data    →  SampleType.testing
    POST /anomaly/data    →  SampleType.testing + extra_metadata.anomaly=True

All business logic lives in app.services.ingestion.process_ingestion.

Expected request body (Edge Impulse envelope format):
    {
      "protected": {"ver": "v1", "alg": "HS256", "iat": <unix_ts>},
      "signature": "<64-char hex HMAC>",
      "payload": {
        "device_name":  "<hardware_id>",         ← stable hardware identifier
        "device_type":  "<board_type>",
        "interval_ms":  10,
        "sensors":      [{"name": "accX", "units": "m/s2"}, ...],
        "values":       [[0.1, 0.2, 0.3], ...],
        "label":        "<label_name>"           ← optional
      }
    }

Request headers:
    x-api-key    (required) device project key
    x-signature  (optional) HMAC-SHA256 of raw body when carried in header
    x-label      (optional) override label for this sample
    x-no-label   (optional) "true" / "1" → store sample without any label
    x-file-name  (optional) filename override for legacy data forwarder clients

Split mapping:
    training route → SampleType.training
    testing  route → SampleType.testing
    anomaly  route → SampleType.testing  (+ extra_metadata.anomaly=True)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.ingestion_auth import resolve_api_key
from app.core.ingestion_body import read_ingestion_body
from app.models.devices import ProjectDeviceKey
from app.services.ingestion import IngestionRequest, process_ingestion

router = APIRouter()


async def _build_ingestion_request(
    request: Request,
    category: str,
    x_signature: Optional[str],
    x_sample_token: Optional[str],
    x_label: Optional[str],
    x_no_label_raw: Optional[str],
    x_file_name: Optional[str],
    key_record: ProjectDeviceKey,
) -> IngestionRequest:
    parsed, raw_body = await read_ingestion_body(request)
    ct = (
        request.headers.get("content-type") or "application/json"
    ).split(";")[0].strip().lower()
    # HTTP header booleans arrive as strings; normalise to bool
    x_no_label = (x_no_label_raw or "").lower() in ("true", "1", "yes")
    return IngestionRequest(
        project_id=key_record.project_id,
        key_record=key_record,
        category=category,
        raw_body=raw_body,
        parsed=parsed,
        content_type=ct,
        x_signature=x_signature,
        x_sample_token=x_sample_token,
        x_label=x_label,
        x_no_label=x_no_label,
        x_file_name=x_file_name,
    )


@router.post("/training/data", status_code=201)
async def ingest_training(
    request: Request,
    x_signature:  Optional[str] = Header(None, alias="x-signature"),
    x_sample_token: Optional[str] = Header(None, alias="x-sample-token"),
    x_label:      Optional[str] = Header(None, alias="x-label"),
    x_no_label:   Optional[str] = Header(None, alias="x-no-label"),
    x_file_name:  Optional[str] = Header(None, alias="x-file-name"),
    key_record: ProjectDeviceKey = Depends(resolve_api_key),
    db: Session = Depends(get_db),
):
    """Ingest a training sample uploaded directly from a device."""
    req = await _build_ingestion_request(
        request, "training", x_signature, x_sample_token, x_label, x_no_label, x_file_name, key_record,
    )
    return process_ingestion(req, db)


@router.post("/testing/data", status_code=201)
async def ingest_testing(
    request: Request,
    x_signature:  Optional[str] = Header(None, alias="x-signature"),
    x_sample_token: Optional[str] = Header(None, alias="x-sample-token"),
    x_label:      Optional[str] = Header(None, alias="x-label"),
    x_no_label:   Optional[str] = Header(None, alias="x-no-label"),
    x_file_name:  Optional[str] = Header(None, alias="x-file-name"),
    key_record: ProjectDeviceKey = Depends(resolve_api_key),
    db: Session = Depends(get_db),
):
    """Ingest a testing sample uploaded directly from a device."""
    req = await _build_ingestion_request(
        request, "testing", x_signature, x_sample_token, x_label, x_no_label, x_file_name, key_record,
    )
    return process_ingestion(req, db)


@router.post("/anomaly/data", status_code=201)
async def ingest_anomaly(
    request: Request,
    x_signature:  Optional[str] = Header(None, alias="x-signature"),
    x_sample_token: Optional[str] = Header(None, alias="x-sample-token"),
    x_label:      Optional[str] = Header(None, alias="x-label"),
    x_no_label:   Optional[str] = Header(None, alias="x-no-label"),
    x_file_name:  Optional[str] = Header(None, alias="x-file-name"),
    key_record: ProjectDeviceKey = Depends(resolve_api_key),
    db: Session = Depends(get_db),
):
    """Ingest an anomaly-detection sample; stored as testing split."""
    req = await _build_ingestion_request(
        request, "anomaly", x_signature, x_sample_token, x_label, x_no_label, x_file_name, key_record,
    )
    return process_ingestion(req, db)
