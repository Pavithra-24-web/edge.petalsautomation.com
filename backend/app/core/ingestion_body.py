"""
Raw-body reader for device ingestion endpoints — Phase 3.

read_ingestion_body(request) -> (parsed_data, raw_bytes)

raw_bytes is the original wire payload so callers can pass it to HMAC
verification without the data being altered by FastAPI's JSON parsing.

Supported content-types:
  application/json  — default; also used for firmware that omits the header
  application/cbor  — requires the cbor2 package; degrades with a 415 if absent
"""
from __future__ import annotations

import json
from typing import Any, Tuple

from fastapi import HTTPException, Request

try:
    import cbor2 as _cbor2  # type: ignore
    _CBOR_AVAILABLE = True
except ImportError:
    _cbor2 = None
    _CBOR_AVAILABLE = False


async def read_ingestion_body(request: Request) -> Tuple[Any, bytes]:
    """
    Read the raw request body and decode it as JSON or CBOR.

    Returns ``(parsed_data, raw_bytes)``.
    ``raw_bytes`` is always the unmodified wire bytes and must be used for
    HMAC computation — never the re-serialised parsed form.
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty request body")

    ct = (
        request.headers.get("content-type") or "application/json"
    ).split(";")[0].strip().lower()

    if ct == "application/cbor":
        if not _CBOR_AVAILABLE:
            raise HTTPException(
                status_code=415,
                detail="CBOR not supported on this server — install the cbor2 package",
            )
        try:
            parsed = _cbor2.loads(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid CBOR body: {exc}")
    else:
        # application/json or any unlabelled content-type from firmware
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")

    return parsed, raw
