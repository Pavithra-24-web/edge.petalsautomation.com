"""
Ingestion authentication and HMAC verification helpers — Phase 3.

Public surface:
    resolve_api_key()       FastAPI dependency — validates x-api-key header,
                            returns the matching ProjectDeviceKey record.
    get_sample_hmac_key()   Transitional abstraction — returns the HMAC signing
                            bytes.  Phase 4 wires in per-sample ephemeral keys
                            here without changing any caller.
    build_hmac_body()       Normalises the raw payload for HMAC computation:
                            zeroes the Edge Impulse "signature" field so the
                            device and server compute over the same bytes.
    verify_hmac()           Timing-safe HMAC-SHA256 check.  Raises HTTPException
                            on every failure so callers stay clean.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.devices import ProjectDeviceKey


# ─── API key auth ────────────────────────────────────────────────────────────

def resolve_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    db: Session = Depends(get_db),
) -> ProjectDeviceKey:
    """
    Resolve a ProjectDeviceKey from the x-api-key request header.

    Raises 401 for a missing or unknown key.
    Raises 403 for a revoked (is_active=False) key.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")

    record = (
        db.query(ProjectDeviceKey)
        .filter(ProjectDeviceKey.api_key == x_api_key)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not record.is_active:
        raise HTTPException(status_code=403, detail="API key has been revoked")
    return record


# ─── HMAC key abstraction ─────────────────────────────────────────────────────

def get_sample_hmac_key(
    key_record: ProjectDeviceKey,
    *,
    sample_token: Optional[str] = None,
) -> bytes:
    """
    Return the HMAC signing key bytes for an ingestion request.

    Phase 3: always uses the project-level hmac_key from ProjectDeviceKey.
    Phase 4 hook: supply sample_token (e.g. from an x-sample-token header)
    and replace the body of this function to look up a per-sample ephemeral
    key from a SampleHmacKey table.  All callers remain unchanged.
    """
    if sample_token:
        from app.core.sample_hmac_store import sample_hmac_store  # avoid circular at module load
        rec = sample_hmac_store.get(sample_token)
        if rec is not None:
            return bytes.fromhex(rec.key_hex)
    return bytes.fromhex(key_record.hmac_key)


# ─── HMAC body normalisation ──────────────────────────────────────────────────

def _zero_sig_json(raw: bytes, sig_hex: str) -> bytes:
    """
    Replace the JSON "signature" value with 64 zeros in the raw bytes.

    Handles both compact (`"signature":"<hex>"`) and spaced
    (`"signature": "<hex>"`) serialisations.  Falls back to a regex replace
    for any other whitespace variant.
    """
    zeros = b"0" * 64
    sig_enc = sig_hex.encode()

    for pat, rep in [
        (b'"signature":"' + sig_enc + b'"',  b'"signature":"' + zeros + b'"'),
        (b'"signature": "' + sig_enc + b'"', b'"signature": "' + zeros + b'"'),
    ]:
        if pat in raw:
            return raw.replace(pat, rep, 1)

    return re.sub(
        rb'"signature"\s*:\s*"[0-9a-fA-F]+"',
        b'"signature":"' + zeros + b'"',
        raw,
        count=1,
    )


def build_hmac_body(raw_body: bytes, parsed: Any, content_type: str) -> bytes:
    """
    Return the canonical bytes the device used when computing its HMAC.

    Edge Impulse canonical flow: the outer envelope contains a "signature"
    field initialised to 64 zeros before HMAC computation.  The device then
    fills in the real signature after signing.  To verify, we zero the field
    again and recompute.

    For JSON: zero the field in the raw bytes (no re-serialisation).
    For CBOR: decode, zero the field, re-encode (CBOR is binary, can't do
              byte-level text substitution).
    If no "signature" field is present the raw body is returned unchanged
    (caller can still verify against an x-signature header).
    """
    if not isinstance(parsed, dict) or "signature" not in parsed:
        return raw_body

    sig_hex: str = parsed.get("signature") or ""

    if "cbor" in content_type:
        try:
            import cbor2  # type: ignore
            zeroed = dict(parsed)
            zeroed["signature"] = "0" * 64
            return cbor2.dumps(zeroed)
        except Exception:
            return raw_body

    return _zero_sig_json(raw_body, sig_hex)


# ─── HMAC verification ────────────────────────────────────────────────────────

def verify_hmac(
    body_for_hmac: bytes,
    key_record: ProjectDeviceKey,
    signature_hex: Optional[str],
    *,
    sample_token: Optional[str] = None,
) -> None:
    """
    Timing-safe HMAC-SHA256 verification of a device payload.

    `body_for_hmac` must already be normalised (signature field zeroed for
    body-embedded signatures; raw bytes for header-carried signatures).

    Raises HTTPException(401) on every failure path.
    """
    if not signature_hex:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")

    key_bytes = get_sample_hmac_key(key_record, sample_token=sample_token)
    expected_hex = hmac.new(key_bytes, body_for_hmac, hashlib.sha256).hexdigest()

    try:
        provided_bytes = bytes.fromhex(signature_hex)
        expected_bytes = bytes.fromhex(expected_hex)
    except ValueError:
        raise HTTPException(status_code=401, detail="Malformed HMAC signature (not valid hex)")

    if not hmac.compare_digest(provided_bytes, expected_bytes):
        raise HTTPException(status_code=401, detail="HMAC signature mismatch")
