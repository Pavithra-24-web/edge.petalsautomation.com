"""
Device-client packages & installer (contract §14).

Stores and serves versioned device-client packages: a single gzipped tarball
per semver version. The `DevicePackage` row is the authoritative index and the
sole source of truth for each package's sha256 (§14.1/§14.6) — the manifest
serves it and the installer verifies downloaded bytes against it.

Scope (Phase 1): store + serve versioned packages. No channels, rollback,
auto-update, or compatibility gating.

Routes (mounted at `/device-client`, §14.3):
  * GET  /latest              → manifest JSON for the highest semver
  * GET  /{version}           → manifest JSON for a specific version (404 if unknown)
  * GET  /{version}/download  → streams the tarball bytes (public)
  * POST /upload              → admin only; publish a new version

`GET /install.sh` is mounted at the application root (outside `/api/v1`) in
`app/main.py`, not here — it must be reachable without the API prefix or auth.
"""
from __future__ import annotations

import hashlib
import io
import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.authz import require_admin
from app.core.database import get_db
from app.core.storage import storage
from app.models.devices import DevicePackage
from app.models.user import User

router = APIRouter()

# Semver with a mandatory leading `v` — the `v` is part of the version string
# everywhere (§14.2): filenames, VERSION, the {version} path, the manifest field.
_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

# All package bytes live under this single storage key prefix (§14.6).
_STORAGE_PREFIX = "device-client/"


def _semver_key(version: str) -> tuple[int, int, int]:
    """Sort key for a validated `vMAJOR.MINOR.PATCH` string."""
    m = _VERSION_RE.match(version)
    assert m is not None  # only called on already-validated versions
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _storage_key(version: str) -> str:
    return f"{_STORAGE_PREFIX}petal-device-{version}.tar.gz"


def _manifest(pkg: DevicePackage, request: Request) -> dict:
    """Build the manifest JSON (§14.4) for a package row.

    `download_url` is derived from the incoming request's base URL so the
    manifest advertises the same host the caller reached us on, rather than a
    hard-coded domain.
    """
    base = str(request.base_url).rstrip("/")
    return {
        "version": pkg.version,
        "filename": pkg.filename,
        "size": pkg.size,
        "sha256": pkg.sha256,
        "download_url": f"{base}/api/v1/device-client/{pkg.version}/download",
        "created_at": pkg.created_at.isoformat() if pkg.created_at else None,
    }


@router.get("/latest")
def get_latest(request: Request, db: Session = Depends(get_db)) -> dict:
    """Manifest JSON for the highest stored semver version (§14.4). Public.

    Reads only the `DevicePackage` rows — never downloads the tarball bytes.
    """
    packages = db.query(DevicePackage).all()
    if not packages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No device-client packages published yet",
        )
    latest = max(packages, key=lambda p: _semver_key(p.version))
    return _manifest(latest, request)


@router.get("/{version}")
def get_version(version: str, request: Request, db: Session = Depends(get_db)) -> dict:
    """Manifest JSON for a specific version (§14.4). Public. 404 if unknown.

    Reads only the `DevicePackage` row — never downloads the tarball bytes.
    """
    pkg = db.query(DevicePackage).filter(DevicePackage.version == version).first()
    if pkg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device-client package {version!r} not found",
        )
    return _manifest(pkg, request)


@router.get("/{version}/download")
def download_version(version: str, db: Session = Depends(get_db)) -> StreamingResponse:
    """Stream the tarball bytes for a version (§14.4). Public.

    Serves `application/gzip` as an attachment straight from storage. 404 if the
    version is unknown.
    """
    pkg = db.query(DevicePackage).filter(DevicePackage.version == version).first()
    if pkg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device-client package {version!r} not found",
        )
    data = storage.download_bytes(_storage_key(version))
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{pkg.filename}"'},
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_package(
    request: Request,
    version: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> dict:
    """Publish a new device-client version (§14.3). Admin only.

    The server is the sole authority for sha256, size, and stored metadata — the
    client sends no checksum. Steps: validate version → reject a duplicate with
    409 (versions are immutable) → read bytes → compute sha256 + size →
    `upload_bytes` to storage → insert the row → return the manifest.
    """
    if not _VERSION_RE.match(version):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid version {version!r}; expected vMAJOR.MINOR.PATCH",
        )

    # Versions are immutable — never overwrite a published tarball or its sha256.
    if db.query(DevicePackage).filter(DevicePackage.version == version).first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Device-client package {version!r} already published",
        )

    data = file.file.read()
    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data)
    filename = f"petal-device-{version}.tar.gz"

    storage.upload_bytes(data, _storage_key(version), content_type="application/gzip")

    pkg = DevicePackage(
        version=version,
        filename=filename,
        size=size,
        sha256=sha256,
    )
    db.add(pkg)
    db.commit()
    db.refresh(pkg)

    return _manifest(pkg, request)
