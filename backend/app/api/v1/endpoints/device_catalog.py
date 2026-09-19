"""
Device catalog — read-only browse (Target Device Phase 1 / 1b / 2).

Serves `DeviceCatalogEntry` rows: known hardware, grouped by family, each
naming the existing `DeployTarget` it resolves to, plus (Phase 2) each
entry's `DeviceSpecification` — the hardware characteristics later phases
compute against. Nothing here creates, updates, or deletes catalog rows or
specifications — that's out of scope for this phase.

Routes (mounted at `/device-catalog`):
  * GET /            → flat list, optional `?family=`, `?q=`, `?device_class=`
                        filters (composable). Identity fields only, unless
                        `?include=specification` embeds the full spec.
  * GET /families    → grouped by family, same `?include=specification` opt-in.
  * GET /{slug}      → single entry, always with its full specification
                        inline. 404 if unknown.

The list/family routes default to identity-only deliberately: the topbar
device picker calls them on every open and needs none of the Phase 2 columns,
so adding a specification must change nothing there unless asked for — the
default payload shape is byte-for-byte what it was before Phase 2.

Any authenticated user may read the catalog — it names no project or user
data, so ownership checks don't apply; this reuses `get_current_user` like
other authenticated-but-unscoped endpoints.
"""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.devices import DeviceCatalogEntry, DeviceSpecification
from app.models.user import DeviceClass, User

router = APIRouter()


def _serialize_specification(spec: DeviceSpecification | None) -> dict | None:
    """NULL passes through as JSON `null` — never substituted with a
    placeholder like "Unknown" or 0. That string belongs in the UI; the
    absence belongs in the payload (Phase 2, §6)."""
    if spec is None:
        return None
    return {
        "vendor": spec.vendor,
        "device_class": spec.device_class.value,
        "processor_family": spec.processor_family,
        "processor": spec.processor,
        "cpu_architecture": spec.cpu_architecture,
        "clock_rate_mhz": spec.clock_rate_mhz,
        "ram_kb": spec.ram_kb,
        "rom_kb": spec.rom_kb,
        "runtime_environment": spec.runtime_environment,
        "has_ai_accelerator": spec.has_ai_accelerator,
        "ai_accelerator": spec.ai_accelerator,
        "latency_budget_ms": spec.latency_budget_ms,
        "latency_budget_basis": spec.latency_budget_basis,
        "supported_precisions": spec.supported_precisions,
    }


def _serialize(entry: DeviceCatalogEntry, *, include_spec: bool = False) -> dict:
    data = {
        "slug": entry.slug,
        "display_name": entry.display_name,
        "family": entry.family,
        "deploy_target": entry.deploy_target.value,
        "accelerator_note": entry.accelerator_note,
    }
    if include_spec:
        data["specification"] = _serialize_specification(entry.specification)
    return data


def _validate_device_class(device_class: str | None) -> None:
    if device_class is not None and device_class not in {c.value for c in DeviceClass}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown device_class {device_class!r}. "
                   f"Valid values: {', '.join(sorted(c.value for c in DeviceClass))}",
        )


@router.get("/")
def list_entries(
    family: str | None = Query(None, description="Filter to one device family"),
    q: str | None = Query(None, description="Case-insensitive substring over board name / family"),
    device_class: str | None = Query(None, description="Filter to one device class"),
    include: str | None = Query(None, description="Set to 'specification' to embed the full spec"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[dict]:
    """Flat list of catalog entries, optionally filtered by family, search
    term, and/or device class — all three compose.

    `q` is a SQL `ilike` over `display_name` and `family` — the catalog is
    seed data, small enough that a substring scan needs no index, but still
    done in SQL rather than pulled into Python to filter. Identity fields
    only by default; pass `?include=specification` to embed each entry's
    full Phase 2 specification.
    """
    _validate_device_class(device_class)
    include_spec = include == "specification"

    query = db.query(DeviceCatalogEntry)
    if include_spec:
        query = query.options(joinedload(DeviceCatalogEntry.specification))
    if family:
        query = query.filter(DeviceCatalogEntry.family == family)
    if q:
        pattern = f"%{q}%"
        query = query.filter(or_(
            DeviceCatalogEntry.display_name.ilike(pattern),
            DeviceCatalogEntry.family.ilike(pattern),
        ))
    if device_class:
        query = query.join(
            DeviceSpecification, DeviceSpecification.catalog_entry_id == DeviceCatalogEntry.id,
        ).filter(DeviceSpecification.device_class == DeviceClass(device_class))
    entries = query.order_by(DeviceCatalogEntry.family, DeviceCatalogEntry.display_name).all()
    return [_serialize(e, include_spec=include_spec) for e in entries]


@router.get("/families")
def list_families(
    include: str | None = Query(None, description="Set to 'specification' to embed the full spec"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[dict]:
    """Catalog entries grouped by family, in family/display-name order.
    Identity fields only by default; pass `?include=specification` to embed
    each entry's full Phase 2 specification."""
    include_spec = include == "specification"
    query = db.query(DeviceCatalogEntry)
    if include_spec:
        query = query.options(joinedload(DeviceCatalogEntry.specification))
    entries = query.order_by(DeviceCatalogEntry.family, DeviceCatalogEntry.display_name).all()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[entry.family].append(_serialize(entry, include_spec=include_spec))
    return [{"family": family, "entries": items} for family, items in grouped.items()]


@router.get("/{slug}")
def get_entry(
    slug: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> dict:
    """Single catalog entry by slug, with its full specification inline.
    404 if unknown."""
    entry = (
        db.query(DeviceCatalogEntry)
        .options(joinedload(DeviceCatalogEntry.specification))
        .filter(DeviceCatalogEntry.slug == slug)
        .first()
    )
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device catalog entry {slug!r} not found",
        )
    return _serialize(entry, include_spec=True)
