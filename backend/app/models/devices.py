"""
SQLAlchemy ORM models for the Devices feature — supporting tables.

Kept separate from user.py so the devices domain can evolve independently.
Both tables are additive and safe to introduce alongside the Phase 1 Device model.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, JSON, Index, BigInteger,
    Integer, Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.user import gen_uuid, DeployTarget, DeviceClass


class ProjectDeviceKey(Base):
    """
    Project-scoped credentials used by devices to authenticate API calls.

    Stored now; runtime enforcement (header validation, HMAC verification)
    is wired up in a later phase. The `api_key` value is the raw token that
    will be sent in an `X-Device-Key` header; `hmac_key` is the shared
    secret used to sign payloads.  Both should be generated server-side and
    returned once on creation — never stored in plaintext in client code.
    """
    __tablename__ = "project_device_keys"

    id         = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    name       = Column(String, nullable=True)          # human-readable label
    api_key    = Column(String, nullable=False, unique=True)
    hmac_key   = Column(String, nullable=False)
    is_active  = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project")


class DeviceInferenceLog(Base):
    """
    Lightweight record of an inference event reported by a device.

    `device_id` references the internal Device PK (Device.id), not the
    external device_id string — this keeps the FK stable even if the
    hardware identifier is updated.  `impulse_id` is nullable so logs can
    be stored before an impulse is linked to the device.
    """
    __tablename__ = "device_inference_logs"

    id         = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    device_id  = Column(String, ForeignKey("devices.id"), nullable=True)
    impulse_id = Column(String, ForeignKey("impulses.id"), nullable=True)
    result     = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Common query: all logs for a project, most recent first
        Index("ix_device_inference_logs_project_created",
              "project_id", "created_at"),
        # Common query: all logs for a specific device
        Index("ix_device_inference_logs_device", "device_id"),
    )

    project = relationship("Project")
    device  = relationship("Device")
    impulse = relationship("Impulse")


class DevicePackage(Base):
    """
    A published, versioned device-client release (contract §14).

    One row per semver version of the `petal-device-<version>.tar.gz` runtime
    bundle. The columns are exactly the manifest fields (contract §14.4): the
    row is the authoritative index for "latest" and the sole source of truth
    for the sha256 checksum — the installer verifies downloaded bytes against
    it, so no `.sha256` sidecar is ever written to storage (§14.1/§14.6).

    `version` is the primary key: a version string identifies exactly one set
    of bytes for its lifetime (versions are immutable — re-uploading a version
    is a `409 Conflict`, §14.3), so it is naturally unique and needs no
    surrogate id. The tarball itself lives in storage under the
    `device-client/` prefix; only its metadata lives here.
    """
    __tablename__ = "device_packages"

    version    = Column(String, primary_key=True)          # e.g. "v1.0.0"
    filename   = Column(String, nullable=False)            # petal-device-v1.0.0.tar.gz
    size       = Column(BigInteger, nullable=False)        # tarball size in bytes
    sha256     = Column(String, nullable=False)            # 64 hex chars, computed at upload
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DeviceCatalogEntry(Base):
    """
    Known hardware — one row per board PetalEdge recognises (Target Device
    Phase 1, docs/target_device_phase1.md).

    This is the identity layer only: which board, which family, and which of
    the seven existing `DeployTarget` build formats it resolves to. Hardware
    characteristics (processor, RAM, ROM) are Phase 2. Project selection is
    Phase 3. Nothing reads this table yet except the deployment worker's
    device-profile resolver, which this table now supersedes as the source
    of truth for the five profiles it used to hard-code.

    `slug` is the compatibility surface — it is what `device_profile` values
    and API callers key on, so existing slugs (e.g. `arduino_nano_33_ble`)
    must never change once seeded. `deploy_target` always names one of the
    existing seven `DeployTarget` values; a board whose accelerator has no
    matching build format still gets an entry, with the limitation recorded
    in `accelerator_note` rather than the board being omitted.
    """
    __tablename__ = "device_catalog_entries"

    id                = Column(String, primary_key=True, default=gen_uuid)
    slug              = Column(String, nullable=False, unique=True, index=True)
    display_name      = Column(String, nullable=False)
    family            = Column(String, nullable=False, index=True)
    deploy_target     = Column(SAEnum(DeployTarget), nullable=False)
    accelerator_note  = Column(String, nullable=True)
    created_at        = Column(DateTime, nullable=False, default=datetime.utcnow)

    specification = relationship(
        "DeviceSpecification", uselist=False, back_populates="catalog_entry",
    )


class DeviceSpecification(Base):
    """
    Hardware specification for one device catalog entry (Target Device
    Phase 2, docs/target_device_phase2.md) — the characteristics validation
    and estimation later compute against. Nothing consumes these values yet.

    One row per `DeviceCatalogEntry` (`catalog_entry_id` is unique; the FK is
    safe here — unlike `projects.target_device_slug` — because both tables
    are migration-owned seed data and a specification with no board is
    meaningless). Identity (`display_name`, `family`, `deploy_target`) is
    deliberately NOT repeated here: it's composed from the catalog entry at
    serialization time so there is exactly one place that owns it.

    Every column below is nullable except `device_class`, and NULL means
    "unknown" — never zero, never an empty string, never a copied sibling
    figure. `has_ai_accelerator` is a three-state boolean on purpose: `True`
    (has one, named in `ai_accelerator`), `False` (confirmed to have none),
    `NULL` (not yet researched) — collapsing `False` and `NULL` would lose
    exactly the distinction this column exists to keep. `clock_rate_mhz`,
    `ram_kb`, and `rom_kb` are always canonical units (MHz / KB) so Phase 5
    can do arithmetic directly; formatting for display is the UI's job.

    No column here carries a Python-side `default=` — a default is exactly
    the silent defaulting this phase forbids. `created_at` is bookkeeping
    about the row, not a hardware fact, so it's exempt.
    """
    __tablename__ = "device_specifications"

    id                    = Column(String, primary_key=True, default=gen_uuid)
    catalog_entry_id      = Column(
        String, ForeignKey("device_catalog_entries.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )

    vendor                = Column(String, nullable=True)
    device_class          = Column(SAEnum(DeviceClass), nullable=False)
    processor_family      = Column(String, nullable=True)
    processor             = Column(String, nullable=True)
    cpu_architecture      = Column(String, nullable=True)
    clock_rate_mhz        = Column(Integer, nullable=True)
    ram_kb                = Column(Integer, nullable=True)
    rom_kb                = Column(Integer, nullable=True)
    runtime_environment   = Column(String, nullable=True)
    has_ai_accelerator    = Column(Boolean, nullable=True)
    ai_accelerator        = Column(String, nullable=True)
    latency_budget_ms     = Column(Integer, nullable=True)
    latency_budget_basis  = Column(String, nullable=True)
    supported_precisions  = Column(JSON, nullable=True)
    created_at            = Column(DateTime, nullable=False, default=datetime.utcnow)

    catalog_entry = relationship("DeviceCatalogEntry", back_populates="specification")
