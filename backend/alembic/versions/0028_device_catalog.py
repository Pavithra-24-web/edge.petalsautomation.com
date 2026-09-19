"""Device catalog — known hardware, as data

Revision ID: 0028_device_catalog
Revises: 0027_device_packages
Create Date: 2026-08-17 00:00:00.000000

Target Device Phase 1 (docs/target_device_phase1.md). Creates
`device_catalog_entries`: one row per board PetalEdge recognises, naming the
family it belongs to and which of the seven existing `deploytarget` values it
resolves to. Identity only — no processor/RAM/ROM columns (Phase 2).

Seeds the 15 boards from the spec plus `generic_tflite`. Five slugs
(`generic_tflite`, `arduino_nano_33_ble`, `esp32_devkit`, `raspberry_pi_4`,
`unoq`) are reused verbatim from the worker's hard-coded
`_SUPPORTED_DEVICE_PROFILES` map so its existing resolutions are preserved
exactly (docs/target_device_phase1.md, "Reuse the existing slugs verbatim").

`deploy_target` reuses the existing Postgres `deploytarget` enum type
(created for `deployments.target`) — `create_type=False` so this migration
does not try to recreate it on installs where it already exists (from
create_all()). On a database built via `alembic upgrade head` alone, that
type has never existed (0005/0013 no-op on such installs — see their
docstrings), so this migration creates it here, once, with the full value
set — the only place in the migration-only chain that needs it to exist.

Additive; does not touch existing tables, and does not add a new
`deploytarget` value.
"""
import uuid
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0028_device_catalog"
down_revision = "0027_device_packages"
branch_labels = None
depends_on = None


_DEPLOY_TARGET_ENUM = postgresql.ENUM(
    "tflite", "arduino", "esp32", "raspberry_pi", "unoq", "cpp", "pxe",
    name="deploytarget",
    create_type=False,
)

# (slug, display_name, family, deploy_target, accelerator_note)
#
# The first five reuse the worker's existing slugs and resolutions verbatim
# (deployment_worker.py:328) so `_resolve_deployment_target` behaves exactly
# as it does today. The rest are new boards from the Phase 1 spec.
_SEED_ENTRIES = [
    # ── Reused verbatim from _SUPPORTED_DEVICE_PROFILES ────────────────────
    ("generic_tflite",       "Generic TFLite",              "Generic",        "tflite",       None),
    ("arduino_nano_33_ble",  "Arduino Nano 33 BLE Sense",   "Arduino",        "arduino",      None),
    ("esp32_devkit",         "ESP32 DevKit",                "ESP32",          "esp32",        None),
    ("raspberry_pi_4",       "Raspberry Pi 4",               "Raspberry Pi",   "raspberry_pi", None),
    ("unoq",                 "Arduino UNO Q",                "Arduino",        "unoq",         None),

    # ── New boards ───────────────────────────────────────────────────────
    ("esp32_s3",             "ESP32-S3",                     "ESP32",          "esp32",        None),
    ("raspberry_pi_5",       "Raspberry Pi 5",                "Raspberry Pi",   "raspberry_pi", None),
    (
        "raspberry_pi_pico", "Raspberry Pi Pico", "Raspberry Pi", "cpp",
        "No FPU or NN accelerator on the RP2040; runs via the generic C++ "
        "build, not the Raspberry Pi Python package.",
    ),
    (
        "stm32h7", "STM32H7", "STM32", "cpp",
        "No dedicated NN accelerator; Cortex-M7 CPU-only inference via the "
        "generic C++ build.",
    ),
    (
        "stm32f7", "STM32F7", "STM32", "cpp",
        "No dedicated NN accelerator; Cortex-M7 CPU-only inference via the "
        "generic C++ build.",
    ),
    (
        "nrf52840", "Nordic nRF52840", "Nordic", "cpp",
        "No NN accelerator; Cortex-M4 CPU-only inference via the generic "
        "C++ build.",
    ),
    (
        "jetson_nano", "NVIDIA Jetson Nano", "NVIDIA Jetson", "raspberry_pi",
        "Onboard GPU/CUDA not exploited; runs via the generic Raspberry Pi "
        "Python package on CPU only.",
    ),
    (
        "jetson_orin_nano", "NVIDIA Jetson Orin Nano", "NVIDIA Jetson", "raspberry_pi",
        "Onboard GPU/DLA accelerators not exploited; runs via the generic "
        "Raspberry Pi Python package on CPU only.",
    ),
    (
        "coral_dev_board", "Google Coral Dev Board", "Google Coral", "raspberry_pi",
        "Onboard Edge TPU not exploited; runs via the generic Raspberry Pi "
        "Python package on CPU only (no Edge TPU-compiled model).",
    ),
    (
        "imx_rt1060", "NXP i.MX RT1060", "NXP", "cpp",
        "No dedicated NN accelerator; Cortex-M7 CPU-only inference via the "
        "generic C++ build.",
    ),
    (
        "himax_we_i_plus", "Himax WE-I Plus", "Himax", "cpp",
        "Onboard HX6537 CV/AI accelerator not exploited; runs via the "
        "generic C++ build on the Cortex-M4F core only.",
    ),
]


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        enum_exists = bind.execute(sa.text(
            "SELECT 1 FROM pg_type WHERE typname = 'deploytarget'"
        )).scalar()
        if not enum_exists:
            # String-column install (see 0005/0013): the type was never
            # created. Create it once here, with the full current value set,
            # since this is the first migration that needs it as an actual
            # Postgres type.
            _DEPLOY_TARGET_ENUM.create(bind, checkfirst=False)
    # On non-Postgres dialects (e.g. SQLite, used by tests that exercise this
    # migration's upgrade() directly) there is no native enum type at all —
    # `postgresql.ENUM` compiles to a plain column type there regardless.

    op.create_table(
        "device_catalog_entries",
        sa.Column("id",               sa.String(), primary_key=True),
        sa.Column("slug",              sa.String(), nullable=False),
        sa.Column("display_name",      sa.String(), nullable=False),
        sa.Column("family",            sa.String(), nullable=False),
        sa.Column("deploy_target",     _DEPLOY_TARGET_ENUM, nullable=False),
        sa.Column("accelerator_note",  sa.String(), nullable=True),
        sa.Column("created_at",        sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_device_catalog_entries_slug", "device_catalog_entries", ["slug"], unique=True,
    )
    op.create_index(
        "ix_device_catalog_entries_family", "device_catalog_entries", ["family"], unique=False,
    )

    table = sa.table(
        "device_catalog_entries",
        sa.column("id", sa.String()),
        sa.column("slug", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("family", sa.String()),
        sa.column("deploy_target", sa.String()),
        sa.column("accelerator_note", sa.String()),
        sa.column("created_at", sa.DateTime()),
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    op.bulk_insert(table, [
        {
            "id": str(uuid.uuid4()),
            "slug": slug,
            "display_name": display_name,
            "family": family,
            "deploy_target": deploy_target,
            "accelerator_note": accelerator_note,
            "created_at": now,
        }
        for slug, display_name, family, deploy_target, accelerator_note in _SEED_ENTRIES
    ])


def downgrade() -> None:
    op.drop_index("ix_device_catalog_entries_family", table_name="device_catalog_entries")
    op.drop_index("ix_device_catalog_entries_slug", table_name="device_catalog_entries")
    op.drop_table("device_catalog_entries")
