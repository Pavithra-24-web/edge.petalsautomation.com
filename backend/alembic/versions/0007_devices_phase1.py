"""Devices Phase 1 — expand device schema

Revision ID: 0007_devices_phase1
Revises: 0006_testresultstatus_uncertain
Create Date: 2026-04-22 00:00:00

Adds the full Phase 1 device fields to the existing `devices` table:
  - device_id (stable external HW identifier, unique per project)
  - connection, protocol_version, supports_snapshot_streaming, remote_mgmt_host
  - sensors (JSON), updated_at, deleted_at (soft-delete)
Existing transitional columns (is_online, ip_address) are kept intact.
"""

from alembic import op
import sqlalchemy as sa

revision      = "0007_devices_phase1"
down_revision = "0006_testresultstatus_uncertain"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    op.add_column("devices", sa.Column("device_id",   sa.String(), nullable=True))
    op.add_column("devices", sa.Column("connection",  sa.String(), nullable=True))
    op.add_column("devices", sa.Column("protocol_version", sa.String(), nullable=True))
    op.add_column("devices", sa.Column("supports_snapshot_streaming",
                                       sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("devices", sa.Column("remote_mgmt_host", sa.String(), nullable=True))
    op.add_column("devices", sa.Column("sensors",  sa.JSON(), nullable=True))
    op.add_column("devices", sa.Column("updated_at", sa.DateTime(), nullable=True))
    op.add_column("devices", sa.Column("deleted_at", sa.DateTime(), nullable=True))

    # Index: fast active-device lookups per project
    op.create_index("ix_devices_project_active", "devices", ["project_id", "deleted_at"])

    # Index: look up a device by its external device_id within a project
    op.create_index("ix_devices_device_id", "devices", ["device_id"])

    # Unique constraint: one device_id per project (NULLs are naturally excluded by PG)
    op.create_unique_constraint(
        "uq_device_project_device_id", "devices", ["project_id", "device_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_device_project_device_id", "devices", type_="unique")
    op.drop_index("ix_devices_device_id", table_name="devices")
    op.drop_index("ix_devices_project_active", table_name="devices")

    for col in ("deleted_at", "updated_at", "sensors", "remote_mgmt_host",
                "supports_snapshot_streaming", "protocol_version", "connection", "device_id"):
        op.drop_column("devices", col)
