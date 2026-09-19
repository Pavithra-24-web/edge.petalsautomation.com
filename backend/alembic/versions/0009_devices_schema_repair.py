"""Devices schema repair for stamped / partially-applied databases.

Revision ID: 0009_devices_schema_repair
Revises: 0008_devices_supporting_tables
Create Date: 2026-04-24 00:00:00

Purpose:
  Repair environments where Alembic was stamped forward but device migrations
  were not fully executed. This migration is intentionally defensive:

  - adds any missing Phase 1 columns on `devices`
  - recreates missing device indexes / unique constraint
  - creates Phase 1 supporting tables if absent
  - creates their expected indexes if absent

It does not backfill data and does not mutate legacy JSON fields.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0009_devices_schema_repair"
down_revision = "0008_devices_supporting_tables"
branch_labels = None
depends_on = None


def _has_index(inspector, table_name: str, index_name: str) -> bool:
    return any(ix.get("name") == index_name for ix in inspector.get_indexes(table_name))


def _has_unique(inspector, table_name: str, constraint_name: str) -> bool:
    return any(
        uq.get("name") == constraint_name
        for uq in inspector.get_unique_constraints(table_name)
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # Repair the devices table if it exists but is missing Phase 1 fields.
    if inspector.has_table("devices"):
        device_columns = {col["name"] for col in inspector.get_columns("devices")}

        missing_columns = [
            ("device_id", sa.Column("device_id", sa.String(), nullable=True)),
            ("connection", sa.Column("connection", sa.String(), nullable=True)),
            ("protocol_version", sa.Column("protocol_version", sa.String(), nullable=True)),
            (
                "supports_snapshot_streaming",
                sa.Column(
                    "supports_snapshot_streaming",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.text("false"),
                ),
            ),
            ("remote_mgmt_host", sa.Column("remote_mgmt_host", sa.String(), nullable=True)),
            ("sensors", sa.Column("sensors", sa.JSON(), nullable=True)),
            ("updated_at", sa.Column("updated_at", sa.DateTime(), nullable=True)),
            ("deleted_at", sa.Column("deleted_at", sa.DateTime(), nullable=True)),
        ]

        for name, column in missing_columns:
            if name not in device_columns:
                op.add_column("devices", column)

        # Refresh inspection after potential ALTER TABLE changes.
        inspector = inspect(bind)
        device_columns = {col["name"] for col in inspector.get_columns("devices")}

        if "device_id" in device_columns and not _has_index(
            inspector, "devices", "ix_devices_device_id"
        ):
            op.create_index("ix_devices_device_id", "devices", ["device_id"])

        if {"project_id", "deleted_at"}.issubset(device_columns) and not _has_index(
            inspector, "devices", "ix_devices_project_active"
        ):
            op.create_index(
                "ix_devices_project_active",
                "devices",
                ["project_id", "deleted_at"],
            )

        if {"project_id", "device_id"}.issubset(device_columns) and not _has_unique(
            inspector, "devices", "uq_device_project_device_id"
        ):
            op.create_unique_constraint(
                "uq_device_project_device_id",
                "devices",
                ["project_id", "device_id"],
            )

    # Create project_device_keys if 0008 never actually ran.
    if not inspector.has_table("project_device_keys"):
        op.create_table(
            "project_device_keys",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("api_key", sa.String(), nullable=False, unique=True),
            sa.Column("hmac_key", sa.String(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_project_device_keys_project", "project_device_keys", ["project_id"])
    else:
        inspector = inspect(bind)
        if not _has_index(inspector, "project_device_keys", "ix_project_device_keys_project"):
            op.create_index(
                "ix_project_device_keys_project",
                "project_device_keys",
                ["project_id"],
            )

    # Create device_inference_logs if 0008 never actually ran.
    if not inspector.has_table("device_inference_logs"):
        op.create_table(
            "device_inference_logs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("device_id", sa.String(), sa.ForeignKey("devices.id"), nullable=True),
            sa.Column("impulse_id", sa.String(), sa.ForeignKey("impulses.id"), nullable=True),
            sa.Column("result", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_device_inference_logs_project_created",
            "device_inference_logs",
            ["project_id", "created_at"],
        )
        op.create_index(
            "ix_device_inference_logs_device",
            "device_inference_logs",
            ["device_id"],
        )
    else:
        inspector = inspect(bind)
        if not _has_index(
            inspector, "device_inference_logs", "ix_device_inference_logs_project_created"
        ):
            op.create_index(
                "ix_device_inference_logs_project_created",
                "device_inference_logs",
                ["project_id", "created_at"],
            )
        if not _has_index(inspector, "device_inference_logs", "ix_device_inference_logs_device"):
            op.create_index(
                "ix_device_inference_logs_device",
                "device_inference_logs",
                ["device_id"],
            )


def downgrade() -> None:
    """No-op downgrade.

    This migration repairs partial state and should not try to guess which
    schema elements pre-existed in a broken environment.
    """
    pass
