"""Devices Phase 1 — supporting tables

Revision ID: 0008_devices_supporting_tables
Revises: 0007_devices_phase1
Create Date: 2026-04-22 00:00:00

Creates two new tables:
  - project_device_keys  : project-scoped device credentials (api_key + hmac_key)
  - device_inference_logs: lightweight per-device inference event log

Both are additive and do not touch existing tables.
"""

from alembic import op
import sqlalchemy as sa

revision      = "0008_devices_supporting_tables"
down_revision = "0007_devices_phase1"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    op.create_table(
        "project_device_keys",
        sa.Column("id",         sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name",       sa.String(), nullable=True),
        sa.Column("api_key",    sa.String(), nullable=False, unique=True),
        sa.Column("hmac_key",   sa.String(), nullable=False),
        sa.Column("is_active",  sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_project_device_keys_project", "project_device_keys", ["project_id"])

    op.create_table(
        "device_inference_logs",
        sa.Column("id",         sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("device_id",  sa.String(), sa.ForeignKey("devices.id"),  nullable=True),
        sa.Column("impulse_id", sa.String(), sa.ForeignKey("impulses.id"), nullable=True),
        sa.Column("result",     sa.JSON(),   nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_device_inference_logs_project_created",
        "device_inference_logs", ["project_id", "created_at"],
    )
    op.create_index(
        "ix_device_inference_logs_device",
        "device_inference_logs", ["device_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_device_inference_logs_device",          table_name="device_inference_logs")
    op.drop_index("ix_device_inference_logs_project_created", table_name="device_inference_logs")
    op.drop_table("device_inference_logs")

    op.drop_index("ix_project_device_keys_project", table_name="project_device_keys")
    op.drop_table("project_device_keys")
