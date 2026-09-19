"""Device/Deployment targeting columns + DeviceUpdateHistory table.

Revision ID: 0010_device_deployment_targeting
Revises: 0009_devices_schema_repair
Create Date: 2026-04-24 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0010_device_deployment_targeting"
down_revision = "0009_devices_schema_repair"
branch_labels = None
depends_on = None


def _has_column(inspector, table: str, col: str) -> bool:
    return any(c["name"] == col for c in inspector.get_columns(table))


def _has_table(inspector, table: str) -> bool:
    return table in inspector.get_table_names()


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)

    # ── devices ───────────────────────────────────────────────────────────────
    for col in ("deployment_target", "device_profile",
                "installed_deployment_id", "installed_model_version"):
        if not _has_column(inspector, "devices", col):
            op.add_column("devices", sa.Column(col, sa.String(), nullable=True))

    # ── deployments ───────────────────────────────────────────────────────────
    for col in ("deployment_target", "device_profile", "project_id"):
        if not _has_column(inspector, "deployments", col):
            op.add_column("deployments", sa.Column(col, sa.String(), nullable=True))

    # ── device_update_history ─────────────────────────────────────────────────
    if not _has_table(inspector, "device_update_history"):
        op.create_table(
            "device_update_history",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(),
                      sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("device_id", sa.String(),
                      sa.ForeignKey("devices.id"), nullable=False),
            sa.Column("deployment_id", sa.String(),
                      sa.ForeignKey("deployments.id"), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("message", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), onupdate=sa.func.now()),
        )


def downgrade():
    op.drop_table("device_update_history")
    for col in ("deployment_target", "device_profile", "project_id"):
        op.drop_column("deployments", col)
    for col in ("deployment_target", "device_profile",
                "installed_deployment_id", "installed_model_version"):
        op.drop_column("devices", col)
