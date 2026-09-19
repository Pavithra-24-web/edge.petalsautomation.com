"""Add device_type column to training_jobs.

Revision ID: 0012_training_device_type
Revises: 0011_sample_payload_hash
Create Date: 2026-04-29 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0012_training_device_type"
down_revision = "0011_sample_payload_hash"
branch_labels = None
depends_on = None


def _has_column(inspector, table: str, col: str) -> bool:
    return any(c["name"] == col for c in inspector.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not _has_column(inspector, "training_jobs", "device_type"):
        op.add_column(
            "training_jobs",
            sa.Column("device_type", sa.String(), nullable=True, server_default="cpu"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if _has_column(inspector, "training_jobs", "device_type"):
        op.drop_column("training_jobs", "device_type")
