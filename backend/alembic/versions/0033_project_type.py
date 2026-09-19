"""Add project_type to projects.

Revision ID: 0033_project_type
Revises: 0032_target_device_config
Create Date: 2026-09-04 00:00:00

Presentation-only field selecting which content the shared dashboard pages
render for a project ("object_detection" | "motion"). Every existing row is
backfilled to "object_detection", which is what it already is. See
docs/Motion recognition/motion_phase0.md §2 / §8.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0033_project_type"
down_revision = "0032_target_device_config"
branch_labels = None
depends_on = None


def _has_column(inspector, table: str, col: str) -> bool:
    return any(c["name"] == col for c in inspector.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not _has_column(inspector, "projects", "project_type"):
        op.add_column(
            "projects",
            sa.Column("project_type", sa.String(), nullable=True),
        )
        op.execute("UPDATE projects SET project_type = 'object_detection' WHERE project_type IS NULL")
        op.alter_column(
            "projects", "project_type",
            existing_type=sa.String(),
            nullable=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if _has_column(inspector, "projects", "project_type"):
        op.drop_column("projects", "project_type")
