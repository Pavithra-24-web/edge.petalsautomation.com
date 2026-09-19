"""Scope post_processing_settings by impulse_id.

Revision ID: 0016_pp_settings_impulse
Revises: 0015_jobstatus_cancelled
Create Date: 2026-05-18 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016_pp_settings_impulse"
down_revision = "0015_jobstatus_cancelled"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "post_processing_settings",
        sa.Column("impulse_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_post_processing_settings_impulse_id_impulses",
        "post_processing_settings",
        "impulses",
        ["impulse_id"],
        ["id"],
    )
    op.drop_constraint(
        "uq_post_processing_settings_project_id",
        "post_processing_settings",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_post_processing_settings_project_impulse_id",
        "post_processing_settings",
        ["project_id", "impulse_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_post_processing_settings_project_impulse_id",
        "post_processing_settings",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_post_processing_settings_project_id",
        "post_processing_settings",
        ["project_id"],
    )
    op.drop_constraint(
        "fk_post_processing_settings_impulse_id_impulses",
        "post_processing_settings",
        type_="foreignkey",
    )
    op.drop_column("post_processing_settings", "impulse_id")
