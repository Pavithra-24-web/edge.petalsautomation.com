"""Add post_processing_settings and processing_jobs tables.

Revision ID: 0014_post_processing_foundation
Revises: 0013_add_pxe_deploy_target
Create Date: 2026-05-14 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014_post_processing_foundation"
down_revision = "0013_add_pxe_deploy_target"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "post_processing_settings",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("tracking_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("keep_grace", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("max_observations", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("class_filter", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", name="uq_post_processing_settings_project_id"),
    )
    op.create_index(
        "ix_post_processing_settings_project_id",
        "post_processing_settings",
        ["project_id"],
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("input_video_path", sa.String(), nullable=False),
        sa.Column("output_video_path", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("inference_time_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_processing_jobs_project_id",
        "processing_jobs",
        ["project_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_project_id", table_name="processing_jobs")
    op.drop_table("processing_jobs")

    op.drop_index("ix_post_processing_settings_project_id", table_name="post_processing_settings")
    op.drop_table("post_processing_settings")
