"""Remove old impulse-level Versioning tables: impulse_versions, impulse_version_samples

Revision ID: 0037_remove_impulse_versions
Revises: 0036_project_versions
Create Date: 2026-09-08 00:00:00

The impulse-scoped Versioning design (0035_impulse_versions) has been fully
superseded by the project-level design (0036_project_versions) and the
application no longer reads or writes these tables — the impulse-level
router, schemas, and models have all been removed. This migration drops the
two now-dead tables. Drops `impulse_version_samples` first since it holds an
FK to `impulse_versions`.

Guarded by an existing-table check so this migration is a safe no-op if the
tables are already gone (mirrors the guard pattern used by 0034/0035).
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "0037_remove_impulse_versions"
down_revision = "0036_project_versions"
branch_labels = None
depends_on = None


def _existing_tables(bind) -> set:
    return set(inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)

    if "impulse_version_samples" in existing:
        op.drop_table("impulse_version_samples")
    if "impulse_versions" in existing:
        op.drop_table("impulse_versions")


def downgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)

    if "impulse_versions" not in existing:
        op.create_table(
            "impulse_versions",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("impulse_id", sa.String(), sa.ForeignKey("impulses.id"), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("dsp_blocks_snapshot", sa.JSON(), nullable=False),
            sa.Column("ml_blocks_snapshot", sa.JSON(), nullable=False),
            sa.Column("output_config_snapshot", sa.JSON(), nullable=False),
            sa.Column("input_config_snapshot", sa.JSON(), nullable=False),
            sa.Column("post_processing_snapshot", sa.JSON(), nullable=True),
            sa.Column("trained_model_id", sa.String(), sa.ForeignKey("trained_models.id"), nullable=True),
            sa.Column("training_job_id", sa.String(), sa.ForeignKey("training_jobs.id"), nullable=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("final_loss", sa.Float(), nullable=True),
            sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("train_sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("test_sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("class_names", sa.JSON(), nullable=False),
            sa.Column("confusion_matrix", sa.JSON(), nullable=True),
            sa.Column("training_history", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="draft"),
            sa.Column("published_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("impulse_id", "version_number", name="uq_impulse_versions_impulse_number"),
        )
        op.create_index(
            "ix_impulse_versions_impulse_created", "impulse_versions", ["impulse_id", "created_at"],
        )
        op.create_index("ix_impulse_versions_project_id", "impulse_versions", ["project_id"])
        op.create_index("ix_impulse_versions_impulse_id", "impulse_versions", ["impulse_id"])

    existing = _existing_tables(bind)
    if "impulse_version_samples" not in existing:
        op.create_table(
            "impulse_version_samples",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "version_id", sa.String(),
                sa.ForeignKey("impulse_versions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "sample_id", sa.String(),
                sa.ForeignKey("samples.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("sample_name", sa.String(), nullable=False),
            sa.Column("label_name", sa.String(), nullable=True),
            sa.Column("sample_type", sa.String(), nullable=False),
        )
        op.create_index(
            "ix_impulse_version_samples_version", "impulse_version_samples", ["version_id"],
        )
        op.create_index(
            "ix_impulse_version_samples_sample", "impulse_version_samples", ["sample_id"],
        )
