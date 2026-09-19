"""Add Versioning (C2, project-level) tables: project_versions,
project_version_impulses, project_version_samples

Revision ID: 0036_project_versions
Revises: 0035_impulse_versions
Create Date: 2026-09-08 00:00:00

See docs/Action/parityfix.md for the full design. This is the project-level
replacement for the impulse-scoped `impulse_versions`/`impulse_version_samples`
tables added in 0035 — parityfix.md states that design "is superseded in
full ... any scaffolding written against it is to be replaced, not extended."
0035's tables are left in place (their Phase-2 restore endpoint still reads
them); this migration only adds the three new tables, no changes to any
existing table:

- project_versions: the manifest header, one row per manual snapshot of a
  whole project (project config + deployment + post-processing settings deep
  copies, dataset totals, and a project-level accuracy/impulse-count rollup).
- project_version_impulses: one row per impulse that existed in the project
  at snapshot time (block/training config deep copy, active-model pointer,
  metrics copied at snapshot time).
- project_version_samples: the dataset manifest, one row per sample captured
  in a version. Denormalises sample_name/label_name/sample_type so the row
  survives a later delete or relabel of the live sample.

`project_version_impulses.impulse_id` and `project_version_samples.sample_id`
both use ON DELETE SET NULL (not CASCADE) — deleting a live impulse or
sample must never delete version history (parityfix.md §5.4).
`project_versions` rows themselves cascade-delete their `project_version_impulses`
and `project_version_samples` rows via each child table's `version_id` ON
DELETE CASCADE.

Follows the existing-table-guard pattern from 0034_ai_labeling_tables.py /
0035_impulse_versions.py so this migration is a safe no-op if these tables
somehow already exist.
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "0036_project_versions"
down_revision = "0035_impulse_versions"
branch_labels = None
depends_on = None


def _existing_tables(bind) -> set:
    return set(inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)

    if "project_versions" not in existing:
        op.create_table(
            "project_versions",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("project_config_snapshot", sa.JSON(), nullable=False),
            sa.Column("deployment_snapshot", sa.JSON(), nullable=False),
            sa.Column("post_processing_snapshot", sa.JSON(), nullable=True),
            sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("train_sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("test_sample_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("class_names", sa.JSON(), nullable=False),
            sa.Column("impulse_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("best_accuracy", sa.Float(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="draft"),
            sa.Column("published_at", sa.DateTime(), nullable=True),
            sa.Column("created_by", sa.String(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("project_id", "version_number", name="uq_project_versions_project_number"),
        )
        op.create_index(
            "ix_project_versions_project_created", "project_versions", ["project_id", "created_at"],
        )
        op.create_index("ix_project_versions_project_id", "project_versions", ["project_id"])

    if "project_version_impulses" not in existing:
        op.create_table(
            "project_version_impulses",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "version_id", sa.String(),
                sa.ForeignKey("project_versions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "impulse_id", sa.String(),
                sa.ForeignKey("impulses.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("impulse_name", sa.String(), nullable=False),
            sa.Column("dsp_blocks_snapshot", sa.JSON(), nullable=False),
            sa.Column("ml_blocks_snapshot", sa.JSON(), nullable=False),
            sa.Column("output_config_snapshot", sa.JSON(), nullable=False),
            sa.Column("input_config_snapshot", sa.JSON(), nullable=False),
            sa.Column("trained_model_id", sa.String(), sa.ForeignKey("trained_models.id"), nullable=True),
            sa.Column("training_job_id", sa.String(), sa.ForeignKey("training_jobs.id"), nullable=True),
            sa.Column("accuracy", sa.Float(), nullable=True),
            sa.Column("final_loss", sa.Float(), nullable=True),
            sa.Column("confusion_matrix", sa.JSON(), nullable=True),
            sa.Column("training_history", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_project_version_impulses_version", "project_version_impulses", ["version_id"],
        )
        op.create_index(
            "ix_project_version_impulses_impulse", "project_version_impulses", ["impulse_id"],
        )

    if "project_version_samples" not in existing:
        op.create_table(
            "project_version_samples",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "version_id", sa.String(),
                sa.ForeignKey("project_versions.id", ondelete="CASCADE"),
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
            "ix_project_version_samples_version", "project_version_samples", ["version_id"],
        )
        op.create_index(
            "ix_project_version_samples_sample", "project_version_samples", ["sample_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)
    if "project_version_samples" in existing:
        op.drop_table("project_version_samples")
    if "project_version_impulses" in existing:
        op.drop_table("project_version_impulses")
    if "project_versions" in existing:
        op.drop_table("project_versions")
