"""Add dsp_feature_jobs table

Revision ID: 0021_dsp_feature_jobs
Revises: 0020_run_kind_active_ptr
Create Date: 2026-06-15 00:00:00.000000

Tracks each DSP feature-generation run so the /jobs union endpoint can
surface them alongside training, deployment, model-test and ai-labeling
rows. Schema mirrors training_jobs at the column level (id, impulse_id,
celery_task_id, status, error_message, started_at/completed_at/created_at)
plus a denormalised project_id so the union query doesn't need a JOIN
through impulses, and an input_type snapshot so the jobs-page label stays
accurate even if the impulse's input_type is later edited.

No backfill: impulses whose features.npz blobs predate this migration
won't show a row in /jobs until the next regeneration. Synthesising
historical rows from blob timestamps would be misleading because we have
no real status, started_at, or duration to assign.
"""

from alembic import op
import sqlalchemy as sa


revision = "0021_dsp_feature_jobs"
down_revision = "0020_run_kind_active_ptr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dsp_feature_jobs",
        sa.Column("id",              sa.String(),  primary_key=True),
        sa.Column("impulse_id",      sa.String(),  sa.ForeignKey("impulses.id"), nullable=False),
        sa.Column("project_id",      sa.String(),  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("celery_task_id",  sa.String(),  nullable=True),
        sa.Column("status",          sa.String(),  nullable=False, server_default="pending"),
        sa.Column("input_type",      sa.String(),  nullable=True),
        sa.Column("error_message",   sa.Text(),    nullable=True),
        sa.Column("started_at",      sa.DateTime(), nullable=True),
        sa.Column("completed_at",    sa.DateTime(), nullable=True),
        sa.Column("created_at",      sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_dsp_feature_jobs_impulse_id",
        "dsp_feature_jobs", ["impulse_id"],
    )
    op.create_index(
        "ix_dsp_feature_jobs_project_id",
        "dsp_feature_jobs", ["project_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_dsp_feature_jobs_project_id", table_name="dsp_feature_jobs")
    op.drop_index("ix_dsp_feature_jobs_impulse_id", table_name="dsp_feature_jobs")
    op.drop_table("dsp_feature_jobs")
