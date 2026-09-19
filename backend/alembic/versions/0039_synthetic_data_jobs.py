"""Add synthetic_data_jobs table

Revision ID: 0039_synthetic_data_jobs
Revises: 0038_pv_training_config
Create Date: 2026-09-09 00:00:00

Backend for the Synthetic Data feature (Phase 2 of
docs/Action/datasynthetic_implementationplan.md). `SyntheticDataJob`
(app/models/user.py) is modelled directly on `AILabelingJob` — a
project-scoped background job with progress counters and an error message —
and this migration follows the same defensive style as
0034_ai_labeling_tables.py: an inspector existence check before
`create_table`, and `status`/`provider` declared as plain VARCHAR (not a
Postgres native enum type) to match every other enum-backed column created by
a migration in this schema.

No user-visible behaviour ships with this migration — the table exists but
nothing writes to it until Phase 3 adds the routes and worker.
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "0039_synthetic_data_jobs"
down_revision = "0038_pv_training_config"
branch_labels = None
depends_on = None


def _existing_tables(bind) -> set:
    return set(inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)

    if "synthetic_data_jobs" not in existing:
        op.create_table(
            "synthetic_data_jobs",
            sa.Column("id",                 sa.String(),  primary_key=True),
            sa.Column("project_id",         sa.String(),  sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("status",             sa.String(),  nullable=True, server_default="pending"),
            sa.Column("provider",           sa.String(),  nullable=False, server_default="openai"),
            sa.Column("model",              sa.String(),  nullable=False, server_default=""),
            sa.Column("prompt",             sa.Text(),    nullable=False, server_default=""),
            sa.Column("label_name",         sa.String(),  nullable=False, server_default=""),
            sa.Column("label_id",           sa.String(),  sa.ForeignKey("labels.id"), nullable=True),
            sa.Column("requested_count",    sa.Integer(), nullable=False, server_default="0"),
            sa.Column("generated_count",    sa.Integer(), nullable=True, server_default="0"),
            sa.Column("failed_count",       sa.Integer(), nullable=True, server_default="0"),
            sa.Column("sample_type",        sa.String(),  nullable=False, server_default="training"),
            sa.Column("parameters",         sa.JSON(),    nullable=True),
            sa.Column("sample_ids",         sa.JSON(),    nullable=True),
            sa.Column("estimated_cost_usd", sa.Float(),   nullable=True),
            sa.Column("error_message",      sa.Text(),    nullable=True),
            sa.Column("started_at",         sa.DateTime(), nullable=True),
            sa.Column("completed_at",       sa.DateTime(), nullable=True),
            sa.Column("created_at",         sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)
    if "synthetic_data_jobs" in existing:
        op.drop_table("synthetic_data_jobs")
