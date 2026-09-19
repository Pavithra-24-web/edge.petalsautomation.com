"""Add AI labeling tables: ai_labeling_actions, ai_labeling_jobs, ai_predictions

Revision ID: 0034_ai_labeling_tables
Revises: 0033_project_type
Create Date: 2026-09-06 00:00:00

Background
----------
AILabelingAction / AILabelingJob / AIPrediction (app/models/user.py) have
never had a migration — on every environment tested so far these three
tables only ever existed because create_all() created them from the ORM
models at startup. On a database built from `alembic upgrade head` alone,
they are missing entirely. This migration creates them to match the current
ORM models.

`ai_labeling_jobs.status` is declared as a native SAEnum(AILabelingStatus) on
the model, but — matching every other enum-backed column created by a
migration in this schema (users.role, samples.sample_type,
training_jobs.status, etc., see 0001_initial) — it is created here as a plain
VARCHAR rather than a Postgres native enum type. SQLAlchemy's Enum type reads
and writes identically against either column kind.

Guarded with an inspector existence check (same pattern as
0022_add_missing_fk_indexes.py): on an install where create_all() already
created these three tables, this migration is a no-op for that table and
just records the new alembic_version — it does not touch existing rows.
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "0034_ai_labeling_tables"
down_revision = "0033_project_type"
branch_labels = None
depends_on = None


def _existing_tables(bind) -> set:
    return set(inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)

    if "ai_labeling_actions" not in existing:
        op.create_table(
            "ai_labeling_actions",
            sa.Column("id",              sa.String(),  primary_key=True),
            sa.Column("project_id",      sa.String(),  sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("name",            sa.String(),  nullable=False),
            sa.Column("model_type",      sa.String(),  nullable=False, server_default="classification"),
            sa.Column("prompt",          sa.Text(),    nullable=True),
            sa.Column("label_names",     sa.Text(),    nullable=True),
            sa.Column("provider",        sa.String(),  nullable=False, server_default="openai"),
            sa.Column("model_config_json", sa.JSON(),  nullable=True),
            sa.Column("created_at",      sa.DateTime(), nullable=True),
            sa.Column("updated_at",      sa.DateTime(), nullable=True),
        )

    if "ai_labeling_jobs" not in existing:
        op.create_table(
            "ai_labeling_jobs",
            sa.Column("id",             sa.String(),  primary_key=True),
            sa.Column("action_id",      sa.String(),  sa.ForeignKey("ai_labeling_actions.id"), nullable=False),
            sa.Column("project_id",     sa.String(),  sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("status",         sa.String(),  nullable=True, server_default="running"),
            sa.Column("total_samples",  sa.Integer(), nullable=True, server_default="0"),
            sa.Column("processed",      sa.Integer(), nullable=True, server_default="0"),
            sa.Column("failed_count",   sa.Integer(), nullable=True, server_default="0"),
            sa.Column("error_message",  sa.Text(),    nullable=True),
            sa.Column("filter_options", sa.JSON(),    nullable=True),
            sa.Column("started_at",     sa.DateTime(), nullable=True),
            sa.Column("completed_at",   sa.DateTime(), nullable=True),
            sa.Column("created_at",     sa.DateTime(), nullable=True),
        )

    if "ai_predictions" not in existing:
        op.create_table(
            "ai_predictions",
            sa.Column("id",              sa.String(),  primary_key=True),
            sa.Column("job_id",          sa.String(),  sa.ForeignKey("ai_labeling_jobs.id"), nullable=False),
            sa.Column("sample_id",       sa.String(),  sa.ForeignKey("samples.id"), nullable=False),
            sa.Column("predicted_label", sa.String(),  nullable=True),
            sa.Column("query_fragment",  sa.String(),  nullable=True),
            sa.Column("confidence",      sa.Float(),   nullable=True),
            sa.Column("bounding_boxes",  sa.JSON(),    nullable=True),
            sa.Column("status",          sa.String(),  nullable=True, server_default="pending"),
            sa.Column("created_at",      sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = _existing_tables(bind)
    if "ai_predictions" in existing:
        op.drop_table("ai_predictions")
    if "ai_labeling_jobs" in existing:
        op.drop_table("ai_labeling_jobs")
    if "ai_labeling_actions" in existing:
        op.drop_table("ai_labeling_actions")
