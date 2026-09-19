"""Model testing tables: model_versions, model_test_runs, metric_results, model_test_samples

Revision ID: 0003_model_testing
Revises: 0002_impulse_input_type
Create Date: 2026-04-02 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision     = "0003_model_testing"
down_revision = "0002_impulse_input_type"
branch_labels = None
depends_on    = None


def upgrade() -> None:

    # ── model_versions ────────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id",               sa.String(), primary_key=True),
        sa.Column("project_id",       sa.String(), sa.ForeignKey("projects.id"),       nullable=False),
        sa.Column("impulse_id",       sa.String(), sa.ForeignKey("impulses.id"),       nullable=False),
        sa.Column("trained_model_id", sa.String(), sa.ForeignKey("trained_models.id"), nullable=True),
        sa.Column("version_number",   sa.String(), nullable=False),
        sa.Column("name",             sa.String(), nullable=False),
        sa.Column("quantization_type",sa.String(), nullable=False, server_default="int8"),
        sa.Column("is_active",        sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at",       sa.DateTime(), nullable=True),
    )
    op.create_index("ix_model_versions_impulse_id", "model_versions", ["impulse_id"])

    # ── model_test_runs ───────────────────────────────────────────────────────
    op.create_table(
        "model_test_runs",
        sa.Column("id",               sa.String(), primary_key=True),
        sa.Column("project_id",       sa.String(), sa.ForeignKey("projects.id"),       nullable=False),
        sa.Column("impulse_id",       sa.String(), sa.ForeignKey("impulses.id"),       nullable=False),
        sa.Column("model_version_id", sa.String(), sa.ForeignKey("model_versions.id"), nullable=True),
        sa.Column("accuracy",         sa.Float(),   nullable=True),
        sa.Column("total_samples",    sa.Integer(), nullable=False, server_default="0"),
        sa.Column("passed_samples",   sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_samples",   sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at",       sa.DateTime(), nullable=True),
    )
    op.create_index("ix_model_test_runs_impulse_id", "model_test_runs", ["impulse_id"])

    # ── metric_results ────────────────────────────────────────────────────────
    op.create_table(
        "metric_results",
        sa.Column("id",                  sa.String(), primary_key=True),
        sa.Column("test_run_id",         sa.String(), sa.ForeignKey("model_test_runs.id"), nullable=False),
        sa.Column("metric_name",         sa.String(), nullable=False),
        sa.Column("metric_display_name", sa.String(), nullable=False),
        sa.Column("metric_value",        sa.Float(),  nullable=True),
        sa.Column("created_at",          sa.DateTime(), nullable=True),
    )

    # ── model_test_samples ────────────────────────────────────────────────────
    op.create_table(
        "model_test_samples",
        sa.Column("id",               sa.String(), primary_key=True),
        sa.Column("project_id",       sa.String(), sa.ForeignKey("projects.id"),        nullable=False),
        sa.Column("impulse_id",       sa.String(), sa.ForeignKey("impulses.id"),        nullable=False),
        sa.Column("sample_id",        sa.String(), sa.ForeignKey("samples.id"),         nullable=True),
        sa.Column("test_run_id",      sa.String(), sa.ForeignKey("model_test_runs.id"), nullable=True),
        sa.Column("sample_name",      sa.String(), nullable=False),
        sa.Column("expected_outcome", sa.String(), nullable=True),
        sa.Column("f1_score",         sa.Float(),  nullable=True),
        sa.Column("result_status",    sa.String(), nullable=False, server_default="pending"),
        sa.Column("created_at",       sa.DateTime(), nullable=True),
        sa.Column("updated_at",       sa.DateTime(), nullable=True),
    )
    op.create_index("ix_model_test_samples_impulse_id", "model_test_samples", ["impulse_id"])


def downgrade() -> None:
    op.drop_table("model_test_samples")
    op.drop_table("metric_results")
    op.drop_table("model_test_runs")
    op.drop_table("model_versions")
