"""Add gap columns to model_test_runs and model_test_samples

Revision ID: 0004_model_testing_gaps
Revises: 0003_model_testing
Create Date: 2026-04-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision      = "0004_model_testing_gaps"
down_revision = "0003_model_testing"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    # ── model_test_runs: async job tracking ───────────────────────────────────
    op.add_column("model_test_runs",
        sa.Column("status", sa.String(), nullable=False, server_default="pending"))
    op.add_column("model_test_runs",
        sa.Column("started_at",   sa.DateTime(), nullable=True))
    op.add_column("model_test_runs",
        sa.Column("completed_at", sa.DateTime(), nullable=True))
    op.add_column("model_test_runs",
        sa.Column("error",        sa.String(),   nullable=True))

    # ── model_test_samples: scored_by FK + per-sample prediction columns ──────
    op.add_column("model_test_samples",
        sa.Column("scored_by_trained_model_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_mts_scored_by", "model_test_samples",
        "trained_models", ["scored_by_trained_model_id"], ["id"],
        ondelete="SET NULL",
    )
    op.add_column("model_test_samples",
        sa.Column("predicted_class",  sa.String(), nullable=True))
    op.add_column("model_test_samples",
        sa.Column("iou_score",        sa.Float(),  nullable=True))
    op.add_column("model_test_samples",
        sa.Column("predicted_boxes",  sa.JSON(),   nullable=True))


def downgrade() -> None:
    op.drop_column("model_test_samples", "predicted_boxes")
    op.drop_column("model_test_samples", "iou_score")
    op.drop_column("model_test_samples", "predicted_class")
    op.drop_constraint("fk_mts_scored_by", "model_test_samples", type_="foreignkey")
    op.drop_column("model_test_samples", "scored_by_trained_model_id")
    op.drop_column("model_test_runs", "error")
    op.drop_column("model_test_runs", "completed_at")
    op.drop_column("model_test_runs", "started_at")
    op.drop_column("model_test_runs", "status")
