"""Add TrainingJob.run_kind and Impulse.active_model_run_id

Revision ID: 0020_run_kind_active_ptr
Revises: 0019_train_subset_percent
Create Date: 2026-06-13 00:00:00.000000

Splits fresh-training vs retrain at the schema level so the frontend gate
can derive results-visibility purely from the latest run's `run_kind`
+ `status` and the impulse's `active_model_run_id` pointer. The pointer is
the single source of truth for "which training run's artifacts are active";
it is set only on successful completion and never cleared on cancel/fail/
new-run-start, preserving recoverability.
"""

from alembic import op
import sqlalchemy as sa


revision = '0020_run_kind_active_ptr'
down_revision = '0019_train_subset_percent'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── TrainingJob.run_kind ────────────────────────────────────────────────
    # Backfilled to "fresh" via server_default so historical jobs (which
    # predate the column) behave like the old default. The retrain endpoint
    # writes "retrain" on new rows going forward.
    op.add_column(
        'training_jobs',
        sa.Column(
            'run_kind',
            sa.String(),
            nullable=False,
            server_default='fresh',
        ),
    )

    # ── Impulse.active_model_run_id ─────────────────────────────────────────
    # Nullable pointer to the run whose artifacts are "currently active".
    # No explicit FK constraint: the existing impulse → training_jobs cascade
    # is "all, delete-orphan", which would form a self-referential cycle with
    # an FK back to training_jobs.id. Application-level invariants keep the
    # pointer honest; orphaned values are harmless (UI just treats them as
    # "no active model").
    op.add_column(
        'impulses',
        sa.Column('active_model_run_id', sa.String(), nullable=True),
    )

    # ── Backfill ────────────────────────────────────────────────────────────
    # For every impulse, point active_model_run_id at the most recent
    # successfully-completed training run so existing trained projects keep
    # showing their results immediately after deploy.
    op.execute(
        """
        UPDATE impulses
           SET active_model_run_id = (
                SELECT tj.id
                  FROM training_jobs tj
                 WHERE tj.impulse_id = impulses.id
                   AND tj.status = 'completed'
                 ORDER BY tj.created_at DESC
                 LIMIT 1
           )
         WHERE active_model_run_id IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column('impulses', 'active_model_run_id')
    op.drop_column('training_jobs', 'run_kind')
