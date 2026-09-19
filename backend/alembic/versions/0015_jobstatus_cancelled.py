"""Add 'cancelled' to jobstatus enum.

Revision ID: 0015_jobstatus_cancelled
Revises: 0014_post_processing_foundation
Create Date: 2026-05-18 00:00:00

Background
----------
TrainingJob.status (and Deployment.status) is backed by the 'jobstatus' PG
native enum on installations created via create_all().  The Python JobStatus
enum now includes 'cancelled' so the cancel endpoint can set a dedicated
status instead of overloading 'failed'.

On String-column installations the column is a VARCHAR and this migration is
a no-op.  The pg_type guard makes the migration idempotent on both install
types.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0015_jobstatus_cancelled"
down_revision = "0014_post_processing_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    enum_exists = bind.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'jobstatus'"
    )).scalar()

    if enum_exists:
        # ALTER TYPE … ADD VALUE must run outside a transaction on PG < 12.
        with op.get_context().autocommit_block():
            op.execute(sa.text(
                "ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'cancelled'"
            ))


def downgrade() -> None:
    # PostgreSQL does not support removing enum values.
    pass
