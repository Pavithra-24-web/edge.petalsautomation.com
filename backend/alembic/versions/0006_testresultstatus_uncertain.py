"""Add 'uncertain' to testresultstatus enum

Revision ID: 0006_testresultstatus_uncertain
Revises: 0005_unoq_deploy_target
Create Date: 2026-04-17 00:00:00

Background
----------
TestResultStatus.uncertain ("uncertain") was added to the Python enum to support
a "correct class but low confidence" bucket, but was never added to the native
PostgreSQL enum type when installations were initialized via create_all().

On String-column installations (created via Alembic migrations from 0003) the
result_status column is a VARCHAR, so no PG enum exists and this migration is a
no-op.  On native-enum installations (created via SQLAlchemy create_all()) the
column is typed as 'testresultstatus' and writing "uncertain" raises
  "invalid input value for enum testresultstatus: 'uncertain'"

The pg_type guard makes this migration idempotent on both install types.
"""

from alembic import op
import sqlalchemy as sa

revision      = "0006_testresultstatus_uncertain"
down_revision = "0005_unoq_deploy_target"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    bind = op.get_bind()
    # Only execute if the native PG enum type actually exists.
    enum_exists = bind.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'testresultstatus'"
    )).scalar()

    if enum_exists:
        # ALTER TYPE … ADD VALUE must run outside a transaction on PG < 12.
        # autocommit_block() commits the current Alembic transaction, runs the
        # statement in autocommit mode, then starts a new transaction.
        with op.get_context().autocommit_block():
            op.execute(sa.text(
                "ALTER TYPE testresultstatus ADD VALUE IF NOT EXISTS 'uncertain'"
            ))


def downgrade() -> None:
    # PostgreSQL does not support removing enum values.
    # To revert: drop and recreate the type excluding 'uncertain', migrating rows.
    # Not implemented — redeploy from 0005 on a clean DB if rollback is needed.
    pass
