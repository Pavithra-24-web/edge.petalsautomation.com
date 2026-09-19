"""Add unoq to deploytarget enum

Revision ID: 0005_unoq_deploy_target
Revises: 0004_model_testing_gaps
Create Date: 2026-04-16 00:00:00

Background
----------
The `deployments.target` column was created by SQLAlchemy's create_all() using
SAEnum(DeployTarget), which produced a native Postgres enum type `deploytarget`.
On a database built via `alembic upgrade head` alone (no create_all()), the
`deployments.target` column created by 0001_initial is a plain VARCHAR and the
`deploytarget` type never exists — see the pg_type guard below, matching the
same pattern 0015_jobstatus_cancelled.py uses for `jobstatus`.

ALTER TYPE … ADD VALUE cannot run inside an open transaction on PostgreSQL < 12.
Alembic 1.8+ provides autocommit_block() to temporarily exit the transaction for
exactly this purpose.  The IF NOT EXISTS guard makes the statement idempotent.
"""

from alembic import op
import sqlalchemy as sa

revision      = "0005_unoq_deploy_target"
down_revision = "0004_model_testing_gaps"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # pg_type / ALTER TYPE are Postgres-only — this migration is exercised
        # directly (not just via `alembic upgrade`) by tests that bind it to a
        # SQLite engine. Nothing to do there: no native enum type exists on
        # any other dialect.
        return

    enum_exists = bind.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'deploytarget'"
    )).scalar()

    if enum_exists:
        # autocommit_block() exits the Alembic transaction for this block, then
        # re-enters it — required for ALTER TYPE … ADD VALUE on PG < 12, harmless
        # on PG 12+.  IF NOT EXISTS makes this safe to re-run.
        with op.get_context().autocommit_block():
            op.execute(sa.text(
                "ALTER TYPE deploytarget ADD VALUE IF NOT EXISTS 'unoq'"
            ))
    # On String-column installations (no create_all() ever ran) this is a
    # no-op: 'unoq' is just a valid VARCHAR value, no type to alter.


def downgrade() -> None:
    # PostgreSQL does not support removing enum values.
    # To revert: drop and recreate the type excluding 'unoq', migrating all rows.
    # Not implemented — redeploy from 0004 on a clean DB if rollback is needed.
    pass
