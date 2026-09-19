"""Add pxe value to deploytarget enum.

Revision ID: 0013_add_pxe_deploy_target
Revises: 0012_training_device_type
Create Date: 2026-05-02 00:00:00

Same class of issue as 0005_unoq_deploy_target.py: `deploytarget` only exists
as a native Postgres enum on installs where create_all() ran at some point.
On a database built via `alembic upgrade head` alone, this is a no-op — 'pxe'
is just a valid VARCHAR value there.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0013_add_pxe_deploy_target"
down_revision = "0012_training_device_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # pg_type / ALTER TYPE are Postgres-only — see 0005_unoq_deploy_target.py.
        return

    enum_exists = bind.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'deploytarget'"
    )).scalar()
    if enum_exists:
        with op.get_context().autocommit_block():
            op.execute(sa.text(
                "ALTER TYPE deploytarget ADD VALUE IF NOT EXISTS 'pxe'"
            ))


def downgrade() -> None:
    # Postgres does not support removing enum values; downgrade is a no-op.
    pass
