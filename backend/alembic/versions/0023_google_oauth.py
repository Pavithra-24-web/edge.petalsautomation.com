"""Add Google OAuth support to users

Revision ID: 0023_google_oauth
Revises: 0022_add_missing_fk_indexes
Create Date: 2026-07-01 00:00:00.000000

Adds users.google_sub (nullable, unique) to link a Google account to a
local user, and relaxes users.hashed_password to nullable since accounts
created via Google Sign-In have no local password.
"""

from alembic import op
import sqlalchemy as sa


revision = "0023_google_oauth"
down_revision = "0022_add_missing_fk_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_sub", sa.String(), nullable=True))
    op.create_index("ix_users_google_sub", "users", ["google_sub"], unique=True)
    op.alter_column("users", "hashed_password", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    op.alter_column("users", "hashed_password", existing_type=sa.String(), nullable=False)
    op.drop_index("ix_users_google_sub", table_name="users")
    op.drop_column("users", "google_sub")
