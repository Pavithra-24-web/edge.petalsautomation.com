"""Add email verification to users

Revision ID: 0024_email_verification
Revises: 0023_google_oauth
Create Date: 2026-07-01 00:00:00.000000

Adds email verification for password-based accounts. Google Sign-In
accounts are created with is_verified=True (Google's email_verified claim
already covers this) and are unaffected.

Adds:
  users.is_verified                     Boolean, default False
  users.verification_token_hash         String, nullable, unique — SHA-256
                                         hash of the token emailed to the
                                         user; the raw token is never stored
  users.verification_token_expires_at   DateTime, nullable
"""
from alembic import op
import sqlalchemy as sa


revision = "0024_email_verification"
down_revision = "0023_google_oauth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("users", sa.Column("verification_token_hash", sa.String(), nullable=True))
    op.add_column("users", sa.Column("verification_token_expires_at", sa.DateTime(), nullable=True))
    op.create_index("ix_users_verification_token_hash", "users", ["verification_token_hash"], unique=True)
    op.alter_column("users", "is_verified", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_users_verification_token_hash", table_name="users")
    op.drop_column("users", "verification_token_expires_at")
    op.drop_column("users", "verification_token_hash")
    op.drop_column("users", "is_verified")
