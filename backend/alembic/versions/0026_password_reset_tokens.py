"""Add password reset token fields to users

Revision ID: 0021_password_reset_tokens
Revises: 0020_run_kind_active_ptr
Create Date: 2026-07-02 00:00:00.000000

Two nullable columns on users: reset_token_hash (SHA-256 of the raw URL-safe
token, unique) and reset_token_expires_at. Kept independent of the existing
verification-token fields so the two flows never interfere.
"""

from alembic import op
import sqlalchemy as sa


revision = '0026_password_reset_tokens'
down_revision = '0025_celery_task_id'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('reset_token_hash', sa.String(), nullable=True))
    op.add_column('users', sa.Column('reset_token_expires_at', sa.DateTime(), nullable=True))
    op.create_index('ix_users_reset_token_hash', 'users', ['reset_token_hash'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_users_reset_token_hash', table_name='users')
    op.drop_column('users', 'reset_token_expires_at')
    op.drop_column('users', 'reset_token_hash')
