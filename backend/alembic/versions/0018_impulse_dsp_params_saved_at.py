"""Add dsp_params_saved_at to impulses

Revision ID: 0018_impulse_dsp_params_saved_at
Revises: 0017_project_impulse_seq
Create Date: 2026-06-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '0018_impulse_dsp_params_saved_at'
down_revision = '0017_project_impulse_seq'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'impulses',
        sa.Column('dsp_params_saved_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('impulses', 'dsp_params_saved_at')
