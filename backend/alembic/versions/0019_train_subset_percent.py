"""Add train_subset_percent to impulses

Revision ID: 0019_train_subset_percent
Revises: 0018_impulse_dsp_params_saved_at
Create Date: 2026-06-11 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '0019_train_subset_percent'
down_revision = '0018_impulse_dsp_params_saved_at'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Percentage of training data to actually use during feature generation +
    # training. Default 100 = use everything. Existing rows are backfilled to
    # 100 via server_default so old impulses behave identically to before.
    op.add_column(
        'impulses',
        sa.Column(
            'train_subset_percent',
            sa.Float(),
            nullable=False,
            server_default='100',
        ),
    )


def downgrade() -> None:
    op.drop_column('impulses', 'train_subset_percent')
