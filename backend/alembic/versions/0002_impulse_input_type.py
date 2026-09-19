"""Add impulse new columns

Revision ID: 0002_impulse_input_type
Revises: 0001_initial
Create Date: 2026-03-23 12:45:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = '0002_impulse_input_type'
down_revision = '0001_initial'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('impulses', sa.Column('input_type', sa.String(), server_default='time-series', nullable=True))
    op.add_column('impulses', sa.Column('input_axes', sa.JSON(), server_default='[]', nullable=True))
    op.add_column('impulses', sa.Column('sensor_type', sa.String(), nullable=True))
    op.add_column('impulses', sa.Column('image_width', sa.Integer(), server_default='96', nullable=True))
    op.add_column('impulses', sa.Column('image_height', sa.Integer(), server_default='96', nullable=True))
    op.add_column('impulses', sa.Column('resize_mode', sa.String(), server_default='Fit shortest axis', nullable=True))
    op.add_column('impulses', sa.Column('output_config', sa.JSON(), server_default='{}', nullable=True))

def downgrade() -> None:
    op.drop_column('impulses', 'output_config')
    op.drop_column('impulses', 'resize_mode')
    op.drop_column('impulses', 'image_height')
    op.drop_column('impulses', 'image_width')
    op.drop_column('impulses', 'sensor_type')
    op.drop_column('impulses', 'input_axes')
    op.drop_column('impulses', 'input_type')
