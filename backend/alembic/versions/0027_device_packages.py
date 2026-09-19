"""Device-client packages — versioned release index

Revision ID: 0027_device_packages
Revises: 0026_password_reset_tokens
Create Date: 2026-07-18 00:00:00.000000

Creates the `device_packages` table (contract §14): one row per published
semver version of the device-client tarball. Columns are exactly the manifest
fields (§14.4); `version` is the primary key (versions are immutable, §14.3).
The sha256 lives only here — no `.sha256` sidecar in storage (§14.1/§14.6).
Additive; does not touch existing tables.
"""

from alembic import op
import sqlalchemy as sa


revision = "0027_device_packages"
down_revision = "0026_password_reset_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "device_packages",
        sa.Column("version",    sa.String(), primary_key=True),
        sa.Column("filename",   sa.String(), nullable=False),
        sa.Column("size",       sa.BigInteger(), nullable=False),
        sa.Column("sha256",     sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("device_packages")
