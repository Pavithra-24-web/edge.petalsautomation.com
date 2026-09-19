"""Project target-device configuration overrides.

Target Device Phase 3 (docs/target_device_phase3.md), the remainder after
selection: adds the four override columns a project may set on top of its
selected board's `DeviceSpecification` —
`target_device_custom_name`, `target_device_ram_kb`, `target_device_rom_kb`,
`target_device_latency_ms`.

All four are nullable and NULL means "use the board's own specification
value" — never a snapshot of a resolved figure. A snapshot would go stale the
first time a specification is corrected, and it would make "the catalog entry
is not modified by an override" true only by accident. See the column
comment on `Project` in app/models/user.py.

These live on `projects` rather than a 1:1 table because `target_device_slug`
already does, and four nullable columns do not earn a join.

Revision ID: 0032_target_device_config
Revises: 0031_device_specifications

Named `0032_target_device_config` rather than `0032_project_target_device_config`
— the longer id is 33 characters, one over the `alembic_version.version_num`
column's `varchar(32)` limit (the table this Postgres uses is shared across
every migration in the project and was sized for the shorter ids elsewhere in
this history; it wasn't altered here since 32 is more than enough for a
readable slug).
"""
from alembic import op
import sqlalchemy as sa


revision = "0032_target_device_config"
down_revision = "0031_device_specifications"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("target_device_custom_name", sa.String(), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("target_device_ram_kb", sa.Integer(), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("target_device_rom_kb", sa.Integer(), nullable=True),
    )
    op.add_column(
        "projects",
        sa.Column("target_device_latency_ms", sa.Integer(), nullable=True),
    )
    # No backfill — every existing project starts with no overrides, which
    # resolves to its selected board's own values (or nothing, for a project
    # with no target device at all).


def downgrade() -> None:
    op.drop_column("projects", "target_device_latency_ms")
    op.drop_column("projects", "target_device_rom_kb")
    op.drop_column("projects", "target_device_ram_kb")
    op.drop_column("projects", "target_device_custom_name")
