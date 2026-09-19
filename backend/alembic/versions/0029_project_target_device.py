"""Project target device selection.

Adds `projects.target_device_slug` — the `DeviceCatalogEntry.slug` a project
targets, or NULL for "not chosen yet".

Part of Target Device Phase 3 (docs/target_device_phase3.md), brought forward
ahead of the rest of that phase so the topbar can surface the selection. The
application budget (RAM / ROM / latency / clock rate) is *not* here: every one
of those values comes from a device hardware specification, which is Phase 2
and does not exist yet.

No foreign key to `device_catalog_entries.slug` — see the column comment on
`Project.target_device_slug` in app/models/user.py. The catalog is migration-
owned seed data; a FK would make retiring a board either impossible or silently
destructive to a project's selection.

Revision ID: 0029_project_target_device
Revises: 0028_device_catalog
"""
from alembic import op
import sqlalchemy as sa


revision = "0029_project_target_device"
down_revision = "0028_device_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("target_device_slug", sa.String(), nullable=True),
    )
    # Every existing project starts with no selection, which is the documented
    # valid state — no backfill, and nothing is defaulted to a board the user
    # never picked.


def downgrade() -> None:
    op.drop_column("projects", "target_device_slug")
