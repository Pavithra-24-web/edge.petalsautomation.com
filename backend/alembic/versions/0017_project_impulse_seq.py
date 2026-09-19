"""Add persistent per-project impulse sequence counter.

Revision ID: 0017_project_impulse_seq
Revises: 0016_pp_settings_impulse
Create Date: 2026-05-25 00:00:00

The counter advances on every default-named impulse creation. Deleting an
impulse never frees its number — the counter is monotonic. Backfill it from
the highest "Impulse N" suffix currently present on each project so existing
projects keep producing forward-only numbers.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0017_project_impulse_seq"
down_revision = "0016_pp_settings_impulse"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "impulse_seq",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    # Backfill: take the highest numeric suffix from any "Impulse N" name
    # currently on each project. Names that don't match the default pattern
    # contribute 0.
    op.execute(
        """
        UPDATE projects p
        SET impulse_seq = sub.max_n
        FROM (
            SELECT
                project_id,
                MAX(
                    CAST(
                        substring(name FROM '^\\s*[Ii]mpulse\\s+(\\d+)\\s*$')
                        AS INTEGER
                    )
                ) AS max_n
            FROM impulses
            WHERE name ~* '^\\s*impulse\\s+\\d+\\s*$'
            GROUP BY project_id
        ) sub
        WHERE p.id = sub.project_id
          AND sub.max_n IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.drop_column("projects", "impulse_seq")
