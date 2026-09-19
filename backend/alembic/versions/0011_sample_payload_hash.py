"""Add dedicated samples.payload_hash column and unique constraint.

Revision ID: 0011_sample_payload_hash
Revises: 0010_device_deployment_targeting
Create Date: 2026-04-28 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0011_sample_payload_hash"
down_revision = "0010_device_deployment_targeting"
branch_labels = None
depends_on = None


def _has_column(inspector, table: str, col: str) -> bool:
    return any(c["name"] == col for c in inspector.get_columns(table))


def _has_index(inspector, table: str, name: str) -> bool:
    return any(ix["name"] == name for ix in inspector.get_indexes(table))


def _has_unique(inspector, table: str, name: str) -> bool:
    return any(uq["name"] == name for uq in inspector.get_unique_constraints(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not _has_column(inspector, "samples", "payload_hash"):
        op.add_column("samples", sa.Column("payload_hash", sa.String(), nullable=True))

    samples = sa.table(
        "samples",
        sa.column("id", sa.String()),
        sa.column("project_id", sa.String()),
        sa.column("payload_hash", sa.String()),
        sa.column("extra_metadata", sa.JSON()),
    )

    rows = list(
        bind.execute(
            sa.select(
                samples.c.id,
                samples.c.project_id,
                samples.c.payload_hash,
                samples.c.extra_metadata,
            )
        )
    )

    seen: dict[tuple[str, str], str] = {}
    duplicate_ids: list[str] = []
    updates: list[dict[str, str]] = []

    for row in rows:
        payload_hash = row.payload_hash
        if not payload_hash and isinstance(row.extra_metadata, dict):
            raw = row.extra_metadata.get("payload_hash")
            if isinstance(raw, str) and raw.strip():
                payload_hash = raw.strip()

        if not payload_hash:
            continue

        key = (row.project_id, payload_hash)
        if key in seen:
            duplicate_ids.append(row.id)
            continue

        seen[key] = row.id
        if row.payload_hash != payload_hash:
            updates.append({"id": row.id, "payload_hash": payload_hash})

    for item in updates:
        bind.execute(
            sa.update(samples)
            .where(samples.c.id == item["id"])
            .values(payload_hash=item["payload_hash"])
        )

    if duplicate_ids:
        bind.execute(
            sa.update(samples)
            .where(samples.c.id.in_(duplicate_ids))
            .values(payload_hash=None)
        )

    inspector = inspect(bind)
    if not _has_index(inspector, "samples", "ix_samples_payload_hash"):
        op.create_index("ix_samples_payload_hash", "samples", ["payload_hash"])

    inspector = inspect(bind)
    if not _has_unique(inspector, "samples", "uq_samples_project_payload_hash"):
        op.create_unique_constraint(
            "uq_samples_project_payload_hash",
            "samples",
            ["project_id", "payload_hash"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if _has_unique(inspector, "samples", "uq_samples_project_payload_hash"):
        op.drop_constraint("uq_samples_project_payload_hash", "samples", type_="unique")

    inspector = inspect(bind)
    if _has_index(inspector, "samples", "ix_samples_payload_hash"):
        op.drop_index("ix_samples_payload_hash", table_name="samples")

    inspector = inspect(bind)
    if _has_column(inspector, "samples", "payload_hash"):
        op.drop_column("samples", "payload_hash")
