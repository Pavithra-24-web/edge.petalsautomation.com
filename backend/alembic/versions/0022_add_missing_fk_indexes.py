"""Add missing foreign-key indexes

Revision ID: 0022_add_missing_fk_indexes
Revises: 0021_dsp_feature_jobs
Create Date: 2026-06-23 00:00:00.000000

Adds indexes on foreign-key columns that are filtered/joined on hot paths
but were never indexed by an earlier migration. See
DATABASE_PERFORMANCE_REPORT.md / DATABASE_INDEX_MIGRATION.md for the full
rationale and expected impact of each index.

Indexes added:
  training_jobs.impulse_id          ix_training_jobs_impulse_id
  labels.project_id                 ix_labels_project_id
  feature_sets.sample_id            ix_feature_sets_sample_id
  feature_sets.impulse_id           ix_feature_sets_impulse_id
  trained_models.training_job_id    ix_trained_models_training_job_id
  deployments.model_id              ix_deployments_model_id
  deployments.project_id            ix_deployments_project_id
  impulses.project_id               ix_impulses_project_id
  projects.owner_id                 ix_projects_owner_id

Index-only migration: no table, column, query, or behavioural change.

Idempotent: each index is created only if it is not already present.
Some of these columns are declared ``index=True`` on the ORM model
(e.g. deployments.project_id) but were never created by a migration — and
an environment built via ``Base.metadata.create_all`` may already have a
subset of them. The inspector guard makes the migration safe to apply in
either state.
"""

from alembic import op
from sqlalchemy import inspect


revision = "0022_add_missing_fk_indexes"
down_revision = "0021_dsp_feature_jobs"
branch_labels = None
depends_on = None


# (index_name, table_name, [columns])
_INDEXES = [
    ("ix_training_jobs_impulse_id",       "training_jobs",  ["impulse_id"]),
    ("ix_labels_project_id",              "labels",         ["project_id"]),
    ("ix_feature_sets_sample_id",         "feature_sets",   ["sample_id"]),
    ("ix_feature_sets_impulse_id",        "feature_sets",   ["impulse_id"]),
    ("ix_trained_models_training_job_id", "trained_models", ["training_job_id"]),
    ("ix_deployments_model_id",           "deployments",    ["model_id"]),
    ("ix_deployments_project_id",         "deployments",    ["project_id"]),
    ("ix_impulses_project_id",            "impulses",       ["project_id"]),
    ("ix_projects_owner_id",              "projects",       ["owner_id"]),
]


def _existing_index_names(inspector, table: str) -> set:
    try:
        return {ix["name"] for ix in inspector.get_indexes(table)}
    except Exception:
        # Table missing entirely (shouldn't happen for these core tables) —
        # treat as "no indexes" so create_index surfaces the real error.
        return set()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    for index_name, table, columns in _INDEXES:
        if index_name not in _existing_index_names(inspector, table):
            op.create_index(index_name, table, columns)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    for index_name, table, _columns in reversed(_INDEXES):
        if index_name in _existing_index_names(inspector, table):
            op.drop_index(index_name, table_name=table)
