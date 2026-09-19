"""Add training_config_snapshot, trained_model_formats_snapshot and
features_storage_key to project_version_impulses.

Revision ID: 0038_pv_training_config
Revises: 0037_remove_impulse_versions
Create Date: 2026-09-08 12:00:00

See docs/Action/parityfix.md §2.2 / §4.3. A Store Version snapshot already
copies an impulse's block config (dsp_blocks_snapshot / ml_blocks_snapshot)
and its training *results* (accuracy / final_loss / confusion_matrix /
training_history), but never its training *configuration* — the
hyperparameters (epochs, learning_rate, batch_size, ...) that produced those
results — nor a self-contained record of every TrainedModel format the
training job produced, nor a pointer to the DSP feature cache
(`features.npz`) generated for it. Restore therefore had to reach into the
*live* TrainingJob/TrainedModel rows via `training_job_id`/`trained_model_id`
to reconstruct a working impulse, which is fragile (those rows can be
deleted between store and restore) and incomplete (Feature Explorer data and
"already generated" state were never restored at all — the user always had
to click "Generate Features" again).

These three columns make the snapshot self-contained for that data, the same
way `accuracy`/`confusion_matrix`/etc. already are — copied at snapshot time,
not just pointed to:

- training_config_snapshot: the hyperparameters (epochs, learning_rate,
  batch_size, validation_split, optimizer, device_type, extra_params,
  run_kind, status, started_at, completed_at).
- trained_model_formats_snapshot: every TrainedModel row's
  {format, version, storage_key, file_size_bytes, model_metadata} the
  training job produced — not just the one `trained_model_id` already
  pointed to (typically the tflite build), so deployment/evaluation can find
  every format a restored impulse would otherwise be missing.
- features_storage_key: the S3 key of the impulse's `features.npz` cache at
  snapshot time, if one existed. Restore downloads it, remaps every
  embedded label id / sample id to the new project's own ids (the same class
  of "soft FK" problem already fixed for `Sample.extra_metadata.
  boundingBoxes`), and re-uploads it under the new impulse's own key — so a
  restored, previously-trained impulse shows its Feature Explorer data
  immediately and can retrain without regenerating.

Nullable/no server_default: existing rows (impulses that were untrained at
snapshot time, or versions stored before this migration) simply have NULL
here, which restore already treats as "nothing to clone."
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "0038_pv_training_config"
down_revision = "0037_remove_impulse_versions"
branch_labels = None
depends_on = None


def _existing_columns(bind, table: str) -> set:
    return {c["name"] for c in inspect(bind).get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    existing = _existing_columns(bind, "project_version_impulses")

    if "training_config_snapshot" not in existing:
        op.add_column(
            "project_version_impulses",
            sa.Column("training_config_snapshot", sa.JSON(), nullable=True),
        )
    if "trained_model_formats_snapshot" not in existing:
        op.add_column(
            "project_version_impulses",
            sa.Column("trained_model_formats_snapshot", sa.JSON(), nullable=True),
        )
    if "features_storage_key" not in existing:
        op.add_column(
            "project_version_impulses",
            sa.Column("features_storage_key", sa.String(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = _existing_columns(bind, "project_version_impulses")

    if "features_storage_key" in existing:
        op.drop_column("project_version_impulses", "features_storage_key")
    if "trained_model_formats_snapshot" in existing:
        op.drop_column("project_version_impulses", "trained_model_formats_snapshot")
    if "training_config_snapshot" in existing:
        op.drop_column("project_version_impulses", "training_config_snapshot")
