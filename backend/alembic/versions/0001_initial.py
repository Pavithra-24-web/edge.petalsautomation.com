"""Initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2025-01-01 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users
    op.create_table("users",
        sa.Column("id",               sa.String(),  primary_key=True),
        sa.Column("email",            sa.String(),  nullable=False, unique=True),
        sa.Column("username",         sa.String(),  nullable=False, unique=True),
        sa.Column("hashed_password",  sa.String(),  nullable=False),
        sa.Column("role",             sa.String(),  default="developer"),
        sa.Column("is_active",        sa.Boolean(), default=True),
        sa.Column("created_at",       sa.DateTime()),
        sa.Column("updated_at",       sa.DateTime()),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # API Keys
    op.create_table("api_keys",
        sa.Column("id",           sa.String(), primary_key=True),
        sa.Column("user_id",      sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name",         sa.String(), nullable=False),
        sa.Column("key_hash",     sa.String(), nullable=False, unique=True),
        sa.Column("last_used",    sa.DateTime(), nullable=True),
        sa.Column("created_at",   sa.DateTime()),
        sa.Column("is_active",    sa.Boolean(), default=True),
    )

    # Projects
    op.create_table("projects",
        sa.Column("id",           sa.String(), primary_key=True),
        sa.Column("owner_id",     sa.String(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name",         sa.String(), nullable=False),
        sa.Column("description",  sa.Text(),   nullable=True),
        sa.Column("created_at",   sa.DateTime()),
        sa.Column("updated_at",   sa.DateTime()),
    )

    # Labels
    op.create_table("labels",
        sa.Column("id",           sa.String(), primary_key=True),
        sa.Column("project_id",   sa.String(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name",         sa.String(), nullable=False),
        sa.Column("color",        sa.String(), default="#3B8BD4"),
        sa.Column("created_at",   sa.DateTime()),
    )

    # Samples
    op.create_table("samples",
        sa.Column("id",                sa.String(),     primary_key=True),
        sa.Column("project_id",        sa.String(),     sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("label_id",          sa.String(),     sa.ForeignKey("labels.id"), nullable=True),
        sa.Column("filename",          sa.String(),     nullable=False),
        sa.Column("storage_key",       sa.String(),     nullable=False),
        sa.Column("sample_type",       sa.String(),     default="training"),
        sa.Column("sensor_type",       sa.String(),     nullable=True),
        sa.Column("frequency_hz",      sa.Float(),      nullable=True),
        sa.Column("duration_ms",       sa.Integer(),    nullable=True),
        sa.Column("num_channels",      sa.Integer(),    default=1),
        sa.Column("num_samples",       sa.Integer(),    nullable=True),
        sa.Column("file_size_bytes",   sa.BigInteger(), nullable=True),
        sa.Column("extra_metadata",    sa.JSON(),       default={}),
        sa.Column("created_at",        sa.DateTime()),
        sa.Column("uploaded_by",       sa.String(),     sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_samples_project_id", "samples", ["project_id"])
    op.create_index("ix_samples_label_id",   "samples", ["label_id"])

    # Impulses
    op.create_table("impulses",
        sa.Column("id",                 sa.String(),  primary_key=True),
        sa.Column("project_id",         sa.String(),  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name",               sa.String(),  nullable=False),
        sa.Column("description",        sa.Text(),    nullable=True),
        sa.Column("window_size_ms",     sa.Integer(), default=1000),
        sa.Column("window_increase_ms", sa.Integer(), default=500),
        sa.Column("frequency_hz",       sa.Float(),   default=100.0),
        sa.Column("zero_pad_allowed",   sa.Boolean(), default=True),
        sa.Column("dsp_blocks",         sa.JSON(),    default=[]),
        sa.Column("ml_blocks",          sa.JSON(),    default=[]),
        sa.Column("created_at",         sa.DateTime()),
        sa.Column("updated_at",         sa.DateTime()),
    )

    # Feature sets
    op.create_table("feature_sets",
        sa.Column("id",            sa.String(), primary_key=True),
        sa.Column("sample_id",     sa.String(), sa.ForeignKey("samples.id"), nullable=False),
        sa.Column("impulse_id",    sa.String(), sa.ForeignKey("impulses.id"), nullable=False),
        sa.Column("storage_key",   sa.String(), nullable=False),
        sa.Column("feature_shape", sa.JSON(),   nullable=True),
        sa.Column("created_at",    sa.DateTime()),
    )

    # Training jobs
    op.create_table("training_jobs",
        sa.Column("id",                    sa.String(),  primary_key=True),
        sa.Column("impulse_id",            sa.String(),  sa.ForeignKey("impulses.id"), nullable=False),
        sa.Column("celery_task_id",        sa.String(),  nullable=True),
        sa.Column("status",                sa.String(),  default="pending"),
        sa.Column("epochs",                sa.Integer(), default=100),
        sa.Column("learning_rate",         sa.Float(),   default=0.001),
        sa.Column("batch_size",            sa.Integer(), default=32),
        sa.Column("validation_split",      sa.Float(),   default=0.2),
        sa.Column("optimizer",             sa.String(),  default="adam"),
        sa.Column("extra_params",          sa.JSON(),    default={}),
        sa.Column("best_accuracy",         sa.Float(),   nullable=True),
        sa.Column("best_loss",             sa.Float(),   nullable=True),
        sa.Column("final_accuracy",        sa.Float(),   nullable=True),
        sa.Column("training_history",      sa.JSON(),    nullable=True),
        sa.Column("confusion_matrix",      sa.JSON(),    nullable=True),
        sa.Column("classification_report", sa.JSON(),    nullable=True),
        sa.Column("error_message",         sa.Text(),    nullable=True),
        sa.Column("started_at",            sa.DateTime(), nullable=True),
        sa.Column("completed_at",          sa.DateTime(), nullable=True),
        sa.Column("created_at",            sa.DateTime()),
    )

    # Trained models
    op.create_table("trained_models",
        sa.Column("id",               sa.String(),     primary_key=True),
        sa.Column("training_job_id",  sa.String(),     sa.ForeignKey("training_jobs.id"), nullable=False),
        sa.Column("version",          sa.String(),     nullable=False),
        sa.Column("format",           sa.String(),     nullable=False),
        sa.Column("storage_key",      sa.String(),     nullable=False),
        sa.Column("file_size_bytes",  sa.BigInteger(), nullable=True),
        sa.Column("model_metadata",   sa.JSON(),       default={}),
        sa.Column("created_at",       sa.DateTime()),
    )

    # Deployments
    op.create_table("deployments",
        sa.Column("id",               sa.String(),  primary_key=True),
        sa.Column("model_id",         sa.String(),  sa.ForeignKey("trained_models.id"), nullable=False),
        sa.Column("target",           sa.String(),  nullable=False),
        sa.Column("status",           sa.String(),  default="pending"),
        sa.Column("storage_key",      sa.String(),  nullable=True),
        sa.Column("download_url",     sa.String(),  nullable=True),
        sa.Column("celery_task_id",   sa.String(),  nullable=True),
        sa.Column("options",          sa.JSON(),    default={}),
        sa.Column("error_message",    sa.Text(),    nullable=True),
        sa.Column("created_at",       sa.DateTime()),
        sa.Column("completed_at",     sa.DateTime(), nullable=True),
    )

    # Devices
    op.create_table("devices",
        sa.Column("id",               sa.String(),  primary_key=True),
        sa.Column("project_id",       sa.String(),  sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name",             sa.String(),  nullable=False),
        sa.Column("device_type",      sa.String(),  nullable=True),
        sa.Column("firmware_version", sa.String(),  nullable=True),
        sa.Column("is_online",        sa.Boolean(), default=False),
        sa.Column("last_seen",        sa.DateTime(), nullable=True),
        sa.Column("ip_address",       sa.String(),  nullable=True),
        sa.Column("device_metadata",         sa.JSON(),    default={}),
        sa.Column("created_at",       sa.DateTime()),
    )


def downgrade() -> None:
    for table in ["devices", "deployments", "trained_models", "training_jobs",
                  "feature_sets", "impulses", "samples", "labels", "projects",
                  "api_keys", "users"]:
        op.drop_table(table)
