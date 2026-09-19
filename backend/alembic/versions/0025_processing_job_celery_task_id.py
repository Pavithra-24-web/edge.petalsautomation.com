"""Add celery_task_id to processing_jobs

Revision ID: 0025_processing_job_celery_task_id
Revises: 0024_email_verification
Create Date: 2026-07-01 00:00:00.000000

Adds:
  processing_jobs.celery_task_id   String, nullable — the Celery AsyncResult
                                    id for the enqueued video-processing task,
                                    recorded so the cancel endpoint can issue
                                    a best-effort celery_app.control.revoke().
"""
from alembic import op
import sqlalchemy as sa


revision = "0025_celery_task_id"
down_revision = "0024_email_verification"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("processing_jobs", sa.Column("celery_task_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("processing_jobs", "celery_task_id")
