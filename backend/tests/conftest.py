"""
Pytest configuration and shared fixtures.

IMPORTANT: env vars and storage mocks must be set before any app module is
imported.  Python's import cache means that once app.core.storage is loaded,
StorageService() has already been instantiated.  The boto3 mock here prevents
that instantiation from trying to connect to a real MinIO/S3 endpoint.
"""
import os
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError as _BotoClientError

# ─── Environment ─────────────────────────────────────────────────────────────

os.environ.setdefault("POSTGRES_HOST",         "localhost")
os.environ.setdefault("POSTGRES_DB",           "edgeimpulse_test")
os.environ.setdefault("POSTGRES_USER",         "test")
os.environ.setdefault("POSTGRES_PASSWORD",     "test")
os.environ.setdefault("SECRET_KEY",            "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET",            "test-jwt-secret-for-testing-only")
os.environ.setdefault("S3_ENDPOINT",           "http://localhost:9000")
os.environ.setdefault("S3_ACCESS_KEY",         "test")
os.environ.setdefault("S3_SECRET_KEY",         "test")
os.environ.setdefault("S3_BUCKET",             "test-bucket")
os.environ.setdefault("S3_REGION",             "us-east-1")
os.environ.setdefault("REDIS_URL",             "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL",     "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# ─── Storage mock (prevents MinIO connection at import time) ─────────────────
#
# StorageService() calls boto3.client(…) then _ensure_bucket() inside __init__.
# Without this mock every test collection fails with EndpointConnectionError.
# The mock makes head_bucket raise ClientError (the "bucket missing" path) so
# _ensure_bucket falls through to create_bucket, which the mock accepts silently.

_mock_s3_client = MagicMock()
_mock_s3_client.head_bucket.side_effect = _BotoClientError(
    {"Error": {"Code": "NoSuchBucket", "Message": "mocked"}}, "HeadBucket"
)
_mock_s3_client.create_bucket.return_value = {}
_mock_s3_client.put_bucket_cors.return_value = {}
# Provide sensible defaults for any test that calls storage methods directly
_mock_s3_client.put_object.return_value = {}
_mock_s3_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"")}
_mock_s3_client.delete_object.return_value = {}
_mock_s3_client.list_objects_v2.return_value = {"Contents": []}
_mock_s3_client.generate_presigned_url.return_value = "http://localhost:9000/test/presigned"

# Start the patcher before any app module is imported.  Never stopped so it
# covers the entire test session (test_api.py and test_ingestion.py alike).
_boto3_client_patcher = patch("boto3.client", return_value=_mock_s3_client)
_boto3_client_patcher.start()

# ─── Shared fixtures ──────────────────────────────────────────────────────────

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_TEST_DB_URL = "sqlite:///./test_edgeimpulse_conftest.db"
_engine = create_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
_Session = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(scope="session", autouse=True)
def _create_conftest_tables():
    from app.core.database import Base
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def db_session(_create_conftest_tables):
    """Provide a rolled-back DB session for unit tests that need direct ORM access."""
    connection = _engine.connect()
    transaction = connection.begin()
    session = _Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()
