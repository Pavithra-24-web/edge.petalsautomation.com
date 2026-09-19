"""
S3-compatible object storage service (MinIO in dev, AWS S3 in prod)
"""
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import uuid
import io
import logging
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, BinaryIO
from urllib.parse import urlparse, urlunparse

from app.core.config import settings
from app.core.logging_config import get_logger, log_event, STORAGE_LOG_ALL

_storage_log = get_logger("storage")


def _bucketless_endpoint(endpoint: Optional[str], bucket: str) -> Optional[str]:
    """Strip the bucket from an endpoint, whether carried as host or path.

    We sign presigned URLs with ``addressing_style='path'``, which always
    prepends the bucket to the URL *path* (``/<bucket>/<key>``). If the
    configured endpoint already carries the bucket — either as a virtual-host
    subdomain (``https://petaledge.s3.us-east-1.amazonaws.com``) or as a leading
    path segment (``https://s3.us-east-1.amazonaws.com/petaledge``) — the bucket
    ends up twice (``/<bucket>/<bucket>/<key>``), so the browser GET 404s while
    the object actually lives at ``<key>``.

    Normalising to the bucketless form (``s3.us-east-1.amazonaws.com``) lets path
    style add the bucket exactly once, so the presigned URL points at the same
    object the upload client wrote. No-op for endpoints that carry no bucket
    (local MinIO, bucketless regional hosts).
    """
    if not endpoint:
        return endpoint
    parsed = urlparse(endpoint)
    # Drop the bucket if it leads the host as a virtual-host subdomain.
    host = parsed.hostname or ""
    prefix = f"{bucket}."
    if host.startswith(prefix):
        new_host = host[len(prefix):]
        netloc = f"{new_host}:{parsed.port}" if parsed.port else new_host
        parsed = parsed._replace(netloc=netloc)
    # Drop the bucket if it leads the path segment (e.g. ``.../petaledge``).
    path = parsed.path.rstrip("/")
    if path == f"/{bucket}" or path.startswith(f"/{bucket}/"):
        parsed = parsed._replace(path=path[len(f"/{bucket}"):])
    return urlunparse(parsed)


@contextmanager
def _timed_storage_op(op: str, key: str, **extra):
    """Time a storage operation and log its duration / outcome to storage.log.

    Pure instrumentation: on success the wrapped block's result is untouched; on
    failure the exception is logged then re-raised so callers behave as before.

    Failures are always logged. Successful ops are only logged when
    ``STORAGE_LOG_ALL`` is enabled — during bulk DSP each sample download would
    otherwise emit a (locked) write per op, which dominates the channel volume.
    """
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log_event(
            _storage_log, "storage.op", level=logging.ERROR,
            op=op, key=key, success=False, error=str(exc),
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
            **extra,
        )
        raise
    else:
        if STORAGE_LOG_ALL:
            log_event(
                _storage_log, "storage.op",
                op=op, key=key, success=True,
                duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
                **extra,
            )


class StorageService:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version="s3v4", s3={'addressing_style': 'path'}),
        )
        # Dedicated client bound to the *public* endpoint, used only to sign URLs
        # the browser will hit. Built once and reused so presigning is not a fresh
        # boto3 client construction on every call.
        self._public_client = boto3.client(
            "s3",
            endpoint_url=_bucketless_endpoint(
                settings.S3_PUBLIC_ENDPOINT, settings.S3_BUCKET
            ),
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
            config=Config(signature_version="s3v4", s3={'addressing_style': 'path'}),
        )
        self.bucket = settings.S3_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            
        cors_config = {
            'CORSRules': [{
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET'],
                'AllowedOrigins': ['*'],
                'ExposeHeaders': ['ETag']
            }]
        }
        try:
            self.client.put_bucket_cors(Bucket=self.bucket, CORSConfiguration=cors_config)
        except ClientError:
            pass

    def upload_file(
        self,
        file_obj: BinaryIO,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a file and return its storage key."""
        with _timed_storage_op("upload_file", key):
            self.client.upload_fileobj(
                file_obj,
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        return key

    def upload_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        with _timed_storage_op("upload_bytes", key, bytes=len(data)):
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        return key

    def download_bytes(self, key: str) -> bytes:
        with _timed_storage_op("download_bytes", key):
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            data = response["Body"].read()
        return data

    def _download_to_file(self, key: str, local_path: str) -> None:
        """Download an object to an arbitrary local filesystem path.

        NOT for use in Celery workers.  Workers must use download_bytes() and
        process data in memory or in a tempfile.TemporaryDirectory() — writing
        to a bare local path produces state that is invisible to every other
        worker and will silently break distributed training/deployment flows.

        This helper is kept for local tooling / one-off scripts only.
        """
        self.client.download_file(self.bucket, key, local_path)

    def get_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        download_filename: Optional[str] = None,
    ) -> str:
        # Reuse the public-endpoint client built in __init__ so URLs are signed
        # against the endpoint the browser uses, without rebuilding a client here.
        public_client = self._public_client
        params: dict = {"Bucket": self.bucket, "Key": key}
        if download_filename:
            params["ResponseContentDisposition"] = (
                f'attachment; filename="{download_filename}"'
            )
        return public_client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    def copy_object(self, src_key: str, dst_key: str) -> str:
        """Server-side copy of one object to a new key within the same bucket.

        Used by project cloning (``app.services.project_clone``): a cloned
        project must own its own copy of every asset, and a server-side copy
        keeps multi-GB datasets off the API process entirely — the bytes never
        leave the object store.
        """
        with _timed_storage_op("copy_object", dst_key, src=src_key):
            self.client.copy_object(
                Bucket=self.bucket,
                CopySource={"Bucket": self.bucket, "Key": src_key},
                Key=dst_key,
            )
        return dst_key

    def object_exists(self, key: str) -> bool:
        """HEAD probe — True when an object exists at ``key``."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete_file(self, key: str):
        with _timed_storage_op("delete_file", key):
            self.client.delete_object(Bucket=self.bucket, Key=key)

    def list_files(self, prefix: str) -> list:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    @staticmethod
    def sample_key(project_id: str, filename: str) -> str:
        return f"projects/{project_id}/samples/{uuid.uuid4()}/{filename}"

    @staticmethod
    def model_key(project_id: str, version: str, filename: str) -> str:
        return f"projects/{project_id}/models/{version}/{filename}"

    @staticmethod
    def video_input_key(project_id: str, job_id: str, filename: str) -> str:
        import os
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
        return f"projects/{project_id}/videos/{job_id}/input{ext}"

    @staticmethod
    def video_output_key(project_id: str, job_id: str) -> str:
        return f"projects/{project_id}/videos/{job_id}/output.mp4"

    @staticmethod
    def video_detections_key(project_id: str, job_id: str) -> str:
        return f"projects/{project_id}/videos/{job_id}/detections.json"


storage = StorageService()


@lru_cache(maxsize=32)
def download_bytes_cached(key: str) -> bytes:
    """Small process-wide LRU over *immutable* object bytes.

    Safe to cache because sample storage keys embed a per-upload UUID
    (``projects/<id>/samples/<uuid>/<file>``), so a given key always maps to the
    same content for its lifetime — re-uploading a sample produces a new key.

    Use ONLY for hot read paths that re-fetch the same immutable object
    repeatedly (e.g. DSP preview / feature explorer while a user tweaks block
    params on one sample). Do NOT use for mutable keys such as the per-impulse
    ``features.npz`` cache, whose content changes in place when features are
    regenerated.
    """
    return storage.download_bytes(key)
