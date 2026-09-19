"""
Application configuration — reads from environment / .env

Credentials (signing keys, DB password, object-storage keys) have no
in-code default: pydantic-settings requires them from the real environment
or the .env file referenced below and raises a ValidationError at startup
if any are missing. Never give these fields a working fallback value here.
"""
from pydantic import model_validator
from pydantic_settings import BaseSettings
from typing import List


# Known dev/test placeholder values — rejected outright when APP_ENV=production
# so a copied-over .env.dev (or similarly weak value) can't reach prod silently.
_PLACEHOLDER_SECRETS = {
    "", "changeme", "change-me", "changeme-jwt-secret", "changeme-secret-key",
    "test", "1234", "password", "postgres", "minioadmin",
}


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Auth signing keys — required, no default (see module docstring)
    SECRET_KEY: str
    JWT_SECRET: str

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5439
    POSTGRES_DB: str = "PetalEdge"
    POSTGRES_USER: str = "PetalEdge"
    POSTGRES_PASSWORD: str  # required, no default (see module docstring)

    # When True, a schema-drift mismatch at startup (live alembic_version !=
    # the migration chain's head) is fatal — the process refuses to boot
    # rather than serve against an unmigrated/mismatched database. Default
    # False so a stale local dev DB gets a loud log line instead of a boot
    # loop; the k8s ConfigMap sets this True for real deployments.
    FAIL_ON_SCHEMA_DRIFT: bool = False

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

  # S3 / MinIO
    S3_ENDPOINT: str
    # URL the *browser* hits for presigned object URLs. Only set explicitly
    # when the public host differs from the one the backend talks to (CDN, or
    # a containerised MinIO reachable at a different hostname). Left unset it
    # falls back to S3_ENDPOINT — see the validator below.
    S3_PUBLIC_ENDPOINT: str | None = None
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_BUCKET: str
    S3_REGION: str
    
    # Auth
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""  # not used by the ID-token flow; reserved for a future server-side OAuth flow

    # Email verification (Gmail SMTP)
    FRONTEND_URL: str = "http://localhost:3000"
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""  # Google App Password
    SMTP_FROM: str = ""
    # Inbox that receives public "Book a Demo" lead notifications. Defaults to
    # SMTP_FROM (our own mailbox) when left unset — see the validator below.
    DEMO_NOTIFY_EMAIL: str = ""
    # Publicly reachable https:// URL of the 40px brand mark used in email
    # headers. Must be fetchable by Google's image proxy, so localhost is no
    # use — leave it empty in dev and the emails fall back to a text-only
    # wordmark. Never embed the logo as a cid: attachment instead: Gmail lists
    # every attached part in the attachment strip even when it renders inline.
    EMAIL_LOGO_URL: str = ""
    EMAIL_VERIFICATION_TOKEN_TTL_SECONDS: int = 86400  # 24 h — must outlast SMTP delivery
    PASSWORD_RESET_TOKEN_TTL_SECONDS: int = 3600  # 60 min

    # ML
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    MAX_TRAINING_WORKERS: int = 4
    # Master switch: may this deployment route training to the GPU queue at all?
    # False means "GPU routing is not permitted" — a GPU request is rejected
    # outright, regardless of GPU_REQUIRE_LIVE_WORKER.
    GPU_ENABLED: bool = False
    # When true, a GPU request is additionally rejected unless some Celery worker
    # is actually consuming `training_gpu`. Default true because no on-demand GPU
    # lifecycle manager exists yet, so an unconsumed queue means the job waits
    # forever. docs/gpu_lifecycle_architecture.md specifies the queue as a
    # *buffer* — when that reconciler lands, flip this to false and enqueueing
    # against a stopped GPU host becomes correct again.
    GPU_REQUIRE_LIVE_WORKER: bool = True
    # Ceiling on the Celery inspect() probe above. It sits in the request path,
    # so this is deliberately short — see app/core/gpu_availability.py.
    GPU_WORKER_PROBE_TIMEOUT_SECONDS: float = 1.0

    # Synthetic Data — OpenAI Image Generation (the only supported provider)
    #
    # Authentication is user-provided per request (docs/Action/
    # datasynthetic_implementationplan.md §0) — there is no server-managed
    # OpenAI credential. OPENAI_API_KEY / OPENAI_ORG_ID are deliberately not
    # settings here; the server never holds a standing OpenAI credential.
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_IMAGE_MODEL: str = "gpt-image-1"
    # Read timeout per generation request. `high` quality at 1536x1024 is
    # genuinely slow — the httpx default would abort valid requests.
    OPENAI_IMAGE_TIMEOUT_SECONDS: float = 180.0
    SYNTHETIC_MAX_IMAGES_PER_JOB: int = 25
    SYNTHETIC_MAX_JOBS_PER_USER_PER_DAY: int = 20
    # A `running` job older than this (API restart mid-generation) is treated
    # as stale and flipped to `failed` on next read.
    SYNTHETIC_JOB_STALE_SECONDS: int = 900

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    @model_validator(mode="after")
    def _default_demo_notify_email(self) -> "Settings":
        # Fall back to our SMTP sender address when no dedicated lead inbox is
        # configured, so demo notifications always have a destination.
        if not self.DEMO_NOTIFY_EMAIL:
            self.DEMO_NOTIFY_EMAIL = self.SMTP_FROM
        return self

    @model_validator(mode="after")
    def _default_public_s3_endpoint(self) -> "Settings":
        # Presigned URLs are signed against S3_PUBLIC_ENDPOINT. Leaving it None
        # makes boto3 resolve the *real AWS* endpoint, so a local MinIO
        # deployment silently signs https://s3.amazonaws.com/... with minioadmin
        # credentials and every browser fetch 403s. Falling back to S3_ENDPOINT
        # keeps a misconfigured deployment pointed at its own storage.
        if not self.S3_PUBLIC_ENDPOINT:
            self.S3_PUBLIC_ENDPOINT = self.S3_ENDPOINT
        return self

    @model_validator(mode="after")
    def _reject_placeholder_secrets_in_production(self) -> "Settings":
        # Presence of these fields is already enforced unconditionally by
        # pydantic (no default = required). This only catches the case where
        # a value technically exists but is a known dev/test placeholder.
        if self.APP_ENV != "production":
            return self

        problems: list[str] = []
        for field, value, min_len in (
            ("SECRET_KEY", self.SECRET_KEY, 32),
            ("JWT_SECRET", self.JWT_SECRET, 32),
            ("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD, 0),
            ("S3_ACCESS_KEY", self.S3_ACCESS_KEY, 0),
            ("S3_SECRET_KEY", self.S3_SECRET_KEY, 0),
        ):
            if value.lower() in _PLACEHOLDER_SECRETS:
                problems.append(f"{field} is a known placeholder value")
            elif min_len and len(value) < min_len:
                problems.append(f"{field} is shorter than {min_len} characters")
        if problems:
            raise ValueError(
                "APP_ENV=production refuses to start with insecure settings: "
                + "; ".join(problems)
            )
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

