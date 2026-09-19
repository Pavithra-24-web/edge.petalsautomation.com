"""
SQLAlchemy ORM Models — full database schema
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, JSON, Text, Enum as SAEnum, BigInteger,
    UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


# ─── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    admin = "admin"
    developer = "developer"
    viewer = "viewer"


class SampleType(str, enum.Enum):
    training = "training"
    testing = "testing"
    automatic = "automatic"       # backend splits 80/20 automatically
    postprocessing = "postprocessing"  # used for post-processing pipelines


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


DeploymentStatus = JobStatus  # alias used by compatibility resolver


class DeployTarget(str, enum.Enum):
    tflite = "tflite"
    arduino = "arduino"
    esp32 = "esp32"
    raspberry_pi = "raspberry_pi"
    unoq = "unoq"
    cpp = "cpp"
    pxe = "pxe"


class DeviceClass(str, enum.Enum):
    """Hardware class of a device catalog entry (Target Device Phase 2,
    docs/target_device_phase2.md). Distinct from `DeployTarget` — this
    describes the physical device, not the build artifact it resolves to."""
    microcontroller = "microcontroller"
    linux_sbc = "linux_sbc"
    accelerator = "accelerator"
    desktop = "desktop"
    mobile = "mobile"


# ─── User & Auth ──────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False)
    # Nullable — accounts created via Google Sign-In have no local password.
    hashed_password = Column(String, nullable=True)
    # Google's stable per-account identifier ("sub" claim). Set only for
    # accounts that have signed in with Google at least once; a
    # password-registered account can later link one by signing in with the
    # matching email via Google.
    google_sub = Column(String, unique=True, nullable=True, index=True)
    role = Column(SAEnum(UserRole), default=UserRole.developer)
    is_active = Column(Boolean, default=True)
    # Email verification for password accounts (Google accounts are created
    # already verified — see google_login()). Distinct from is_active, which
    # is an account-disable flag.
    is_verified = Column(Boolean, default=False)
    verification_token_hash = Column(String, unique=True, nullable=True, index=True)
    verification_token_expires_at = Column(DateTime, nullable=True)
    reset_token_hash = Column(String, unique=True, nullable=True, index=True)
    reset_token_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False, unique=True)
    last_used = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="api_keys")


# ─── Project ──────────────────────────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=gen_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Monotonic counter for default "Impulse N" naming. Only ever advances —
    # deleting an impulse does not free its number.
    impulse_seq = Column(Integer, nullable=False, default=0, server_default="0")

    # Which content the shared dashboard pages render for this project —
    # "object_detection" | "motion". Presentation-only: written once at
    # project creation and read by exactly the frontend plus two backend
    # helpers (impulse-create seeding, catalog filtering). It is never a
    # runtime pipeline discriminator — `Impulse.input_type` / `sensor_type`
    # remain the sole authority for DSP, training, evaluation and deployment.
    # See docs/Motion recognition/motion_phase0.md §3.
    project_type = Column(String, nullable=False)

    # The hardware this project targets — a `DeviceCatalogEntry.slug`, or NULL
    # for "not chosen yet", which is a valid state everywhere it is read.
    #
    # Deliberately NOT a foreign key. The catalog is seed data owned by
    # migrations: a board can be renamed or retired by a later migration, and a
    # FK would either block that or cascade a project's selection away silently.
    # Resolution is a lookup that tolerates a miss — see `_target_device` in
    # api/v1/endpoints/projects.py.
    target_device_slug = Column(String, nullable=True)

    # Per-project overrides of the selected board's application budget
    # (Target Device Phase 3, docs/target_device_phase3.md). NULL means "use
    # the board's own DeviceSpecification value" — these are overrides only,
    # never a snapshot of the resolved figure, so a later correction to the
    # specification is picked up automatically by any project that hasn't
    # overridden it. Resolution lives in `_target_device_config` in
    # api/v1/endpoints/projects.py. Cleared together whenever
    # `target_device_slug` changes to a different board (same file,
    # `update_project`) — a value carried from one board is not a
    # customisation of another, it's a wrong number.
    target_device_custom_name = Column(String, nullable=True)
    target_device_ram_kb = Column(Integer, nullable=True)
    target_device_rom_kb = Column(Integer, nullable=True)
    target_device_latency_ms = Column(Integer, nullable=True)

    owner = relationship("User", back_populates="projects")
    labels = relationship("Label", back_populates="project", cascade="all, delete-orphan")
    samples = relationship("Sample", back_populates="project", cascade="all, delete-orphan")
    impulses = relationship("Impulse", back_populates="project", cascade="all, delete-orphan")
    devices = relationship("Device", back_populates="project", cascade="all, delete-orphan")
    ai_labeling_actions = relationship("AILabelingAction", back_populates="project", cascade="all, delete-orphan")
    synthetic_jobs = relationship("SyntheticDataJob", back_populates="project", cascade="all, delete-orphan")
    post_processing_settings = relationship(
        "PostProcessingSettings", back_populates="project",
        uselist=False, cascade="all, delete-orphan",
    )
    processing_jobs = relationship("ProcessingJob", back_populates="project", cascade="all, delete-orphan")


# ─── Dataset / Samples ────────────────────────────────────────────────────────

class Label(Base):
    __tablename__ = "labels"

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    color = Column(String, default="#3B8BD4")
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="labels")
    samples = relationship("Sample", back_populates="label")


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (
        UniqueConstraint("project_id", "payload_hash", name="uq_samples_project_payload_hash"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    label_id = Column(String, ForeignKey("labels.id"), nullable=True)
    filename = Column(String, nullable=False)
    storage_key = Column(String, nullable=False)
    sample_type = Column(SAEnum(SampleType), default=SampleType.training)

    # Metadata
    sensor_type = Column(String, nullable=True)   # accelerometer, microphone, etc.
    frequency_hz = Column(Float, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    num_channels = Column(Integer, default=1)
    num_samples = Column(Integer, nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    payload_hash = Column(String, nullable=True, index=True)
    extra_metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=True)

    project = relationship("Project", back_populates="samples")
    label = relationship("Label", back_populates="samples")
    features = relationship("FeatureSet", back_populates="sample", cascade="all, delete-orphan")
    predictions = relationship("AIPrediction", back_populates="sample", cascade="all, delete-orphan")


# ─── Impulse / Pipeline ───────────────────────────────────────────────────────

class Impulse(Base):
    """An impulse is the full ML pipeline: DSP blocks + learning blocks."""
    __tablename__ = "impulses"

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Input config
    window_size_ms = Column(Integer, default=1000)
    window_increase_ms = Column(Integer, default=500)
    frequency_hz = Column(Float, default=100.0)
    zero_pad_allowed = Column(Boolean, default=True)

    input_type = Column(String, default="time-series")
    input_axes = Column(JSON, default=list)
    sensor_type = Column(String, nullable=True)
    
    image_width = Column(Integer, default=96)
    image_height = Column(Integer, default=96)
    resize_mode = Column(String, default="Fit shortest axis")

    # Percent of training samples actually used by feature generation +
    # training. 100 = use everything; lower values accelerate iteration by
    # randomly subsampling the training split (test split is always kept whole).
    train_subset_percent = Column(Float, default=100.0, server_default="100", nullable=False)

    output_config = Column(JSON, default=dict)

    # DSP block config (JSON)
    dsp_blocks = Column(JSON, default=list)   # [{type, params, input_axes}]
    # ML block config (JSON)
    ml_blocks = Column(JSON, default=list)    # [{architecture, params}]

    # Backend-confirmed save state for the Parameters step. NULL until the user
    # successfully POSTs the parameters form; downstream UI (Generate features)
    # is gated on this timestamp being set. Cleared whenever the DSP block set
    # itself is restructured (block added/removed) so the user re-saves.
    dsp_params_saved_at = Column(DateTime, nullable=True)

    # Pointer to the training run whose artifacts are the "currently active"
    # model for this impulse. Set ONLY on successful completion (fresh or
    # retrain). NEVER cleared on start/cancel/fail — the previous pointer
    # remains in the database so internal rollback/audit stays possible. The
    # UI decides whether to display the pointed-to artifacts; the DB never lies.
    # Nullable FK without an explicit FK constraint to TrainingJob.id to avoid
    # a circular cascade with training_jobs.cascade="all, delete-orphan".
    active_model_run_id = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="impulses")
    training_jobs = relationship("TrainingJob", back_populates="impulse", cascade="all, delete-orphan")
    feature_sets = relationship("FeatureSet", back_populates="impulse", cascade="all, delete-orphan")
    dsp_feature_jobs = relationship(
        "DspFeatureJob", back_populates="impulse", cascade="all, delete-orphan",
    )


# ─── DSP Features ─────────────────────────────────────────────────────────────

class FeatureSet(Base):
    """Extracted DSP features for a sample + impulse combination."""
    __tablename__ = "feature_sets"

    id = Column(String, primary_key=True, default=gen_uuid)
    sample_id = Column(String, ForeignKey("samples.id"), nullable=False)
    impulse_id = Column(String, ForeignKey("impulses.id"), nullable=False)
    storage_key = Column(String, nullable=False)   # .npy file in S3
    feature_shape = Column(JSON, nullable=True)    # [rows, cols]
    created_at = Column(DateTime, default=datetime.utcnow)

    sample = relationship("Sample", back_populates="features")
    impulse = relationship("Impulse", back_populates="feature_sets")


# ─── Training ─────────────────────────────────────────────────────────────────

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    impulse_id = Column(String, ForeignKey("impulses.id"), nullable=False)
    celery_task_id = Column(String, nullable=True)
    status = Column(SAEnum(JobStatus), default=JobStatus.pending)

    # Distinguishes a fresh training run from a retrain. "fresh" runs are
    # destructive in intent — the user signaled they want to replace the
    # active model, so result panels collapse to placeholders the moment the
    # run starts. "retrain" runs are non-destructive — the active model stays
    # visible while the new run is in flight. Stored as a free-form String
    # rather than SAEnum so old jobs without this column read back as "fresh"
    # via a server_default; the application enforces the two-value vocabulary.
    run_kind = Column(String, nullable=False, default="fresh", server_default="fresh")

    # Hyperparameters
    epochs = Column(Integer, default=100)
    learning_rate = Column(Float, default=0.001)
    batch_size = Column(Integer, default=32)
    validation_split = Column(Float, default=0.2)
    optimizer = Column(String, default="adam")
    device_type = Column(String, default="cpu")   # "cpu" | "gpu"
    extra_params = Column(JSON, default=dict)

    # Results
    best_accuracy = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    training_history = Column(JSON, nullable=True)  # per-epoch metrics
    confusion_matrix = Column(JSON, nullable=True)
    classification_report = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    impulse = relationship("Impulse", back_populates="training_jobs")
    trained_models = relationship("TrainedModel", back_populates="training_job", cascade="all, delete-orphan")


class DspFeatureJob(Base):
    """A single DSP feature-generation run for an impulse.

    Mirrors TrainingJob's shape so the unified /jobs endpoint can fold it
    into the same response. The Celery task that writes `features.npz`
    updates this row at start (running, started_at) and at end
    (completed/failed, completed_at, error_message).

    project_id is denormalised — feature-gen is impulse-scoped, but the
    /jobs union filters by project, so we avoid a JOIN through impulses
    on a hot read path.

    input_type is snapshotted at run time so the jobs page label stays
    accurate even if the impulse's input_type is later edited.
    """
    __tablename__ = "dsp_feature_jobs"

    id              = Column(String, primary_key=True, default=gen_uuid)
    impulse_id      = Column(String, ForeignKey("impulses.id"), nullable=False, index=True)
    project_id      = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    celery_task_id  = Column(String, nullable=True)
    status          = Column(SAEnum(JobStatus), default=JobStatus.pending, nullable=False)
    input_type      = Column(String, nullable=True)
    error_message   = Column(Text, nullable=True)

    started_at      = Column(DateTime, nullable=True)
    completed_at    = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    impulse = relationship("Impulse", back_populates="dsp_feature_jobs")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(String, primary_key=True, default=gen_uuid)
    training_job_id = Column(String, ForeignKey("training_jobs.id"), nullable=False)
    version = Column(String, nullable=False)
    format = Column(String, nullable=False)           # keras, tflite, onnx
    storage_key = Column(String, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    model_metadata = Column(JSON, default=dict)       # input/output shapes, ops count
    created_at = Column(DateTime, default=datetime.utcnow)

    training_job = relationship("TrainingJob", back_populates="trained_models")
    deployments = relationship("Deployment", back_populates="model", cascade="all, delete-orphan")


# ─── Deployment ───────────────────────────────────────────────────────────────

class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(String, primary_key=True, default=gen_uuid)
    model_id = Column(String, ForeignKey("trained_models.id"), nullable=False)
    target = Column(SAEnum(DeployTarget), nullable=False)
    status = Column(SAEnum(JobStatus), default=JobStatus.pending)
    storage_key = Column(String, nullable=True)     # generated package key
    download_url = Column(String, nullable=True)
    celery_task_id = Column(String, nullable=True)
    options = Column(JSON, default=dict)             # quantization, optimization flags
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Targeting (4.5)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)
    deployment_target = Column(String, nullable=True)
    device_profile = Column(String, nullable=True)   # NULL = target-wide fallback

    model = relationship("TrainedModel", back_populates="deployments")


# ─── Devices ──────────────────────────────────────────────────────────────────

class Device(Base):
    __tablename__ = "devices"

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)

    # Stable external identifier (e.g. MAC address, serial, hardware ID).
    # Unique per project; devices self-report this on registration/heartbeat.
    device_id = Column(String, nullable=True, index=True)

    name = Column(String, nullable=False)
    device_type = Column(String, nullable=True)        # esp32, arduino, rpi, custom, …
    connection = Column(String, nullable=True)         # wifi, ethernet, serial, ble, …
    firmware_version = Column(String, nullable=True)
    protocol_version = Column(String, nullable=True)
    supports_snapshot_streaming = Column(Boolean, default=False, nullable=False)
    remote_mgmt_host = Column(String, nullable=True)

    sensors = Column(JSON, default=list)               # [{name, type, freq_hz, axes}]
    device_metadata = Column(JSON, default=dict)

    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)       # soft-delete; None = active

    # Transitional: kept for backward-compat with heartbeat callers.
    # Do not use as authoritative online state — derive from last_seen instead.
    is_online = Column(Boolean, default=False)
    ip_address = Column(String, nullable=True)

    # Deployment targeting (4.5)
    deployment_target = Column(String, nullable=True)
    device_profile = Column(String, nullable=True)
    installed_deployment_id = Column(String, nullable=True)
    installed_model_version = Column(String, nullable=True)

    __table_args__ = (
        # A device_id must be unique within a project (NULLs are excluded by PG).
        UniqueConstraint("project_id", "device_id", name="uq_device_project_device_id"),
        # Fast lookup of active (non-deleted) devices per project.
        Index("ix_devices_project_active", "project_id", "deleted_at"),
    )

    project = relationship("Project", back_populates="devices")


# ─── Device Update History ────────────────────────────────────────────────────

class DeviceUpdateHistory(Base):
    __tablename__ = "device_update_history"

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    device_id = Column(String, ForeignKey("devices.id"), nullable=False)
    deployment_id = Column(String, ForeignKey("deployments.id"), nullable=False)
    status = Column(String, nullable=False)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ─── AI Labeling ──────────────────────────────────────────────────────────────

class AILabelingStatus(str, enum.Enum):
    draft = "draft"
    running = "running"
    completed = "completed"
    failed = "failed"


class AILabelingAction(Base):
    """Reusable AI labeling configuration (prompt + model settings)."""
    __tablename__ = "ai_labeling_actions"

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False, default="classification")  # classification | detection
    prompt = Column(Text, nullable=True)
    label_names = Column(Text, nullable=True)
    provider = Column(String, nullable=False, default="openai")  # openai | local | custom
    model_config_json = Column(JSON, default=dict)  # api_key, model_name, endpoint, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="ai_labeling_actions")
    jobs = relationship("AILabelingJob", back_populates="action", cascade="all, delete-orphan")


class AILabelingJob(Base):
    """A single execution of an AI labeling action against a set of samples."""
    __tablename__ = "ai_labeling_jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    action_id = Column(String, ForeignKey("ai_labeling_actions.id"), nullable=False)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    status = Column(SAEnum(AILabelingStatus), default=AILabelingStatus.running)

    total_samples = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    # Filtering options stored for reference
    filter_options = Column(JSON, default=dict)  # {skip_labeled, filter_classes, sample_type}

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    action = relationship("AILabelingAction", back_populates="jobs")
    predictions = relationship("AIPrediction", back_populates="job", cascade="all, delete-orphan")


class AIPrediction(Base):
    """Temporary AI-generated prediction — NOT a final label until approved."""
    __tablename__ = "ai_predictions"

    id = Column(String, primary_key=True, default=gen_uuid)
    job_id = Column(String, ForeignKey("ai_labeling_jobs.id"), nullable=False)
    sample_id = Column(String, ForeignKey("samples.id"), nullable=False)

    predicted_label = Column(String, nullable=True)
    query_fragment = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    bounding_boxes = Column(JSON, default=list)  # [{x, y, w, h, label, confidence}]

    status = Column(String, default="pending")  # pending | approved | rejected

    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("AILabelingJob", back_populates="predictions")
    sample = relationship("Sample", back_populates="predictions")


# ─── Synthetic Data ───────────────────────────────────────────────────────────

class SyntheticJobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class SyntheticDataJob(Base):
    """A single OpenAI image-generation run against a project."""
    __tablename__ = "synthetic_data_jobs"

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    status = Column(SAEnum(SyntheticJobStatus), default=SyntheticJobStatus.pending)

    # Present for readability of stored rows; not a dispatch key — there is no
    # second provider branch (see docs/Action/datasynthetic_implementationplan.md §2.1).
    provider = Column(String, nullable=False, default="openai")
    model = Column(String, nullable=False)

    prompt = Column(Text, nullable=False)
    label_name = Column(String, nullable=False)
    label_id = Column(String, ForeignKey("labels.id"), nullable=True)

    requested_count = Column(Integer, nullable=False)
    generated_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)

    sample_type = Column(String, nullable=False)
    parameters = Column(JSON, default=dict)  # {size, quality, background, output_format, n_per_request}
    sample_ids = Column(JSON, default=list)  # ids of the `samples` rows this job created

    estimated_cost_usd = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="synthetic_jobs")


# ─── Post-Processing Pipeline (Phase 1 foundation) ────────────────────────────

class PostProcessingSettings(Base):
    """Stores video post-processing pipeline configuration per project/impulse scope."""
    __tablename__ = "post_processing_settings"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "impulse_id",
            name="uq_post_processing_settings_project_impulse_id",
        ),
    )

    id         = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    impulse_id = Column(String, ForeignKey("impulses.id"), nullable=True)

    enabled            = Column(Boolean, default=True)
    threshold          = Column(Float, default=0.5)
    tracking_enabled   = Column(Boolean, default=False)
    keep_grace         = Column(Integer, default=3)
    max_observations   = Column(Integer, default=5)
    class_filter       = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="post_processing_settings")


class ProcessingJob(Base):
    """Records a single video post-processing job run against a project."""
    __tablename__ = "processing_jobs"

    id         = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)

    status             = Column(String, nullable=False, default="pending")
    input_video_path   = Column(String, nullable=False)
    output_video_path  = Column(String, nullable=True)
    error_message      = Column(Text, nullable=True)
    inference_time_ms  = Column(Float, nullable=True)
    celery_task_id     = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="processing_jobs")


# ─── Versioning (C2) — project-level ────────────────────────────────────────────
# See docs/Action/parityfix.md for the full design. A project version is a
# manual, user-initiated snapshot of the
# *whole project*: every impulse's block/training config (deep copy) plus a
# pointer to its active trained model, the project's deployment and
# post-processing settings (deep copy), and a manifest of the dataset at
# snapshot time. Nothing here is ever written by a background job — only the
# Store Project Version endpoint inserts rows (parityfix.md §1.0).

class ProjectVersion(Base):
    """The manifest header — one row per project snapshot (parityfix.md §2.1)."""
    __tablename__ = "project_versions"
    __table_args__ = (
        UniqueConstraint("project_id", "version_number", name="uq_project_versions_project_number"),
        Index("ix_project_versions_project_created", "project_id", "created_at"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)

    # Monotonic per-project counter, maintained the same way Project.impulse_seq
    # already is — incremented in the same transaction as the insert.
    version_number = Column(Integer, nullable=False)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)

    # Project configuration snapshot — deep copy, not a pointer.
    project_config_snapshot = Column(JSON, nullable=False)      # name, description, project_type
    deployment_snapshot = Column(JSON, nullable=False)          # target-device fields + per-impulse
                                                                  # last-used deployment target/profile/options
    post_processing_snapshot = Column(JSON, nullable=True)      # every PostProcessingSettings row
                                                                  # for the project (project- and impulse-scoped)

    # Dataset totals captured at snapshot time.
    sample_count = Column(Integer, nullable=False, default=0)
    train_sample_count = Column(Integer, nullable=False, default=0)
    test_sample_count = Column(Integer, nullable=False, default=0)
    class_names = Column(JSON, nullable=False, default=list)

    # Project-level rollup for the history table — per-impulse detail lives
    # on ProjectVersionImpulse.
    impulse_count = Column(Integer, nullable=False, default=0)
    best_accuracy = Column(Float, nullable=True)   # highest per-impulse accuracy in this
                                                     # snapshot, or NULL if none had a trained model

    status = Column(String, nullable=False, default="draft", server_default="draft")
    published_at = Column(DateTime, nullable=True)
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project")
    creator = relationship("User")
    impulses = relationship(
        "ProjectVersionImpulse", back_populates="version",
        cascade="all, delete-orphan", order_by="ProjectVersionImpulse.impulse_name",
    )
    samples = relationship(
        "ProjectVersionSample", back_populates="version", cascade="all, delete-orphan",
    )


class ProjectVersionImpulse(Base):
    """One row per impulse that existed in the project at snapshot time —
    what makes the snapshot cover the whole project rather than a single
    pipeline (parityfix.md §2.2)."""
    __tablename__ = "project_version_impulses"
    __table_args__ = (
        Index("ix_project_version_impulses_version", "version_id"),
        Index("ix_project_version_impulses_impulse", "impulse_id"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    version_id = Column(String, ForeignKey("project_versions.id", ondelete="CASCADE"), nullable=False)
    # ON DELETE SET NULL (not CASCADE) — deleting a live impulse must never
    # delete version history. See parityfix.md §5.4.
    impulse_id = Column(String, ForeignKey("impulses.id", ondelete="SET NULL"), nullable=True)
    # Denormalised so the row stays meaningful after the live impulse is
    # deleted or renamed.
    impulse_name = Column(String, nullable=False)

    # Block + training configuration snapshot — deep copy, not a pointer.
    dsp_blocks_snapshot = Column(JSON, nullable=False)
    ml_blocks_snapshot = Column(JSON, nullable=False)      # learning-block hyperparameters
    output_config_snapshot = Column(JSON, nullable=False)
    input_config_snapshot = Column(JSON, nullable=False)   # window/frequency/image-size fields
                                                              # + train_subset_percent

    # Trained model — pointer, not a copy. Nullable: a version can be taken
    # before any training has completed for this impulse.
    trained_model_id = Column(String, ForeignKey("trained_models.id"), nullable=True)
    training_job_id = Column(String, ForeignKey("training_jobs.id"), nullable=True)

    # Metrics copied at snapshot time so they survive a later cascade-delete
    # of the training_job/trained_model row.
    accuracy = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    training_history = Column(JSON, nullable=True)

    # Training configuration — the hyperparameters that produced the metrics
    # above (epochs, learning_rate, batch_size, validation_split, optimizer,
    # device_type, extra_params, run_kind, status, started_at, completed_at).
    # Copied, not just pointed-to, for the same reason the metrics above are:
    # restore must not depend on the source TrainingJob row still existing
    # (migration 0038).
    training_config_snapshot = Column(JSON, nullable=True)

    # Every TrainedModel row's {format, version, storage_key, file_size_bytes,
    # model_metadata} the training job produced — not just the single
    # `trained_model_id` above (typically the tflite build) — so a restored
    # impulse has every format a live one would (migration 0038).
    trained_model_formats_snapshot = Column(JSON, nullable=True)

    # The impulse's `features.npz` DSP feature-cache S3 key at snapshot time,
    # if one existed. Restore downloads it, remaps every embedded label/
    # sample id to the new project's own ids, and re-uploads it under the new
    # impulse's own key, so Feature Explorer and training don't require
    # regenerating features from scratch (migration 0038).
    features_storage_key = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    version = relationship("ProjectVersion", back_populates="impulses")
    impulse = relationship("Impulse")
    trained_model = relationship("TrainedModel")
    training_job = relationship("TrainingJob")


class ProjectVersionSample(Base):
    """The dataset manifest — one row per sample captured in a version.

    Denormalises sample_name/label_name/sample_type so the row stays
    meaningful even after the live sample is edited or deleted (parityfix.md
    §5.4) — a snapshot, not a live view onto `samples`.
    """
    __tablename__ = "project_version_samples"
    __table_args__ = (
        Index("ix_project_version_samples_version", "version_id"),
        Index("ix_project_version_samples_sample", "sample_id"),
    )

    id = Column(String, primary_key=True, default=gen_uuid)
    version_id = Column(String, ForeignKey("project_versions.id", ondelete="CASCADE"), nullable=False)
    # ON DELETE SET NULL (not CASCADE) — deleting a live sample must never
    # delete version history. See parityfix.md §5.4.
    sample_id = Column(String, ForeignKey("samples.id", ondelete="SET NULL"), nullable=True)
    sample_name = Column(String, nullable=False)
    label_name = Column(String, nullable=True)
    sample_type = Column(String, nullable=False)

    version = relationship("ProjectVersion", back_populates="samples")
    sample = relationship("Sample")
