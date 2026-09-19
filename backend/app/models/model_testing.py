"""
SQLAlchemy ORM models for the Model Testing feature.

These tables extend the existing schema to support:
  - Named model versions (Quantized int8, Float32, EON Compiled)
  - Per-sample test result tracking with F1 score + expected outcome
  - Test run aggregates (accuracy, sample counts)
  - Per-metric rows (Precision, Recall, F1 Score)
"""
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, ForeignKey, Text, Enum as SAEnum, JSON,
)
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.user import gen_uuid


# ─── Exceptions ─────────────────────────────────────────────────────────────────

class UnsupportedModelTestingError(Exception):
    """Raised when an output_type has no Model Testing scoring path yet.

    Lives here (a dependency-light module imported by both the service and the
    worker) so the worker can catch it at module level without importing the
    heavier model_testing_service. It is an expected, known terminal condition
    (e.g. SSD detection), not a crash — the worker marks the run failed with a
    clear message rather than logging an unexpected-exception traceback.
    """


# ─── Enums ────────────────────────────────────────────────────────────────────

class TestResultStatus(str, enum.Enum):
    pending   = "pending"   # Not yet classified
    pass_     = "pass"      # Classified — result matches expected (or scored ≥ threshold)
    fail      = "fail"      # Classified — result does not match expected
    uncertain = "uncertain" # Correct class but confidence below uncertain_threshold


class TestRunStatus(str, enum.Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"
    cancelled = "cancelled"


class QuantizationType(str, enum.Enum):
    int8     = "int8"
    float32  = "float32"
    eon      = "eon_compiled"


# ─── ModelVersion ─────────────────────────────────────────────────────────────

class ModelVersion(Base):
    """
    A user-selectable model variant (e.g. 'Quantized (int8)') linked to
    the impulse. In production this references a TrainedModel artifact;
    during testing it can stand alone with simulated results.
    """
    __tablename__ = "model_versions"

    id                  = Column(String, primary_key=True, default=gen_uuid)
    project_id          = Column(String, ForeignKey("projects.id"),       nullable=False)
    impulse_id          = Column(String, ForeignKey("impulses.id"),       nullable=False)
    # Optional back-reference to an existing trained model artifact
    trained_model_id    = Column(String, ForeignKey("trained_models.id"), nullable=True)

    version_number      = Column(String,   nullable=False)          # "v1", "v2" …
    name                = Column(String,   nullable=False)          # human-readable
    quantization_type   = Column(SAEnum(QuantizationType), default=QuantizationType.int8)
    is_active           = Column(Boolean,  default=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project       = relationship("Project",      foreign_keys=[project_id])
    impulse       = relationship("Impulse",      foreign_keys=[impulse_id])
    trained_model = relationship("TrainedModel", foreign_keys=[trained_model_id])
    test_runs     = relationship("ModelTestRun", back_populates="model_version",
                                 cascade="all, delete-orphan")


# ─── ModelTestRun ─────────────────────────────────────────────────────────────

class ModelTestRun(Base):
    """
    A single 'Classify all' execution. Stores aggregate accuracy and links
    to per-sample results (ModelTestSample) and per-metric rows (MetricResult).
    """
    __tablename__ = "model_test_runs"

    id               = Column(String, primary_key=True, default=gen_uuid)
    project_id       = Column(String, ForeignKey("projects.id"),      nullable=False)
    impulse_id       = Column(String, ForeignKey("impulses.id"),      nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)

    # Async job tracking
    status           = Column(String, default=TestRunStatus.pending, nullable=False)
    started_at       = Column(DateTime, nullable=True)
    completed_at     = Column(DateTime, nullable=True)
    error            = Column(String,   nullable=True)

    accuracy         = Column(Float,   nullable=True)    # 0.0 – 100.0
    total_samples    = Column(Integer, default=0)
    passed_samples   = Column(Integer, default=0)
    failed_samples   = Column(Integer, default=0)

    created_at       = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project       = relationship("Project",      foreign_keys=[project_id])
    impulse       = relationship("Impulse",      foreign_keys=[impulse_id])
    model_version = relationship("ModelVersion", back_populates="test_runs")
    metrics       = relationship("MetricResult", back_populates="test_run",
                                  cascade="all, delete-orphan")
    sample_results = relationship("ModelTestSample", back_populates="test_run",
                                   cascade="all, delete-orphan")


# ─── MetricResult ─────────────────────────────────────────────────────────────

class MetricResult(Base):
    """
    One metric value row within a ModelTestRun.
    E.g. metric_name='precision_non_background', metric_value=0.0
    """
    __tablename__ = "metric_results"

    id                  = Column(String, primary_key=True, default=gen_uuid)
    test_run_id         = Column(String, ForeignKey("model_test_runs.id"), nullable=False)
    metric_name         = Column(String, nullable=False)    # internal key
    metric_display_name = Column(String, nullable=False)    # UI label
    metric_value        = Column(Float,  nullable=True)
    created_at          = Column(DateTime, default=datetime.utcnow)

    test_run = relationship("ModelTestRun", back_populates="metrics")


# ─── ModelTestSample ──────────────────────────────────────────────────────────

class ModelTestSample(Base):
    """
    Per-sample testing record. Tracks the expected outcome set by the user,
    the computed F1 score after classification, and the final pass/fail status.

    sample_id is nullable so seed rows can exist independently of real
    uploaded samples (useful for demo / testing).
    """
    __tablename__ = "model_test_samples"

    id               = Column(String, primary_key=True, default=gen_uuid)
    project_id       = Column(String, ForeignKey("projects.id"),       nullable=False)
    impulse_id       = Column(String, ForeignKey("impulses.id"),       nullable=False)
    # Soft reference — nullable for standalone / seed rows
    sample_id        = Column(String, ForeignKey("samples.id"),        nullable=True)
    test_run_id      = Column(String, ForeignKey("model_test_runs.id"), nullable=True)

    # NULL = never scored, or reset after a retrain
    scored_by_trained_model_id = Column(
        String, ForeignKey("trained_models.id"), nullable=True
    )

    sample_name      = Column(String,  nullable=False)     # display name / filename
    expected_outcome = Column(String,  nullable=True)      # user-defined label, "-" if unset
    f1_score         = Column(Float,   nullable=True)      # 0–100 or None (not yet run)
    result_status    = Column(SAEnum(TestResultStatus, native_enum=False), default=TestResultStatus.pending)

    # What the model actually predicted per sample
    predicted_class  = Column(String, nullable=True)
    iou_score        = Column(Float,  nullable=True)
    predicted_boxes  = Column(JSON,   nullable=True)   # [{x,y,x2,y2,score,label}, ...]

    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project          = relationship("Project",      foreign_keys=[project_id])
    impulse          = relationship("Impulse",      foreign_keys=[impulse_id])
    sample           = relationship("Sample",       foreign_keys=[sample_id])
    test_run         = relationship("ModelTestRun", back_populates="sample_results")
    scored_by_model  = relationship("TrainedModel", foreign_keys=[scored_by_trained_model_id])
