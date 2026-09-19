"""
Pydantic schemas for project-level Versioning (C2) — see docs/Action/parityfix.md.

Phase 1 covers store + list + detail (the read path). Phase 2 adds restore
(parityfix.md §4.3, §5). Phase 3 (this file also now includes) adds compare
and publish (parityfix.md §8 Phase 3).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProjectVersionCreate(BaseModel):
    project_id: str
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectVersionSummary(BaseModel):
    """List view — header fields only, no per-impulse or sample detail.
    Matches the existing 'list is cheap, detail is not' pattern (GET
    /training/jobs)."""
    id: str
    project_id: str
    version_number: int
    name: Optional[str] = None
    description: Optional[str] = None
    sample_count: int
    train_sample_count: int
    test_sample_count: int
    class_names: List[str]
    impulse_count: int
    best_accuracy: Optional[float] = None
    status: str
    published_at: Optional[datetime] = None
    created_by: Optional[str] = None
    created_by_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VersionImpulseOut(BaseModel):
    """One project_version_impulses row (parityfix.md §2.2)."""
    id: str
    impulse_id: Optional[str] = None
    impulse_name: str
    dsp_blocks_snapshot: List[Dict[str, Any]]
    ml_blocks_snapshot: List[Dict[str, Any]]
    output_config_snapshot: Dict[str, Any]
    input_config_snapshot: Dict[str, Any]
    trained_model_id: Optional[str] = None
    training_job_id: Optional[str] = None
    accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    confusion_matrix: Optional[Any] = None
    training_history: Optional[Any] = None
    # Self-contained training configuration + artifact/feature-cache pointers
    # (parityfix.md §2.2, migration 0038) — see VersionImpulseOut's restore
    # counterpart, which reads these instead of the live TrainingJob/
    # TrainedModel tables.
    training_config_snapshot: Optional[Dict[str, Any]] = None
    trained_model_formats_snapshot: Optional[List[Dict[str, Any]]] = None
    features_storage_key: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class VersionSampleOut(BaseModel):
    id: str
    sample_id: Optional[str] = None
    sample_name: str
    label_name: Optional[str] = None
    sample_type: str

    model_config = ConfigDict(from_attributes=True)


class VersionImpulseDiff(BaseModel):
    """One impulse's status in the diff against the project's current live
    state (parityfix.md §4.5)."""
    impulse_name: str
    impulse_id: Optional[str] = None
    status: str                    # "unchanged" | "changed" | "deleted" | "added_since"
    changed_fields: List[str] = Field(default_factory=list)
    accuracy_delta: Optional[float] = None
    live_accuracy: Optional[float] = None


class VersionDiffSummary(BaseModel):
    """Read-only comparison of this version's snapshot against the project's
    current live state. Computed on read — nothing here is persisted."""
    project_config_changed: bool
    changed_project_fields: List[str] = Field(default_factory=list)
    deployment_changed: bool
    sample_count_delta: int
    class_names_added: List[str] = Field(default_factory=list)
    class_names_removed: List[str] = Field(default_factory=list)
    impulses: List[VersionImpulseDiff] = Field(default_factory=list)
    best_accuracy_delta: Optional[float] = None
    live_best_accuracy: Optional[float] = None


class VersionRestoreRequest(BaseModel):
    """Body of POST /project-versions/{id}/restore (parityfix.md §4.3, §5).
    Restore creates a new project from the snapshot — this is the new
    project's name and optional description, nothing else."""
    name: str
    description: Optional[str] = None


class VersionRestoreResponse(BaseModel):
    """The new project created by the restore. The source project is never
    touched, so there is nothing to reconcile or report back.

    The counts are diagnostics, not a reconciliation report: restore either
    produced a complete clone or failed and left nothing behind (see
    ``app.services.project_clone``), so there is no partial state for the
    caller to inspect or repair. They exist so an operator can see at a
    glance what a restore moved, and so the one genuinely lossy case —
    samples whose media was deleted from storage after the version was
    stored, and therefore cannot be cloned by any implementation — is stated
    rather than silently absorbed."""
    new_project_id: str
    new_project_name: str
    cloned_row_counts: Dict[str, int] = Field(default_factory=dict)
    cloned_asset_count: int = 0
    skipped_sample_names: List[str] = Field(default_factory=list)


class VersionCompareResponse(BaseModel):
    """Compact per-category summary of two project versions (parityfix.md
    §4.5, §7.3) — computed on read from the two stored rows, no third table
    needed. Deliberately has no per-impulse array: the per-impulse matching
    is still done server-side to produce the counts below, but is not
    serialised, since the compare dialog shows a summary, not a listing."""
    version_a: ProjectVersionSummary
    version_b: ProjectVersionSummary
    project_config_same: bool
    deployment_same: bool
    sample_count_a: int
    sample_count_b: int
    class_count_a: int
    class_count_b: int
    impulse_total_a: int
    impulse_total_b: int
    impulses_added: int         # in b, not in a (matched by impulse_name)
    impulses_removed: int       # in a, not in b
    impulses_modified: int      # in both, with differing block/input/output config


class ProjectVersionDetail(ProjectVersionSummary):
    project_config_snapshot: Dict[str, Any]
    deployment_snapshot: Dict[str, Any]
    post_processing_snapshot: Optional[Any] = None
    impulses: List[VersionImpulseOut]
    samples: List[VersionSampleOut]
    samples_total: int
    samples_page: int
    samples_page_size: int
    diff_summary: VersionDiffSummary
