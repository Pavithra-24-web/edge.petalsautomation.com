"""
Pydantic schemas for the Synthetic Data feature (Phase 3 of
docs/Action/datasynthetic_implementationplan.md).

Structural validation only (types/shape). The enum/range/guard rules in the
plan's §2.4 validation table (prompt length, label leading-brace rejection,
count cap, allowed size/quality/background/sample_type values) are enforced
in the endpoint as explicit 400s, not here — see synthetic_data.py — so every
rejection carries the exact message the plan specifies rather than a generic
pydantic 422.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class SyntheticDataGenerateRequest(BaseModel):
    """`prompt`/`label`/`count` default to an empty/invalid value rather than
    being pydantic-required, so an omitted field fails the endpoint's own
    400 in §2.4's validation table instead of FastAPI's generic 422 — every
    rejection in that table is a 400 with a specific message, deliberately,
    per §2.4 ("all 400 unless noted")."""

    project_id: str = ""
    # User-provided OpenAI key (plan §0/§2.4), sent once per request and never
    # persisted. Empty-string default so a missing key fails the endpoint's
    # own 400 rather than pydantic's generic 422 — see the class docstring.
    api_key: str = ""
    prompt: str = ""
    label: str = ""
    # `Any`, not `int`: a non-integer count (e.g. a string) must reach the
    # endpoint's own 400 rather than being rejected by pydantic coercion as
    # a generic 422 — see the class docstring.
    count: Any = 3
    sample_type: str = "automatic"
    size: str = "1024x1024"
    quality: str = "standard"
    background: str = "auto"


class SyntheticDataConfig(BaseModel):
    # No `configured` field (plan §0/§2.4) — there is nothing server-side left
    # to be configured or not; the server holds no OpenAI credential. The
    # frontend gates the Generate button on whether the user has entered a
    # key into the form, never on this response.
    provider: str = "openai"
    provider_label: str = "OpenAI Image Generation"
    model: str
    max_images_per_job: int
    sizes: List[str]
    qualities: List[str]
    backgrounds: List[str]
    # {"<size>:<ui_quality>": usd_per_image}, e.g. "1024x1024:standard" — UI
    # vocabulary throughout, matching `sizes`/`qualities` above (see §4.6 of
    # the plan). Best effort: a combination missing from the provider's price
    # table is simply absent here rather than guessed.
    price_per_image_usd: Dict[str, float]


class SyntheticDataJobOut(BaseModel):
    id: str
    project_id: str
    status: str
    provider: str
    model: str
    prompt: str
    label_name: str
    label_id: Optional[str] = None
    requested_count: int
    generated_count: int
    failed_count: int
    sample_type: str
    parameters: Dict[str, Any]
    sample_ids: List[str]
    estimated_cost_usd: Optional[float] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
