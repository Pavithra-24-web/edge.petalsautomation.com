"""
OpenAI Image Generation client — the only module in the codebase that talks
to OpenAI or knows the `images/generations` request/response shape.

Scope constraint (see docs/Action/datasynthetic_implementationplan.md): the
only supported provider is OpenAI Image Generation. Nothing here is a
multi-provider abstraction — there is exactly one branch, ever.

Authentication (§0 of the plan): this module is stateless with respect to
auth. `api_key` is a required argument on every call, supplied by the
caller (the endpoint's pre-flight check, or the worker's generation loop) —
never read from `settings`. No module-level cache or client is kept alive
across calls, and no credential is retained by this module before, during,
or after a call.

Model/pricing note (verify before relying on for billing — not independently
re-checked against OpenAI's live pricing page in this pass):
    Model id and parameter surface (size/quality/background) match
    `gpt-image-1` as of the plan's writing. PRICE_TABLE_USD below is an
    order-of-magnitude estimate for `estimated_cost_usd`, not an invoice.
"""
from __future__ import annotations

import base64
import random
import time
from typing import Optional

import httpx

from app.core.config import settings


# ─── Errors ─────────────────────────────────────────────────────────────────

# OpenAI error `code`/`type` fragments that indicate a moderation/content-policy
# rejection rather than a malformed request. Matched case-insensitively against
# both fields since the exact vocabulary is not contractually stable.
_MODERATION_MARKERS = ("moderation", "safety", "content_policy")

# OpenAI reports "no paid access/credits" as HTTP 429 with one of these
# `code`/`type` fragments — indistinguishable from a transient rate limit by
# status code alone. Unlike a rate limit, retrying never helps (the account
# has no quota, full stop), so this is classified non-retryable with its own
# message rather than falling into the generic "try again" 429 branch.
_QUOTA_MARKERS = ("insufficient_quota", "billing")


class OpenAIImageError(Exception):
    """
    Normalised failure from the OpenAI Image Generation API.

    .status      HTTP status code, or None for a transport-level failure
                 (timeout, connection error) that never got a response.
    .code        provider error `code`, if any.
    .retryable   True for 429 / 5xx / timeout / connection errors — conditions
                 that may succeed on a later attempt with no change of input.
    .is_moderation  True when the provider rejected the *prompt itself*
                 (content policy), not any other parameter — the one
                 skip-this-image-and-continue case (§4.1 of the plan).
    .user_message  safe to show to the end user; never contains the API key.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: Optional[str] = None,
        retryable: bool = False,
        is_moderation: bool = False,
        user_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable
        self.is_moderation = is_moderation
        self.user_message = user_message or message


def _is_moderation_error(status: int, code: Optional[str], error_type: Optional[str]) -> bool:
    if status != 400:
        return False
    haystack = f"{code or ''} {error_type or ''}".lower()
    return any(marker in haystack for marker in _MODERATION_MARKERS)


def _is_quota_error(status: int, code: Optional[str], error_type: Optional[str]) -> bool:
    if status != 429:
        return False
    haystack = f"{code or ''} {error_type or ''}".lower()
    return any(marker in haystack for marker in _QUOTA_MARKERS)


def _error_from_response(resp: httpx.Response) -> OpenAIImageError:
    status = resp.status_code
    try:
        body = resp.json()
        err = body.get("error") or {}
    except Exception:
        err = {}
    code = err.get("code")
    error_type = err.get("type")
    provider_message = err.get("message") or resp.text[:500] or f"HTTP {status}"

    if status == 401:
        return OpenAIImageError(
            "OpenAI rejected the API key.",
            status=status, code=code, retryable=False,
            user_message=(
                "OpenAI rejected this API key. Check that the key is valid and that "
                "your OpenAI API account has available API access/credits."
            ),
        )
    if status == 403:
        return OpenAIImageError(
            provider_message,
            status=status, code=code, retryable=False,
            user_message=(
                f"{provider_message} Your OpenAI organization may need "
                "verification for this model."
            ),
        )
    if status == 400 and _is_moderation_error(status, code, error_type):
        return OpenAIImageError(
            provider_message,
            status=status, code=code, retryable=False, is_moderation=True,
            user_message="Image was rejected by OpenAI's content policy. Try rephrasing the prompt.",
        )
    if status == 400:
        return OpenAIImageError(
            provider_message,
            status=status, code=code, retryable=False,
            user_message=provider_message,
        )
    if status == 429 and _is_quota_error(status, code, error_type):
        return OpenAIImageError(
            provider_message,
            status=status, code=code, retryable=False,
            user_message=(
                "OpenAI rejected this API key. Check that the key is valid and that "
                "your OpenAI API account has available API access/credits."
            ),
        )
    if status == 429:
        return OpenAIImageError(
            provider_message,
            status=status, code=code, retryable=True,
            user_message="Rate limited by OpenAI — try again in a few minutes.",
        )
    if 500 <= status < 600:
        return OpenAIImageError(
            provider_message,
            status=status, code=code, retryable=True,
            user_message=provider_message,
        )
    return OpenAIImageError(
        provider_message,
        status=status, code=code, retryable=False,
        user_message=provider_message,
    )


# ─── UI ↔ API parameter mapping ─────────────────────────────────────────────
#
# The UI vocabulary is fixed by the product spec; the OpenAI API vocabulary is
# not identical (e.g. "Standard" -> "medium", never "low" — see the module-level
# rationale below). Kept as one dict so a model swap changes one place.

UI_QUALITY_TO_API = {
    "standard": "medium",
    "high": "high",
}

UI_BACKGROUND_TO_API = {
    "transparent": "transparent",
    "opaque": "opaque",
    "auto": "auto",
}

# UI size labels already match the API's own size strings (e.g. "1024x1024"),
# so this is an allowlist/validation map rather than a translation.
UI_SIZE_TO_API = {
    "1024x1024": "1024x1024",
    "1024x1536": "1024x1536",
    "1536x1024": "1536x1024",
}

# output_format is never user-facing (forced png for transparency support —
# see §3.2 of the plan) so it has no UI-side entry.
API_OUTPUT_FORMAT = "png"


def map_ui_params(size: str, quality: str, background: str) -> tuple[str, str, str]:
    """Translate UI-vocabulary field values to the API's own vocabulary.

    Raises ValueError for any value outside the fixed UI enums — this is the
    single place that both directions of the mapping table are enforced.
    """
    try:
        api_size = UI_SIZE_TO_API[size]
    except KeyError:
        raise ValueError(f"Unsupported image size: {size!r}")
    try:
        api_quality = UI_QUALITY_TO_API[quality.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unsupported quality: {quality!r}")
    try:
        api_background = UI_BACKGROUND_TO_API[background.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unsupported background: {background!r}")
    return api_size, api_quality, api_background


# ─── Price table ─────────────────────────────────────────────────────────────
#
# Indicative USD cost per image, keyed by (api_size, api_quality). Order of
# magnitude only — re-verify against OpenAI's current pricing page before
# treating `estimated_cost_usd` as anything but an estimate.
PRICE_TABLE_USD: dict[tuple[str, str], float] = {
    ("1024x1024", "low"): 0.011,
    ("1024x1024", "medium"): 0.042,
    ("1024x1024", "high"): 0.167,
    ("1024x1536", "low"): 0.016,
    ("1024x1536", "medium"): 0.063,
    ("1024x1536", "high"): 0.25,
    ("1536x1024", "low"): 0.016,
    ("1536x1024", "medium"): 0.063,
    ("1536x1024", "high"): 0.25,
}


def estimate_cost_usd(api_size: str, api_quality: str, count: int) -> Optional[float]:
    """Best-effort estimate for `count` images at the given API-vocabulary
    size/quality. Returns None if the combination isn't in the price table
    rather than guessing."""
    per_image = PRICE_TABLE_USD.get((api_size, api_quality))
    if per_image is None:
        return None
    return round(per_image * count, 4)


# ─── Retry/backoff ────────────────────────────────────────────────────────────

_MAX_ATTEMPTS = 4
_BACKOFF_CAP_SECONDS = 30


def _backoff_seconds(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(2 ** attempt, _BACKOFF_CAP_SECONDS) * random.random()


# ─── Request/response ─────────────────────────────────────────────────────────

def _build_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _fetch_from_url(url: str, timeout: httpx.Timeout, transport: Optional[httpx.BaseTransport]) -> bytes:
    """Defensive fallback for a provider response that carries `url` instead
    of `b64_json` (older/other models). Not the supported gpt-image-1 path —
    see §3.3 of the plan. https-only, strict timeout, no redirects followed
    into a non-https scheme."""
    if not url.startswith("https://"):
        raise OpenAIImageError(
            f"Refusing to fetch generated image from non-https URL: {url!r}",
            retryable=False,
        )
    with httpx.Client(timeout=timeout, transport=transport) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def generate_images(
    *,
    api_key: str,
    prompt: str,
    count: int,
    size: str,
    quality: str,
    background: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    transport: Optional[httpx.BaseTransport] = None,
) -> list[bytes]:
    """
    Call OpenAI's `images/generations` endpoint and return `count` PNG byte
    strings, authenticated with the caller-supplied `api_key`.

    `size`, `quality`, `background` are already API-vocabulary values (see
    `map_ui_params` for the UI -> API translation, done by the caller before
    this call). Raises `OpenAIImageError` — non-retryable errors (auth,
    invalid params, moderation) raise immediately; 429/5xx/timeout are retried
    with backoff up to `_MAX_ATTEMPTS` attempts before raising.

    `transport` is test-only: pass an `httpx.MockTransport` to exercise this
    function without a real network call.
    """
    read_timeout = timeout if timeout is not None else settings.OPENAI_IMAGE_TIMEOUT_SECONDS
    http_timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0)

    payload = {
        "model": model or settings.OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "n": count,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": API_OUTPUT_FORMAT,
    }

    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/images/generations"
    last_error: Optional[OpenAIImageError] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=http_timeout, transport=transport) as client:
                resp = client.post(url, headers=_build_headers(api_key), json=payload)
        except httpx.TimeoutException as exc:
            last_error = OpenAIImageError(
                f"Timed out contacting OpenAI: {exc}",
                retryable=True,
                user_message="OpenAI image generation timed out.",
            )
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_backoff_seconds(attempt, None))
                continue
            raise last_error
        except httpx.TransportError as exc:
            last_error = OpenAIImageError(
                f"Connection error contacting OpenAI: {exc}",
                retryable=True,
                user_message="Could not reach OpenAI.",
            )
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_backoff_seconds(attempt, None))
                continue
            raise last_error

        if resp.status_code == 200:
            body = resp.json()
            images: list[bytes] = []
            for item in body.get("data", []):
                b64 = item.get("b64_json")
                if b64 is not None:
                    images.append(base64.b64decode(b64))
                elif item.get("url"):
                    images.append(_fetch_from_url(item["url"], http_timeout, transport))
                else:
                    raise OpenAIImageError(
                        "OpenAI response contained neither b64_json nor url.",
                        retryable=False,
                    )
            return images

        err = _error_from_response(resp)
        if not err.retryable or attempt == _MAX_ATTEMPTS - 1:
            raise err
        time.sleep(_backoff_seconds(attempt, resp.headers.get("Retry-After")))
        last_error = err

    # Unreachable in practice — the loop always returns or raises — but keeps
    # the function's contract explicit if _MAX_ATTEMPTS is ever set to 0.
    raise last_error or OpenAIImageError("OpenAI image generation failed with no attempts made.")


def validate_api_key(
    *,
    api_key: str,
    timeout: Optional[float] = None,
    transport: Optional[httpx.BaseTransport] = None,
) -> None:
    """
    Pre-flight check (plan §2.4/§3.5/§4.1): a cheap authenticated call (the
    models list, no image generated, no meaningful cost) that raises
    `OpenAIImageError` — mapped the same way as a generation failure — if the
    key is invalid, revoked, or unauthorized for the configured model's
    organization. Raises nothing on success.

    Exactly one HTTP call is made — an invalid key is not a transient
    condition, so this deliberately does not use the retry/backoff loop
    `generate_images` uses for 429/5xx.

    `transport` is test-only: pass an `httpx.MockTransport` to exercise this
    function without a real network call.
    """
    read_timeout = timeout if timeout is not None else settings.OPENAI_IMAGE_TIMEOUT_SECONDS
    http_timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0)
    url = f"{settings.OPENAI_BASE_URL.rstrip('/')}/models"

    try:
        with httpx.Client(timeout=http_timeout, transport=transport) as client:
            resp = client.get(url, headers=_build_headers(api_key))
    except httpx.TimeoutException as exc:
        raise OpenAIImageError(
            f"Timed out contacting OpenAI: {exc}",
            retryable=True,
            user_message="Could not verify the OpenAI API key (timed out).",
        )
    except httpx.TransportError as exc:
        raise OpenAIImageError(
            f"Connection error contacting OpenAI: {exc}",
            retryable=True,
            user_message="Could not reach OpenAI to verify the API key.",
        )

    if resp.status_code == 200:
        return
    raise _error_from_response(resp)
