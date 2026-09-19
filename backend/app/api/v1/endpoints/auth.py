"""
Auth endpoints — register, login, refresh, API keys, Google Sign-In
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime, timedelta
from typing import Optional
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
import hashlib, logging, secrets, threading, time

import redis
import redis.exceptions

from app.core.config import settings
from app.core.database import get_db
from app.core.auth import hash_password, verify_password, create_access_token, get_current_user
from app.core.email import send_verification_email, send_password_reset_email
from app.core.storage import storage
from app.models.user import User, APIKey, Project, Sample
from app.api.v1.endpoints.projects import _purge_project_children

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Verification-resend throttle ─────────────────────────────────────────────
# A 60s per-email cooldown enforced server-side so the resend endpoint can't be
# used to blast SMTP or (combined with timing) probe for accounts. Redis is the
# source of truth (shared across app instances — it's already the Celery
# broker/backend); the in-memory map is only a degraded fallback for when Redis
# is unreachable, and is per-process so it won't coordinate across workers.
_RESEND_COOLDOWN_SECONDS = 60
_resend_memory_cache: dict[str, float] = {}
_resend_memory_lock = threading.Lock()
_redis_client: "redis.Redis | None" = None
_redis_client_ready = False


def _get_redis_client():
    """Lazily build a Redis client from REDIS_URL, memoizing failure so a dead
    Redis doesn't cost a connection attempt on every request."""
    global _redis_client, _redis_client_ready
    if _redis_client_ready:
        return _redis_client
    _redis_client_ready = True
    try:
        _redis_client = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=0.5)
    except Exception as e:  # pragma: no cover - construction rarely raises
        logger.warning(f"Resend throttle: Redis unavailable, using in-memory fallback: {e}")
        _redis_client = None
    return _redis_client


def _resend_allowed(prefix: str, normalized_email: str) -> bool:
    """Atomically claim the 60s cooldown slot for (prefix, email). Returns True if
    the caller may send (slot was free), False if a send already happened within
    the window. Never distinguishes account state."""
    key = f"{prefix}:{normalized_email}"
    client = _get_redis_client()
    if client is not None:
        try:
            # SET NX EX is atomic: only the first caller in the window gets True.
            was_set = client.set(key, "1", nx=True, ex=_RESEND_COOLDOWN_SECONDS)
            return bool(was_set)
        except redis.exceptions.RedisError as e:
            logger.warning(f"Resend throttle: Redis error, falling back to memory: {e}")
    now = time.monotonic()
    with _resend_memory_lock:
        expires = _resend_memory_cache.get(key)
        if expires is not None and expires > now:
            return False
        # Opportunistically drop expired keys so the map can't grow unbounded.
        for k in [k for k, exp in _resend_memory_cache.items() if exp <= now]:
            del _resend_memory_cache[k]
        _resend_memory_cache[key] = now + _RESEND_COOLDOWN_SECONDS
        return True


# ─── Schemas ──────────────────────────────────────────────────────────────────

_GMAIL_DOMAIN = "@gmail.com"


def _require_gmail(value: str) -> str:
    """Reject any address that is not an @gmail.com mailbox (case-insensitive)."""
    if not value.lower().endswith(_GMAIL_DOMAIN):
        raise ValueError("Only @gmail.com email addresses are allowed")
    return value


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleLoginRequest(BaseModel):
    credential: str  # ID token from Google Identity Services


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    email: str = ""
    message: Optional[str] = None
    # True only when this response corresponds to an account that was *just*
    # created (first-ever sign-in). Drives the frontend first-time onboarding
    # tour. Returning users and account-linking always report False.
    is_new_user: bool = False


class RegisterResponse(BaseModel):
    message: str
    email: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    password: str
    confirm_password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class CreateAPIKeyRequest(BaseModel):
    name: str


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: str   # returned only on creation
    created_at: datetime


# ─── Endpoints ────────────────────────────────────────────────────────────────

def _issue_verification_token(user: User) -> str:
    """Generate a verification token, storing only its hash on the user."""
    raw_token = secrets.token_urlsafe(32)
    user.verification_token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    user.verification_token_expires_at = datetime.utcnow() + timedelta(
        seconds=settings.EMAIL_VERIFICATION_TOKEN_TTL_SECONDS
    )
    return raw_token


def _issue_reset_token(user: User) -> str:
    """Generate a password-reset token, storing only its hash on the user."""
    raw_token = secrets.token_urlsafe(32)
    user.reset_token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    user.reset_token_expires_at = datetime.utcnow() + timedelta(
        seconds=settings.PASSWORD_RESET_TOKEN_TTL_SECONDS
    )
    return raw_token


@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email=req.email,
        username=req.username,
        hashed_password=hash_password(req.password),
        is_verified=False,
    )
    raw_token = _issue_verification_token(user)
    db.add(user)
    db.commit()
    db.refresh(user)

    send_verification_email(user.email, raw_token)

    return RegisterResponse(
        message="Registration successful. Check your email to verify your account before logging in.",
        email=user.email,
    )


@router.get("/verify-email", response_model=TokenResponse)
def verify_email(token: str, db: Session = Depends(get_db)):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    user = db.query(User).filter(User.verification_token_hash == token_hash).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    # Idempotency: the token hash is intentionally kept (not nulled) on success below,
    # so a retry/double-invoke with the same token lands here and short-circuits to
    # success instead of "invalid or expired" — without re-validating expiry, since an
    # already-verified account should never flip back to unverified.
    if user.is_verified:
        access_token = create_access_token({"sub": user.id})
        return TokenResponse(
            access_token=access_token,
            user_id=user.id,
            username=user.username,
            email=user.email,
            message="Email already verified. You can now log in.",
        )
    if not user.verification_token_expires_at:
        raise HTTPException(status_code=400, detail="Invalid verification token")
    if user.verification_token_expires_at < datetime.utcnow():
        # Only the holder of this exact token reaches the expired branch, so echoing
        # their own email back (for login prefill) is not account enumeration.
        raise HTTPException(
            status_code=400,
            detail={
                "code": "expired",
                "message": "This verification link has expired. Please request a new one.",
                "email": user.email,
            },
        )

    user.is_verified = True
    db.commit()

    access_token = create_access_token({"sub": user.id})
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        username=user.username,
        email=user.email,
        message="Email verified successfully. You can now log in.",
    )


# Every resend outcome returns this exact string. Unknown email, already-verified,
# unverified-and-sent, and throttled all look identical to the caller so the
# endpoint leaks neither account existence nor verification status.
_RESEND_GENERIC_MESSAGE = "If an account exists for this email, a verification email has been sent."


def _send_verification_email_bg(to_email: str, raw_token: str) -> None:
    """Run send_verification_email in a background task. It logs its own SMTP
    failures but re-raises; a raise here would surface as an unhandled error in
    the background runner (and never reaches the caller anyway), so we catch and
    log to keep the failure visible without noise."""
    try:
        send_verification_email(to_email, raw_token)
    except Exception:
        logger.exception(f"Background verification email failed for {to_email}")


@router.post("/resend-verification")
def resend_verification(
    req: ResendVerificationRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Re-send the verification email for an unverified account, without ever
    revealing which case occurred.

    - Unknown email → do nothing.
    - Already verified → do nothing.
    - Unverified → issue a fresh token (replacing the old) and send the email.

    A 60s server-side throttle (Redis SET NX EX, in-memory fallback) caps this to
    at most one email per email address per minute regardless of account state.
    The email is normalized (strip + lowercase) for both the DB lookup and the
    throttle key so casing/whitespace can't bypass either.

    The SMTP send runs in a FastAPI BackgroundTask (fired after the response),
    so all four outcomes — unknown, already-verified, unverified-sent, and
    throttled — return in constant time and the send latency is no longer a
    timing side-channel for enumeration. Token issuance + commit stay synchronous
    so the new token is durable before we respond.
    """
    normalized_email = req.email.strip().lower()

    # Throttle first: a throttled request must be indistinguishable from a
    # processed one, so we skip all work (including the DB read) and return the
    # generic message. Claiming the slot up front also means the expensive
    # SMTP path runs at most once per window.
    if not _resend_allowed("verification_resend", normalized_email):
        return {"message": _RESEND_GENERIC_MESSAGE}

    user = db.query(User).filter(User.email == normalized_email).first()
    if user and not user.is_verified:
        raw_token = _issue_verification_token(user)
        db.commit()
        background.add_task(_send_verification_email_bg, user.email, raw_token)

    return {"message": _RESEND_GENERIC_MESSAGE}


_FORGOT_PASSWORD_GENERIC_MESSAGE = (
    "If an account exists for this email, a password reset link has been sent."
)


def _send_reset_email_bg(to_email: str, raw_token: str) -> None:
    try:
        send_password_reset_email(to_email, raw_token)
    except Exception:
        logger.exception(f"Background password reset email failed for {to_email}")


@router.post("/forgot-password")
def forgot_password(
    req: ForgotPasswordRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Request a password reset link.

    Always returns the same generic message regardless of whether the email
    exists, belongs to a Google-only account, or hit the throttle — this
    prevents account enumeration.

    Eligibility: the user must have a local password (hashed_password is not
    None). Google-only accounts have no local password and are excluded.

    The email is fired via BackgroundTask (after response) so all outcomes
    return in constant time with no SMTP timing side-channel.
    """
    normalized_email = req.email.strip().lower()

    if not _resend_allowed("password_reset", normalized_email):
        return {"message": _FORGOT_PASSWORD_GENERIC_MESSAGE}

    user = db.query(User).filter(User.email == normalized_email).first()
    if user and user.hashed_password is not None:
        raw_token = _issue_reset_token(user)
        db.commit()
        background.add_task(_send_reset_email_bg, user.email, raw_token)

    return {"message": _FORGOT_PASSWORD_GENERIC_MESSAGE}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Consume a single-use reset token and set a new password.

    Rejects mismatched passwords, invalid tokens, and expired tokens with
    400. Clears the token fields on success so the link cannot be reused.
    Does not issue a session — the user must log in normally afterward.
    """
    if not req.password:
        raise HTTPException(status_code=400, detail="Password must not be empty")
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    token_hash = hashlib.sha256(req.token.encode()).hexdigest()
    user = db.query(User).filter(User.reset_token_hash == token_hash).first()
    if (
        not user
        or not user.reset_token_expires_at
        or user.reset_token_expires_at < datetime.utcnow()
    ):
        raise HTTPException(
            status_code=400, detail="Invalid or expired password reset link"
        )

    user.hashed_password = hash_password(req.password)
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    db.commit()
    return {"message": "Password has been reset successfully. You can now log in."}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.hashed_password or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Email not verified")

    token = create_access_token({"sub": user.id})
    return TokenResponse(access_token=token, user_id=user.id, username=user.username, email=user.email)


def _unique_username_from_email(db: Session, email: str) -> str:
    base = email.split("@")[0] or "user"
    username = base
    suffix = 0
    while db.query(User).filter(User.username == username).first():
        suffix += 1
        username = f"{base}{suffix}"
    return username


@router.post("/google", response_model=TokenResponse)
def google_login(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google Sign-In is not configured")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            req.credential, google_requests.Request(), settings.GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10,
        )
    except ValueError as e:
        logger.warning(f"Google ID token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid Google token")

    if not idinfo.get("email_verified"):
        raise HTTPException(status_code=401, detail="Google email is not verified")

    email = idinfo["email"]
    try:
        _require_gmail(email)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    google_sub = idinfo["sub"]
    # Tracks whether this request created a brand-new account (vs. matching an
    # existing google_sub or linking Google onto an existing password account).
    # Only a freshly inserted User is a "new user" for onboarding purposes.
    is_new_user = False
    user = db.query(User).filter(User.google_sub == google_sub).first()
    if not user:
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.google_sub = google_sub  # link existing password account
        else:
            user = User(
                email=email,
                username=_unique_username_from_email(db, email),
                hashed_password=None,
                google_sub=google_sub,
                is_verified=True,  # Google's email_verified claim already covers this
            )
            db.add(user)
            is_new_user = True
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token({"sub": user.id})
    return TokenResponse(
        access_token=token, user_id=user.id, username=user.username,
        email=user.email, is_new_user=is_new_user,
    )


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "role": current_user.role,
        "created_at": current_user.created_at,
        "has_password": current_user.hashed_password is not None,
    }


@router.post("/password")
def change_password(
    req: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.hashed_password is None:
        raise HTTPException(
            status_code=400,
            detail="This account uses Google Sign-In and has no password to change.",
        )
    if not verify_password(req.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    if req.new_password == req.current_password:
        raise HTTPException(
            status_code=400, detail="New password must differ from the current one."
        )
    current_user.hashed_password = hash_password(req.new_password)
    db.commit()
    return {"message": "Password updated successfully."}


@router.delete("/me", status_code=204)
def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Permanently delete the caller's account and everything they own.

    Order matters:
      1) Purge object-storage blobs for each owned project. Every key is under
         ``projects/<project_id>/...``, so listing that prefix and deleting
         each key clears samples, features.npz, and model files. Done first so
         a storage failure aborts before any DB row is removed (nothing is
         committed on error), rather than orphaning blobs with no DB pointer.
      2) Close the same project-child FK gaps ``delete_project`` handles, via
         the shared ``_purge_project_children`` helper, then remove each
         project row (ORM cascade takes the configured children).
      3) Null ``Sample.uploaded_by`` for any samples that reference this user
         but live outside the deleted projects — it is a nullable FK to
         users.id with no cascade, so a stray reference would block the delete.
      4) Delete the user row. ``APIKey`` rows go via the
         User.api_keys cascade="all, delete-orphan" relationship.
    """
    projects = db.query(Project).filter(Project.owner_id == current_user.id).all()
    for p in projects:
        for key in storage.list_files(f"projects/{p.id}/"):
            storage.delete_file(key)
        _purge_project_children(db, p.id)
        db.delete(p)

    db.query(Sample).filter(Sample.uploaded_by == current_user.id).update(
        {Sample.uploaded_by: None}, synchronize_session=False
    )

    db.delete(current_user)
    db.commit()
    # 204 must carry no body. Returning None would let the default JSONResponse
    # serialize `null` (content-length 4) onto a no-content status, which h11
    # rejects → the client sees a network error despite the commit succeeding.
    return Response(status_code=204)


@router.post("/api-keys", response_model=APIKeyResponse)
def create_api_key(
    req: CreateAPIKeyRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    raw_key = f"ef_{secrets.token_hex(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = APIKey(
        user_id=current_user.id,
        name=req.name,
        key_hash=key_hash,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return APIKeyResponse(id=api_key.id, name=api_key.name, key=raw_key, created_at=api_key.created_at)


@router.get("/api-keys")
def list_api_keys(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id, APIKey.is_active == True).all()
    return [{"id": k.id, "name": k.name, "created_at": k.created_at, "last_used": k.last_used} for k in keys]


@router.delete("/api-keys/{key_id}", status_code=204)
def revoke_api_key(
    key_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    key = db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    db.commit()
    return Response(status_code=204)
