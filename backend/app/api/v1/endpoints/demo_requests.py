"""
Public "Book a Demo" lead capture — no auth.

A prospective user submits their contact details from the marketing site; we
email the lead to our own inbox (reusing the SMTP transport in app.core.email).
This is intentionally unauthenticated, mirroring the unauthenticated auth
endpoints (register / forgot-password) in this package.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, field_validator

from app.core.email import send_demo_request_email

logger = logging.getLogger(__name__)

router = APIRouter()


class DemoRequest(BaseModel):
    name: str
    mobile: str
    email: EmailStr
    description: str
    source: str  # the "How did you hear about us?" dropdown value
    company: str | None = None

    @field_validator("name", "mobile", "description", "source")
    @classmethod
    def _required_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("This field is required")
        return v

    @field_validator("company")
    @classmethod
    def _trim_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


@router.post("/demo-requests")
def create_demo_request(req: DemoRequest):
    try:
        send_demo_request_email(req.model_dump())
    except Exception:
        # SMTP failures are logged server-side; the client gets a generic error
        # so we never leak transport/config details.
        logger.exception("Failed to send demo request email")
        raise HTTPException(
            status_code=500,
            detail="Could not submit your request right now. Please try again later.",
        )
    return {"status": "ok"}
