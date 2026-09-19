"""
Public "Contact us" lead capture — no auth.

A visitor picks a category (Sales inquiries / Technical support / Product demo /
Product feedback) and submits their details from the marketing site; we email the
message to our own inbox (reusing the SMTP transport in app.core.email). This is
intentionally unauthenticated, mirroring the "Book a Demo" flow in demo_requests.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, field_validator

from app.core.email import send_contact_request_email

logger = logging.getLogger(__name__)

router = APIRouter()

_CATEGORIES = {
    "Sales inquiries",
    "Technical support",
    "Product demo",
    "Product feedback",
}


class ContactRequest(BaseModel):
    category: str
    name: str
    mobile: str
    email: EmailStr
    company: str
    description: str
    job_title: str | None = None

    @field_validator("name", "mobile", "company", "description")
    @classmethod
    def _required_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("This field is required")
        return v

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("This field is required")
        if v not in _CATEGORIES:
            raise ValueError("Invalid category")
        return v

    @field_validator("job_title")
    @classmethod
    def _trim_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


@router.post("/contact-requests")
def create_contact_request(req: ContactRequest):
    try:
        send_contact_request_email(req.model_dump())
    except Exception:
        # SMTP failures are logged server-side; the client gets a generic error
        # so we never leak transport/config details.
        logger.exception("Failed to send contact request email")
        raise HTTPException(
            status_code=500,
            detail="Could not submit your request right now. Please try again later.",
        )
    return {"status": "ok"}
