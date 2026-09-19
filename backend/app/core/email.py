"""
Transactional email — Gmail SMTP via smtplib.
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core import email_templates as templates
from app.core.config import settings

logger = logging.getLogger(__name__)


def _send(
    subject: str,
    to: str,
    text: str,
    html: str,
    reply_to: str | None = None,
) -> None:
    """Send one multipart/alternative email.

    Images are never attached: any attached part shows up in Gmail's attachment
    strip whether or not it also renders inline, so artwork is referenced by
    public https URL instead (see ``settings.EMAIL_LOGO_URL``).
    """
    message = MIMEMultipart("alternative")
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    message["Subject"] = subject
    message["From"] = settings.SMTP_FROM
    message["To"] = to
    if reply_to:
        message["Reply-To"] = reply_to
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM, [to], message.as_string())
    except smtplib.SMTPException as e:
        logger.error(f"Failed to send email to {to}: {e}")
        raise


def send_verification_email(to_email: str, token: str) -> None:
    verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    ttl_hours = settings.EMAIL_VERIFICATION_TOKEN_TTL_SECONDS // 3600
    _send(
        subject="Verify your Petal Edge email address",
        to=to_email,
        text=templates.verification_email_text(verify_url, ttl_hours),
        html=templates.verification_email_html(
            verify_url, ttl_hours, settings.EMAIL_LOGO_URL
        ),
    )


def send_demo_request_email(payload: dict) -> None:
    """Notify our own inbox of a new public "Book a Demo" lead.

    Reuses the same SMTP transport as the account emails. Reply-To is set to the
    submitter so a reply from the inbox goes straight to the lead. This is a
    one-off notification — it does NOT touch the account verification flow.
    """
    name = payload["name"]
    company = payload.get("company") or "—"

    def _row(label: str, value: str) -> str:
        return f"{label}: {value}"

    text = "\n".join(
        [
            "New demo request received.",
            "",
            _row("Name", name),
            _row("Mobile", payload["mobile"]),
            _row("Email", payload["email"]),
            _row("Company / Institution", company),
            _row("Heard about us via", payload["source"]),
            "",
            "Description / use case:",
            payload["description"],
        ]
    )

    def _esc(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    html = (
        "<h2>New demo request</h2>"
        "<table cellpadding='4' style='border-collapse:collapse'>"
        f"<tr><td><strong>Name</strong></td><td>{_esc(name)}</td></tr>"
        f"<tr><td><strong>Mobile</strong></td><td>{_esc(payload['mobile'])}</td></tr>"
        f"<tr><td><strong>Email</strong></td><td>{_esc(payload['email'])}</td></tr>"
        f"<tr><td><strong>Company / Institution</strong></td><td>{_esc(company)}</td></tr>"
        f"<tr><td><strong>Heard about us via</strong></td><td>{_esc(payload['source'])}</td></tr>"
        "</table>"
        "<p><strong>Description / use case:</strong></p>"
        f"<p>{_esc(payload['description'])}</p>"
    )

    _send(
        subject=f"New demo request — {name}",
        to=settings.DEMO_NOTIFY_EMAIL,
        text=text,
        html=html,
        reply_to=payload["email"],
    )


def send_contact_request_email(payload: dict) -> None:
    """Notify our own inbox of a new public "Contact us" message.

    Reuses the same SMTP transport as the demo requests and account emails, and
    lands in the same inbox (settings.DEMO_NOTIFY_EMAIL). Reply-To is always set
    to the submitter's (compulsory) email, so a reply from the inbox goes
    straight back to them. This is a one-off notification only.
    """
    category = payload["category"]
    name = payload["name"]
    email = payload["email"]
    company = payload["company"]
    job_title = payload.get("job_title") or "—"

    # Subject is driven by the category, then prefixed with the sender's name.
    if category == "Product feedback":
        subject_base = "Feedback"
    elif category == "Technical support":
        subject_base = "Technical support"
    else:
        subject_base = category

    # The free-text block is "Feedback" for feedback, "Description" otherwise.
    block_label = "Feedback" if category == "Product feedback" else "Description"

    def _row(label: str, value: str) -> str:
        return f"{label}: {value}"

    text = "\n".join(
        [
            "New contact request received.",
            "",
            _row("Category", category),
            _row("Name", name),
            _row("Mobile", payload["mobile"]),
            _row("Email", email),
            _row("Company / Institution", company),
            _row("Job title", job_title),
            "",
            f"{block_label}:",
            payload["description"],
        ]
    )

    def _esc(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    html = (
        "<h2>New contact request</h2>"
        "<table cellpadding='4' style='border-collapse:collapse'>"
        f"<tr><td><strong>Category</strong></td><td>{_esc(category)}</td></tr>"
        f"<tr><td><strong>Name</strong></td><td>{_esc(name)}</td></tr>"
        f"<tr><td><strong>Mobile</strong></td><td>{_esc(payload['mobile'])}</td></tr>"
        f"<tr><td><strong>Email</strong></td><td>{_esc(email)}</td></tr>"
        f"<tr><td><strong>Company / Institution</strong></td><td>{_esc(company)}</td></tr>"
        f"<tr><td><strong>Job title</strong></td><td>{_esc(job_title)}</td></tr>"
        "</table>"
        f"<p><strong>{block_label}:</strong></p>"
        f"<p>{_esc(payload['description'])}</p>"
    )

    _send(
        subject=f"{subject_base} — {name}",
        to=settings.DEMO_NOTIFY_EMAIL,
        text=text,
        html=html,
        reply_to=email,
    )


def send_password_reset_email(to_email: str, token: str) -> None:
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    ttl_minutes = settings.PASSWORD_RESET_TOKEN_TTL_SECONDS // 60
    _send(
        subject="Reset your PetalEdge password",
        to=to_email,
        text=(
            f"You requested a password reset for your PetalEdge account.\n\n"
            f"Reset your password here:\n{reset_url}\n\n"
            f"This link expires in {ttl_minutes} minutes. "
            f"If you did not request this, you can safely ignore this email."
        ),
        html=(
            f"<p>You requested a password reset for your PetalEdge account.</p>"
            f"<p>Click the link below to choose a new password:</p>"
            f'<p><a href="{reset_url}">{reset_url}</a></p>'
            f"<p>This link expires in {ttl_minutes} minutes. "
            f"If you did not request this, you can safely ignore this email.</p>"
        ),
    )
