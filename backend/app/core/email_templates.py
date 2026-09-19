"""
Branded HTML for the transactional emails.

Everything here is written to survive real mail clients, which is why it looks
nothing like the app's React code:

* Layout is nested ``<table>`` -- Gmail, Outlook and Yahoo all strip or ignore
  flexbox and grid.
* Styles are inline on every element. The one ``<style>`` block carries only the
  media queries, which clients that support them honour and the rest drop
  harmlessly (the desktop widths are already fluid).
* No SVG, and no ``cid:`` attachment either. Gmail lists every attached part in
  the attachment strip even when it also renders inline, so an embedded logo
  always shows up as "1 attachment". The brand mark is therefore a plain
  ``https`` image when ``EMAIL_LOGO_URL`` is configured, and a text-only
  wordmark when it is not (dev, where no public URL exists).
* The CTA gradient is paired with a solid ``background-color`` fallback and a
  VML ``roundrect`` for Outlook's Word rendering engine, which supports neither
  gradients nor border-radius.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Design tokens, kept as constants so the templates read from one place.
# ---------------------------------------------------------------------------
PAGE_BG = "#f4f3fb"
CARD_BG = "#ffffff"
CARD_BORDER = "#eae7f6"
INK = "#0f172a"
BODY_TEXT = "#4b5563"
MUTED = "#8a8fa3"
ACCENT = "#6d28d9"
TINT_BG = "#faf9fe"
TINT_BORDER = "#ece9fa"
GRAD_FROM = "#9333ea"
GRAD_TO = "#2563eb"
GRAD_SOLID = "#6d28d9"  # flat fallback wherever linear-gradient is unsupported

FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Helvetica, Arial, sans-serif"
)


def escape(value: str) -> str:
    """Minimal HTML escaping for values interpolated into a template."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _head(preheader: str) -> str:
    # Mobile trim only: the desktop widths are already fluid, so the query
    # tightens padding and steps the type down one notch. The dark-mode rule
    # pins the card surface for clients that would otherwise invert it.
    styles = (
        "@media only screen and (max-width:600px){"
        ".pe-card{border-radius:16px!important}"
        ".pe-pad{padding:34px 24px 30px 24px!important}"
        ".pe-h1{font-size:23px!important;line-height:31px!important}"
        ".pe-lead{font-size:15px!important;line-height:24px!important}"
        ".pe-gutter{padding:28px 16px!important}"
        "}"
        "@media (prefers-color-scheme:dark){"
        f".pe-card{{background-color:{CARD_BG}!important}}"
        "}"
    )
    return (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="en">'
        "<head>"
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        '<meta name="x-apple-disable-message-reformatting"/>'
        '<meta name="color-scheme" content="light"/>'
        '<meta name="supported-color-schemes" content="light"/>'
        "<title>Petal Edge</title>"
        "<!--[if mso]><xml><o:OfficeDocumentSettings>"
        "<o:PixelsPerInch>96</o:PixelsPerInch>"
        "</o:OfficeDocumentSettings></xml><![endif]-->"
        f"<style>{styles}</style>"
        "</head>"
        # Preheader: the grey snippet clients show next to the subject line.
        # Hidden in the body itself by the usual zero-size trick, then padded
        # with invisible characters so the client does not spill message text
        # into the preview after it.
        f'<body style="margin:0;padding:0;width:100%;background-color:{PAGE_BG};'
        '-webkit-font-smoothing:antialiased;-webkit-text-size-adjust:100%;'
        '-ms-text-size-adjust:100%;">'
        '<div style="display:none;font-size:1px;line-height:1px;max-height:0;'
        'max-width:0;opacity:0;overflow:hidden;mso-hide:all;">'
        f"{escape(preheader)}"
        + "&#847;&zwnj;&nbsp;" * 30
        + "</div>"
    )


def _brand_lockup(logo_url: str = "") -> str:
    """Wordmark, optionally preceded by the brand mark.

    ``logo_url`` must be publicly fetchable over https for Google's image proxy
    to load it. When it is empty the image cell is dropped entirely rather than
    emitted with a dead src, so the header degrades to a clean wordmark instead
    of a broken-image glyph.
    """
    mark = (
        '<td style="padding-right:12px;line-height:0;">'
        f'<img src="{escape(logo_url)}" width="40" height="40" alt="Petal Edge" '
        'style="display:block;width:40px;height:40px;border:0;border-radius:10px;"/>'
        "</td>"
    ) if logo_url else ""
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0">'
        "<tr>"
        + mark
        + f'<td style="font-family:{FONT};font-size:21px;line-height:40px;'
        f'font-weight:700;letter-spacing:-0.4px;color:{INK};white-space:nowrap;">'
        "Petal Edge"
        "</td>"
        "</tr>"
        "</table>"
    )


def _footer() -> str:
    return (
        f'<p style="margin:0 0 6px 0;font-family:{FONT};font-size:13px;'
        f'line-height:20px;color:{MUTED};">'
        "&copy; 2026 Petal Edge &middot; Secure Edge AI Platform"
        "</p>"
        f'<p style="margin:0;font-family:{FONT};font-size:12px;line-height:18px;'
        f'color:{MUTED};">'
        "This is an automated message &mdash; please don&rsquo;t reply."
        "</p>"
    )


def _shell(preheader: str, card_html: str, logo_url: str = "") -> str:
    """Wrap card content in the page chrome: background, lockup, card, footer."""
    return (
        _head(preheader)
        + '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'width="100%" style="background-color:{PAGE_BG};">'
        "<tr>"
        '<td align="center" class="pe-gutter" style="padding:40px 20px;">'
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        'width="560" style="width:560px;max-width:560px;">'
        "<tr>"
        '<td align="center" style="padding:0 0 28px 0;">'
        + _brand_lockup(logo_url)
        + "</td>"
        "</tr>"
        "<tr>"
        f'<td class="pe-card" style="background-color:{CARD_BG};'
        f'border:1px solid {CARD_BORDER};border-radius:20px;overflow:hidden;'
        'box-shadow:0 1px 2px rgba(15,23,42,0.04),0 14px 36px rgba(76,29,149,0.07);">'
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">'
        # Gradient hairline along the card's top edge -- the whole design's
        # single decorative flourish.
        "<tr>"
        f'<td height="4" style="height:4px;line-height:4px;font-size:0;'
        f'background-color:{GRAD_SOLID};'
        f'background-image:linear-gradient(90deg,{GRAD_FROM} 0%,{GRAD_TO} 100%);">'
        "&nbsp;</td>"
        "</tr>"
        "<tr>"
        '<td class="pe-pad" style="padding:44px 48px 38px 48px;">'
        + card_html
        + "</td>"
        "</tr>"
        "</table>"
        "</td>"
        "</tr>"
        "<tr>"
        '<td align="center" style="padding:26px 12px 0 12px;">'
        + _footer()
        + "</td>"
        "</tr>"
        "</table>"
        "</td>"
        "</tr>"
        "</table>"
        "</body></html>"
    )


def _button(url: str, label: str) -> str:
    """Full-width gradient CTA, with a VML twin for Outlook."""
    safe_url = escape(url)
    safe_label = escape(label)
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">'
        "<tr>"
        '<td align="center">'
        "<!--[if mso]>"
        '<v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:w="urn:schemas-microsoft-com:office:word" '
        f'href="{safe_url}" style="height:52px;v-text-anchor:middle;width:464px;" '
        f'arcsize="23%" stroke="f" fillcolor="{GRAD_SOLID}">'
        "<w:anchorlock/>"
        f'<center style="color:#ffffff;font-family:{FONT};font-size:16px;'
        f'font-weight:600;">{safe_label}</center>'
        "</v:roundrect>"
        "<![endif]-->"
        "<!--[if !mso]><!-- -->"
        f'<a href="{safe_url}" '
        'style="display:block;width:100%;box-sizing:border-box;padding:16px 24px;'
        f'font-family:{FONT};font-size:16px;line-height:20px;font-weight:600;'
        'letter-spacing:-0.1px;color:#ffffff;text-decoration:none;text-align:center;'
        f'border-radius:12px;background-color:{GRAD_SOLID};'
        f'background-image:linear-gradient(90deg,{GRAD_FROM} 0%,{GRAD_TO} 100%);'
        'box-shadow:0 6px 16px rgba(109,40,217,0.28);">'
        f"{safe_label}"
        "</a>"
        "<!--<![endif]-->"
        "</td>"
        "</tr>"
        "</table>"
    )


def _divider() -> str:
    return (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">'
        "<tr>"
        f'<td height="1" style="height:1px;line-height:1px;font-size:0;'
        f'background-color:{CARD_BORDER};">&nbsp;</td>'
        "</tr>"
        "</table>"
    )


def _fallback_link(url: str) -> str:
    """The 'button not working?' block: label plus the raw, wrappable URL."""
    safe_url = escape(url)
    return (
        _divider()
        + f'<p style="margin:26px 0 12px 0;font-family:{FONT};font-size:13px;'
        f'line-height:20px;color:{MUTED};">'
        "Button not working? Paste this link into your browser:"
        "</p>"
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" '
        f'style="background-color:{TINT_BG};border:1px solid {TINT_BORDER};'
        'border-radius:10px;">'
        "<tr>"
        '<td style="padding:14px 16px;">'
        f'<a href="{safe_url}" '
        f'style="font-family:{FONT};font-size:13px;line-height:20px;color:{ACCENT};'
        'text-decoration:none;word-break:break-all;overflow-wrap:break-word;">'
        f"{safe_url}"
        "</a>"
        "</td>"
        "</tr>"
        "</table>"
    )


def _hours(ttl_hours: int) -> str:
    return f"{ttl_hours} hour" + ("" if ttl_hours == 1 else "s")


def verification_email_html(
    verify_url: str, ttl_hours: int, logo_url: str = ""
) -> str:
    """The "verify your email address" message."""
    card = (
        f'<h1 class="pe-h1" style="margin:0 0 12px 0;font-family:{FONT};'
        f'font-size:27px;line-height:35px;font-weight:700;letter-spacing:-0.6px;'
        f'color:{INK};text-align:center;">'
        "Verify your email address"
        "</h1>"
        f'<p class="pe-lead" style="margin:0 0 28px 0;font-family:{FONT};'
        f'font-size:16px;line-height:26px;color:{BODY_TEXT};text-align:center;">'
        "Thanks for creating your Petal Edge account. Confirm this address to "
        "activate it and start building."
        "</p>"
        + _button(verify_url, "Verify email")
        + f'<p style="margin:16px 0 30px 0;font-family:{FONT};font-size:13px;'
        f'line-height:20px;color:{MUTED};text-align:center;">'
        f"This link expires in {_hours(ttl_hours)}."
        "</p>"
        + _fallback_link(verify_url)
        + f'<p style="margin:20px 0 0 0;font-family:{FONT};font-size:13px;'
        f'line-height:20px;color:{MUTED};">'
        "If you didn&rsquo;t create a Petal Edge account, you can safely ignore "
        "this email."
        "</p>"
    )
    return _shell(
        "Confirm your email address to activate your Petal Edge account.",
        card,
        logo_url,
    )


def verification_email_text(verify_url: str, ttl_hours: int) -> str:
    """Plain-text twin of :func:`verification_email_html`."""
    return (
        "PETAL EDGE\n\n"
        "Verify your email address\n"
        "-------------------------\n\n"
        "Thanks for creating your Petal Edge account. Confirm this address to "
        "activate it and start building.\n\n"
        f"{verify_url}\n\n"
        f"This link expires in {_hours(ttl_hours)}.\n\n"
        "If you didn't create a Petal Edge account, you can safely ignore this "
        "email.\n\n"
        "-- \n"
        "(c) 2026 Petal Edge - Secure Edge AI Platform\n"
        "This is an automated message - please don't reply.\n"
    )
