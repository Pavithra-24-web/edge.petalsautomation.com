"""Edge-Impulse-style download filename builder for .pxe packages.

The canonical pattern is:

    {project-name}-{impulse-name}-v{deploy-version}.pxe

where each name component is slugified to lowercase ASCII letters, digits and
single hyphens (any other character becomes a hyphen; runs of hyphens collapse;
leading and trailing hyphens strip).
"""
from __future__ import annotations

import re

__all__ = ["slugify", "build_pxe_download_filename"]

_NON_SLUG_CHAR_RE = re.compile(r"[^a-z0-9-]+")
_MULTI_HYPHEN_RE = re.compile(r"-{2,}")


def slugify(value: str) -> str:
    """Lowercase, replace non-[a-z0-9-] runs with '-', collapse, strip."""
    if value is None:
        return ""
    s = str(value).lower()
    s = _NON_SLUG_CHAR_RE.sub("-", s)
    s = _MULTI_HYPHEN_RE.sub("-", s)
    return s.strip("-")


def build_pxe_download_filename(
    project_name: str,
    impulse_name: str,
    deploy_version: int,
) -> str:
    """Return the canonical EI-style .pxe download filename."""
    project_slug = slugify(project_name) or "project"
    impulse_slug = slugify(impulse_name) or "impulse"
    version = int(deploy_version)
    if version < 1:
        version = 1
    return f"{project_slug}-{impulse_slug}-v{version}.pxe"
