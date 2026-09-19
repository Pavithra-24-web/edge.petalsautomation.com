"""
Motion API router aggregate — Batch 3.1.

Mounted by H1 (`app/api/v1/__init__.py`) at prefix "/devices", alongside the
existing shared `devices.router` — same prefix, disjoint paths, so motion
routes sit beside `endpoints/devices.py`'s snapshot/inference streams without
forking that module.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.motion.api.streams import router as streams_router

router = APIRouter()
router.include_router(streams_router)
