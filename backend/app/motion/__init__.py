"""
Motion business logic — Phase 1 (motion_phase1.md §2).

Everything specific to motion/gesture recognition lives under this package.
Shared infrastructure (auth, the device WebSocket, StreamStore, sampling,
ingestion, Celery, ...) is reused via a small set of integration hooks in
the existing modules — never duplicated here. Reached by WS message type,
route, or a device's own declared capability; never by `Project.project_type`
(see motion_phase1.md §2.3 and Phase 0 acceptance criterion 13).

Sub-phase 1.1: package skeleton + sensor inventory normalisation only.
"""
