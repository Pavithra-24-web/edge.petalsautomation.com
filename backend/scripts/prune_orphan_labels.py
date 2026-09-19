"""One-shot backfill: prune orphan labels across every project.

A label is "orphan" when no sample's `label_id` and no sample's bounding-box
annotation references it (either by id or by name). The same logic backs the
`POST /labels/project/{project_id}/prune-orphans` endpoint — this script reuses
that function so behavior stays in sync.

Run from the backend/ directory:
    PYTHONPATH=. python scripts/prune_orphan_labels.py          # Linux/Mac
    $env:PYTHONPATH="."; python scripts/prune_orphan_labels.py  # PowerShell
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.core.database import SessionLocal
from app.models.user import Project
from app.api.v1.endpoints.labels import _prune_orphans_in_project

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    db = SessionLocal()
    try:
        projects = db.query(Project).all()
        log.info("Scanning %d project(s) for orphan labels...", len(projects))
        total = 0
        for p in projects:
            deleted = _prune_orphans_in_project(db, p.id)
            if deleted:
                names = ", ".join(repr(d["name"]) for d in deleted)
                log.info("  [%s] deleted %d orphan(s): %s", p.id, len(deleted), names)
                total += len(deleted)
        log.info("Done. Deleted %d orphan label(s) total.", total)
    except Exception:
        db.rollback()
        log.exception("Backfill failed — rolled back.")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
