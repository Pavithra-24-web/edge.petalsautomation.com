"""
One-shot script: reassign 14 samples from composite labels (cat,dog / dog,cat)
to their correct single-name labels, then delete the composite label rows.

Project:  7b88dd54-f16e-4445-8b0d-f9f8f3f5b7e4
Bad label aa2c7253-f5ea-4d04-9bc4-11bee2f39ee4  "cat,dog"  → reassign to "cat"
Bad label 982941ae-7557-466b-8ae9-70e6f956fd2e  "dog,cat"  → reassign to "dog"

Run from the backend/ directory:
    PYTHONPATH=. python scripts/fix_composite_labels.py          # Linux/Mac
    $env:PYTHONPATH="."; python scripts/fix_composite_labels.py  # PowerShell
"""

import sys
import os

# Allow running from the backend/ directory without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.core.database import SessionLocal
from app.models.user import Label, Sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ID   = "7b88dd54-f16e-4445-8b0d-f9f8f3f5b7e4"
BAD_CAT_DOG  = "aa2c7253-f5ea-4d04-9bc4-11bee2f39ee4"  # "cat,dog"
BAD_DOG_CAT  = "982941ae-7557-466b-8ae9-70e6f956fd2e"  # "dog,cat"

REPLACEMENTS = {
    BAD_CAT_DOG: "cat",
    BAD_DOG_CAT: "dog",
}


def main() -> None:
    db = SessionLocal()
    try:
        # Resolve the real single-name label IDs in this project
        target_ids: dict[str, str] = {}
        for canonical in ("cat", "dog"):
            row = (
                db.query(Label)
                .filter(Label.project_id == PROJECT_ID, Label.name == canonical)
                .first()
            )
            if row is None:
                log.error("Label %r not found in project — aborting.", canonical)
                sys.exit(1)
            target_ids[canonical] = str(row.id)
            log.info("Resolved label %r → %s", canonical, row.id)

        reassigned_ids: list[str] = []

        for bad_id, canonical in REPLACEMENTS.items():
            good_id = target_ids[canonical]
            samples = (
                db.query(Sample)
                .filter(Sample.label_id == bad_id)
                .all()
            )
            for s in samples:
                s.label_id = good_id
                reassigned_ids.append(str(s.id))
            log.info(
                "Reassigned %d sample(s) from %r (%s) → %s",
                len(samples), bad_id, REPLACEMENTS[bad_id], good_id,
            )

        db.flush()

        # Now safe to delete the composite labels (no FK references remain)
        for bad_id in REPLACEMENTS:
            label = db.query(Label).filter(Label.id == bad_id).first()
            if label:
                db.delete(label)
                log.info("Deleted composite label %r (%r)", bad_id, label.name)
            else:
                log.warning("Composite label %r already absent — skipping.", bad_id)

        db.commit()

        log.info("Done. %d sample(s) reassigned:", len(reassigned_ids))
        for sid in reassigned_ids:
            log.info("  %s", sid)

    except Exception:
        db.rollback()
        log.exception("Migration failed — rolled back.")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
