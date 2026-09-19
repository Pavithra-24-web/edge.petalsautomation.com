"""Labels endpoints"""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from pydantic import BaseModel
from typing import Optional
from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import assert_project_owner, assert_label_owner
from app.models.user import User, Label, Sample, AIPrediction

router = APIRouter()

class LabelCreate(BaseModel):
    project_id: str
    name: str
    color: Optional[str] = "#3B8BD4"

@router.post("/", status_code=201)
def create_label(req: LabelCreate, db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    if "," in req.name:
        raise HTTPException(422, "Label name must not contain commas")
    assert_project_owner(db, req.project_id, current_user)
    label = Label(project_id=req.project_id, name=req.name, color=req.color)
    db.add(label); db.commit(); db.refresh(label)
    return {"id": label.id, "name": label.name, "color": label.color,
            "project_id": label.project_id, "created_at": label.created_at.isoformat()}

@router.get("/project/{project_id}")
def list_labels(project_id: str, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    assert_project_owner(db, project_id, current_user)
    labels = db.query(Label).filter(Label.project_id == project_id).all()
    return [{"id": l.id, "name": l.name, "color": l.color} for l in labels]

@router.delete("/{label_id}", status_code=204)
def delete_label(label_id: str, db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    l = assert_label_owner(db, label_id, current_user)
    # Null out any sample top-level labels that reference this label so the
    # FK constraint doesn't block the delete (PostgreSQL enforces it strictly).
    # Samples keep their data and bounding-box annotations; only the top-level
    # classification label assignment is cleared.
    affected = (
        db.query(Sample)
        .filter(Sample.label_id == label_id)
        .update({"label_id": None}, synchronize_session="fetch")
    )
    if affected:
        import logging
        logging.getLogger(__name__).info(
            f"delete_label: cleared label_id on {affected} sample(s) before deleting label {label_id!r}."
        )

    # Also scrub any stored bounding-box annotations that still reference the
    # deleted label by UUID or display name. Otherwise old boxes can keep
    # surfacing as ghost classes in Generate Features and training flows.
    target_name = (l.name or "").strip().lower()
    samples = db.query(Sample).filter(Sample.project_id == l.project_id).all()
    for sample in samples:
        meta = sample.extra_metadata or {}
        boxes = meta.get("boundingBoxes")
        if not isinstance(boxes, list):
            continue
        changed = False
        new_boxes = []
        for box in boxes:
            if not isinstance(box, dict):
                new_boxes.append(box)
                continue
            b = dict(box)
            box_label_id = str(b.get("label_id") or "")
            box_label_name = str(b.get("label") or "").strip().lower()
            if box_label_id == label_id or (target_name and box_label_name == target_name):
                b["label"] = None
                b["label_id"] = None
                changed = True
            new_boxes.append(b)
        if changed:
            meta = dict(meta)
            meta["boundingBoxes"] = new_boxes
            sample.extra_metadata = meta
            flag_modified(sample, "extra_metadata")

    # Pending/applied AI predictions can also carry the deleted label string and
    # otherwise recreate it when approved later.
    predictions = (
        db.query(AIPrediction)
        .join(Sample, Sample.id == AIPrediction.sample_id)
        .filter(Sample.project_id == l.project_id)
        .all()
    )
    for pred in predictions:
        changed = False
        if str(pred.predicted_label or "").strip().lower() == target_name:
            pred.predicted_label = None
            changed = True
        boxes = pred.bounding_boxes or []
        new_boxes = []
        for box in boxes:
            if not isinstance(box, dict):
                new_boxes.append(box)
                continue
            b = dict(box)
            box_label_id = str(b.get("label_id") or "")
            box_label_name = str(b.get("label") or "").strip().lower()
            if box_label_id == label_id or (target_name and box_label_name == target_name):
                b["label"] = None
                b["label_id"] = None
                changed = True
            new_boxes.append(b)
        if changed:
            pred.bounding_boxes = new_boxes

    db.delete(l); db.commit()
    return Response(status_code=204)


def _prune_orphans_in_project(db: Session, project_id: str) -> list[dict]:
    """Hard-delete labels in `project_id` that no sample (top-level `label_id`
    or `extra_metadata.boundingBoxes[*]`) references. Also scrubs matching
    AIPrediction rows so a later "approve" can't resurrect the name.

    Returns the list of deleted labels as `[{id, name}, ...]`.
    """
    all_labels = db.query(Label).filter(Label.project_id == project_id).all()
    if not all_labels:
        return []

    referenced_ids: set[str] = set()
    referenced_names_lc: set[str] = set()

    referenced_ids.update(
        row[0]
        for row in db.query(Sample.label_id)
        .filter(Sample.project_id == project_id, Sample.label_id.isnot(None))
        .distinct()
        .all()
        if row[0]
    )

    box_rows = (
        db.query(Sample.extra_metadata)
        .filter(
            Sample.project_id == project_id,
            Sample.extra_metadata.isnot(None),
        )
        .all()
    )
    for (meta,) in box_rows:
        if not isinstance(meta, dict):
            continue
        for box in meta.get("boundingBoxes") or []:
            if not isinstance(box, dict):
                continue
            bid = box.get("label_id")
            if bid:
                referenced_ids.add(str(bid))
            bname = box.get("label")
            if isinstance(bname, str) and bname.strip():
                referenced_names_lc.add(bname.strip().lower())

    orphans = [
        l for l in all_labels
        if l.id not in referenced_ids
        and (l.name or "").strip().lower() not in referenced_names_lc
    ]
    if not orphans:
        return []

    orphan_names_lc = {(l.name or "").strip().lower() for l in orphans if l.name}
    predictions = (
        db.query(AIPrediction)
        .join(Sample, Sample.id == AIPrediction.sample_id)
        .filter(Sample.project_id == project_id)
        .all()
    )
    for pred in predictions:
        if str(pred.predicted_label or "").strip().lower() in orphan_names_lc:
            pred.predicted_label = None

    deleted = [{"id": l.id, "name": l.name} for l in orphans]
    for l in orphans:
        db.delete(l)
    db.commit()
    return deleted


@router.post("/project/{project_id}/prune-orphans")
def prune_orphan_labels(project_id: str, db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)):
    """Delete labels in a project that no sample or bbox references."""
    assert_project_owner(db, project_id, current_user)
    deleted = _prune_orphans_in_project(db, project_id)
    return {"deleted": len(deleted), "labels": deleted}
