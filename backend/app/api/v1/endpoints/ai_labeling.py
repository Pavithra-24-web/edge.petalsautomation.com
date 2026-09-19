"""
AI Labeling endpoints — actions CRUD, job execution, predictions, and apply/reject
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import threading, logging, copy

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import (
    assert_project_owner,
    assert_ai_action_owner,
    assert_ai_job_owner,
    assert_ai_prediction_owner,
)
from app.core.storage import storage
from app.models.user import (
    User, Project, Sample, Label,
    AILabelingAction, AILabelingJob, AIPrediction,
    AILabelingStatus,
)
from app.workers.sample_utils import IS_BACKGROUND_KEY, enforce_background_invariant
from app.services.labeling_status import is_sample_unlabeled

router = APIRouter()
logger = logging.getLogger(__name__)


_PLACEHOLDER_LABELS = {"", "unknown", "unlabeled", "unlabelled", "object"}


def _is_placeholder_label(name: Optional[str]) -> bool:
    return not name or name.strip().lower() in _PLACEHOLDER_LABELS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _action_to_dict(a: AILabelingAction) -> dict:
    return {
        "id": a.id,
        "project_id": a.project_id,
        "name": a.name,
        "model_type": a.model_type,
        "prompt": a.prompt,
        "provider": a.provider,
        "model_config": a.model_config_json or {},
        "label_names": getattr(a, "label_names", None),
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "updated_at": a.updated_at.isoformat() if a.updated_at else None,
    }


def _job_to_dict(j: AILabelingJob) -> dict:
    return {
        "id": j.id,
        "action_id": j.action_id,
        "project_id": j.project_id,
        "status": j.status.value if j.status else "running",
        "total_samples": j.total_samples,
        "processed": j.processed,
        "failed_count": j.failed_count,
        "error_message": j.error_message,
        "filter_options": j.filter_options or {},
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "created_at": j.created_at.isoformat() if j.created_at else None,
    }


def _pred_to_dict(p: AIPrediction) -> dict:
    return {
        "id": p.id,
        "job_id": p.job_id,
        "sample_id": p.sample_id,
        "predicted_label": p.predicted_label,
        "confidence": p.confidence,
        "bounding_boxes": p.bounding_boxes or [],
        "status": p.status,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        # Include sample info for frontend display
        "sample_filename": p.sample.filename if p.sample else None,
    }


# ─── Actions CRUD ─────────────────────────────────────────────────────────────

@router.post("/actions", status_code=201)
def create_action(
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new AI labeling action (reusable configuration)."""
    project_id = body.get("project_id")
    if not project_id:
        raise HTTPException(400, "project_id required")

    project = db.query(Project).filter(
        Project.id == project_id, Project.owner_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(404, "Project not found")

    action = AILabelingAction(
        project_id=project_id,
        name=body.get("name", "Untitled Action"),
        model_type=body.get("model_type", "classification"),
        prompt=body.get("prompt", ""),
        label_names=body.get("label_names"),
        provider=body.get("provider", "openai"),
        model_config_json=body.get("model_config", {}),
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return _action_to_dict(action)


@router.get("/actions/project/{project_id}")
def list_actions(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all AI labeling actions for a project."""
    assert_project_owner(db, project_id, current_user)
    actions = db.query(AILabelingAction).filter(
        AILabelingAction.project_id == project_id
    ).order_by(AILabelingAction.created_at.desc()).all()
    return [_action_to_dict(a) for a in actions]


@router.get("/actions/{action_id}")
def get_action(
    action_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    action = assert_ai_action_owner(db, action_id, current_user)
    return _action_to_dict(action)


@router.patch("/actions/{action_id}")
def update_action(
    action_id: str,
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    action = assert_ai_action_owner(db, action_id, current_user)

    for field in ["name", "model_type", "prompt", "label_names", "provider"]:
        if field in body:
            setattr(action, field, body[field])
    if "model_config" in body:
        action.model_config_json = body["model_config"]

    db.commit()
    db.refresh(action)
    return _action_to_dict(action)


@router.delete("/actions/{action_id}")
def delete_action(
    action_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    action = assert_ai_action_owner(db, action_id, current_user)
    db.delete(action)
    db.commit()
    return {"deleted": True}


# ─── Job Execution ────────────────────────────────────────────────────────────

def _load_owlvit():
    """
    Lazily load the OWL-ViT model + processor.
    Returns (processor, model) or (None, None) if unavailable.
    """
    try:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        import torch

        logger.info("Loading OWL-ViT model (google/owlvit-base-patch32)...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        model.eval()
        logger.info("OWL-ViT model loaded successfully")
        return processor, model
    except Exception as exc:
        logger.warning(f"OWL-ViT unavailable, falling back to heuristic: {exc}")
        return None, None


def _run_owlvit_detection(image_bytes: bytes, text_queries: list, processor, model, threshold: float = 0.10) -> list:
    """
    Run OWL-ViT zero-shot object detection on raw image bytes.
    Returns list of {x, y, w, h, label, confidence, id}.
    """
    import torch, uuid, io
    from PIL import Image as PILImage

    try:
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = img.size

        inputs = processor(text=[text_queries], images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])  # (h, w)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        boxes = []
        for score, label_idx, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            conf = round(score.item(), 4)
            label_text = text_queries[label_idx.item()] if label_idx.item() < len(text_queries) else "object"

            boxes.append({
                "x": round(x1),
                "y": round(y1),
                "w": round(x2 - x1),
                "h": round(y2 - y1),
                "label": label_text,
                "confidence": conf,
                "id": str(uuid.uuid4()),
            })

        # Sort by confidence descending, keep top 25
        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        return boxes[:25]

    except Exception as exc:
        logger.error(f"OWL-ViT inference failed: {exc}")
        return []


def _infer_label_heuristic(filename: str, prompt_lower: str):
    """Heuristic label inference from filename / folder structure."""
    import re
    parts = filename.replace("\\", "/").split("/")

    inferred_label = None
    confidence = 0.0

    if len(parts) > 1:
        inferred_label = parts[-2]
        confidence = 0.85
    elif "." in parts[-1]:
        base = parts[-1].rsplit(".", 1)[0]
        name_part = re.sub(r'[_\-\s]*\d+$', '', base)
        if name_part and len(name_part) > 1:
            inferred_label = name_part
            confidence = 0.65

    if prompt_lower and inferred_label:
        if inferred_label.lower() in prompt_lower:
            confidence = min(confidence + 0.1, 0.99)

    if not inferred_label:
        inferred_label = "unknown"
        confidence = 0.30

    return inferred_label, confidence


def _get_image_dimensions(image_bytes: bytes) -> tuple:
    """
    Read image dimensions from raw bytes using PIL.
    Returns (width, height) or (0, 0) on failure.
    """
    try:
        import io
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes))
        return img.size  # (width, height)
    except Exception:
        return (0, 0)


def _run_ai_labeling_job(job_id: str, action_id: str, sample_ids: list, db_url: str):
    """
    Background worker: processes samples through the AI model.

    ALL predictions MUST include bounding boxes:
    - detection mode  → OWL-ViT zero-shot (falls back to full-image box)
    - classification  → full-image bounding box with predicted label

    Non-image samples are skipped (no valid prediction without boxes).
    Uses a separate DB session since this runs in a background thread.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import uuid

    eng = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=eng)
    session = Session()

    try:
        job = session.query(AILabelingJob).filter(AILabelingJob.id == job_id).first()
        action = session.query(AILabelingAction).filter(AILabelingAction.id == action_id).first()
        if not job or not action:
            return

        prompt_lower = (action.prompt or "").lower()
        model_type = action.model_type or "classification"

        # Parse text queries from the prompt for OWL-ViT
        import re
        raw_fragments = [q.strip() for q in re.split(r',|\bor\b|\band\b', prompt_lower) if q.strip() and len(q.strip()) > 1]
        
        # Build a clean label map: fragment -> normalized label
        user_labels = []
        if getattr(action, "label_names", None):
            user_labels = [l.strip() for l in action.label_names.split(",") if l.strip()]

        if user_labels and len(user_labels) != len(raw_fragments):
            logger.warning(
                f"label_names count ({len(user_labels)}) does not match "
                f"prompt fragment count ({len(raw_fragments)}). Falling back to .title() for unmatched fragments."
            )
        
        label_map = {}
        for i, frag in enumerate(raw_fragments):
            if i < len(user_labels):
                label_map[frag] = user_labels[i]
            else:
                label_map[frag] = frag.title()

        text_queries = raw_fragments
        if not text_queries:
            text_queries = ["object"]
            label_map = {"object": "Object"}

        # Load OWL-ViT for detection mode
        owlvit_proc, owlvit_model = (None, None)
        if model_type == "detection":
            owlvit_proc, owlvit_model = _load_owlvit()
            # Detection mode has no heuristic fallback — if the model can't load
            # (e.g. torch missing or weights unreachable in prod), every sample
            # would be saved unlabeled at 0% confidence while the job still
            # reported "completed". Fail loudly instead so the cause is visible.
            if not owlvit_proc or not owlvit_model:
                raise RuntimeError(
                    "OWL-ViT model unavailable for detection job — check that "
                    "torch is installed and the model weights are reachable. "
                    "Refusing to write empty predictions."
                )

        processed = 0
        failed = 0

        for sid in sample_ids:
            try:
                sample = session.query(Sample).filter(Sample.id == sid).first()
                if not sample:
                    failed += 1
                    continue

                filename = sample.filename or ""
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                is_image = ext in ("jpg", "jpeg", "png", "bmp", "webp")

                # CRITICAL: Only process images — non-image samples cannot
                # produce valid bounding boxes and must be skipped.
                if not is_image:
                    logger.info(f"Skipping non-image sample {sid} ({filename})")
                    failed += 1
                    continue

                bboxes = []
                inferred_label = None
                confidence = 0.0

                # Download image bytes — needed for both detection and
                # classification so we can get real dimensions.
                image_bytes = None
                img_w, img_h = 0, 0
                try:
                    image_bytes = storage.download_bytes(sample.storage_key)
                    img_w, img_h = _get_image_dimensions(image_bytes)
                except Exception as dl_exc:
                    logger.warning(f"Could not download sample {sid}: {dl_exc}")

                # ── Detection mode ────────────────────────────────────────
                if model_type == "detection":
                    if owlvit_proc and owlvit_model and image_bytes:
                        # Ensure text queries are valid
                        valid_queries = text_queries if text_queries else ["object"]
                        bboxes = _run_owlvit_detection(image_bytes, valid_queries, owlvit_proc, owlvit_model, threshold=0.10)
                        
                        # Add fallback: If no boxes detected -> retry with lower threshold OR 'object'
                        if not bboxes:
                            logger.info(f"No boxes detected for {sid} at threshold 0.10, retrying with lower threshold...")
                            bboxes = _run_owlvit_detection(image_bytes, valid_queries, owlvit_proc, owlvit_model, threshold=0.02)
                        
                        if not bboxes and valid_queries != ["object"]:
                            logger.info(f"Still no boxes for {sid}, retrying with 'object' query...")
                            bboxes = _run_owlvit_detection(image_bytes, ["object"], owlvit_proc, owlvit_model, threshold=0.05)
                        
                        # Apply label_map to clean up names
                        for bx in bboxes:
                            raw_frag = bx.get("label", "object")
                            bx["query_fragment"] = raw_frag
                            bx["label"] = label_map.get(raw_frag, raw_frag.title())


                # ── Fallback / Classification: full-image bounding box ────
                if not bboxes and model_type == "classification":
                    inferred_label, confidence = _infer_label_heuristic(filename, prompt_lower)
                    # Use real image dimensions if available, otherwise a
                    # sensible default that covers the visible area.
                    box_w = img_w if img_w > 0 else 640
                    box_h = img_h if img_h > 0 else 480
                    bboxes = [{
                        "x": 0,
                        "y": 0,
                        "w": box_w,
                        "h": box_h,
                        "label": inferred_label,
                        "confidence": round(confidence, 4),
                        "id": str(uuid.uuid4()),
                    }]

                # If nothing was detected, keep the prediction unlabeled instead
                # of manufacturing a fake "Unlabeled" class.
                if not bboxes:
                    logger.warning(f"No bounding boxes for sample {sid}, leaving prediction unlabeled")
                    inferred_label = None
                    raw_frag = None
                    confidence = 0.0
                else:
                    # Derive top label from boxes
                    top_box = max(bboxes, key=lambda b: b.get("confidence", 0))
                    inferred_label = top_box["label"]
                    raw_frag = top_box.get("query_fragment")
                    confidence = top_box["confidence"]

                pred = AIPrediction(
                    job_id=job_id,
                    sample_id=sid,
                    predicted_label=inferred_label,
                    query_fragment=raw_frag,
                    confidence=round(confidence, 4),
                    bounding_boxes=bboxes,
                    status="pending",
                )
                session.add(pred)
                processed += 1

            except Exception as exc:
                logger.error(f"AI labeling failed for sample {sid}: {exc}")
                failed += 1

            # Update progress periodically
            if processed % 10 == 0:
                job.processed = processed
                job.failed_count = failed
                session.commit()

        # Finalize
        job.processed = processed
        job.failed_count = failed
        job.status = AILabelingStatus.completed
        job.completed_at = datetime.utcnow()
        session.commit()

    except Exception as exc:
        logger.error(f"AI labeling job {job_id} failed: {exc}")
        try:
            job = session.query(AILabelingJob).filter(AILabelingJob.id == job_id).first()
            if job:
                job.status = AILabelingStatus.failed
                job.error_message = str(exc)
                job.completed_at = datetime.utcnow()
                session.commit()
        except Exception:
            pass
    finally:
        session.close()


@router.post("/run")
def run_ai_labeling(
    body: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Start an AI labeling job.
    
    Body: {
        action_id, project_id,
        sample_ids (optional — if empty, use all matching),
        skip_labeled (bool), sample_type (str), filter_labels (list)
    }
    """
    action_id = body.get("action_id")
    project_id = body.get("project_id")
    
    if not action_id or not project_id:
        raise HTTPException(400, "action_id and project_id required")

    action = assert_ai_action_owner(db, action_id, current_user)
    # Caller must own the named project, and the action must belong to it.
    assert_project_owner(db, project_id, current_user)
    if action.project_id != project_id:
        raise HTTPException(404, "Action not found")

    # Build sample query
    sample_ids = body.get("sample_ids", [])
    skip_labeled = body.get("skip_labeled", True)
    sample_type = body.get("sample_type")
    
    if not sample_ids:
        if skip_labeled:
            # Shared predicate (app/services/labeling_status.py) rather than
            # a bare `label_id IS NULL` check — that older check re-fed
            # background-marked negatives into AI labeling on every run,
            # since a background sample deliberately has no label_id. Same
            # definition the Dataset page's "Unlabeled" filter and the
            # labeling-status-summary endpoint use, so what this run
            # processes can never silently disagree with what the UI shows
            # as outstanding.
            q = db.query(Sample.id, Sample.filename, Sample.sample_type, Sample.extra_metadata).filter(
                Sample.project_id == project_id
            )
            if sample_type:
                q = q.filter(Sample.sample_type == sample_type)
            sample_ids = [row.id for row in q.all() if is_sample_unlabeled(row)]
        else:
            q = db.query(Sample.id).filter(Sample.project_id == project_id)
            if sample_type:
                q = q.filter(Sample.sample_type == sample_type)
            sample_ids = [row[0] for row in q.all()]
        logger.info(f"AI Labeling: DB query fetched {len(sample_ids)} samples. Project: {project_id}, Type: {sample_type}, Skip Labeled: {skip_labeled}")
    else:
        logger.info(f"AI Labeling: Received {len(sample_ids)} explicit sample IDs from frontend.")
    
    if not sample_ids:
        raise HTTPException(400, "No samples to process matching your criteria")
    
    # Create the job
    job = AILabelingJob(
        action_id=action_id,
        project_id=project_id,
        status=AILabelingStatus.running,
        total_samples=len(sample_ids),
        filter_options={
            "skip_labeled": skip_labeled,
            "sample_type": sample_type,
            "sample_count": len(sample_ids),
        },
        started_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Launch background processing
    from app.core.config import settings
    db_url = settings.DATABASE_URL
    
    thread = threading.Thread(
        target=_run_ai_labeling_job,
        args=(job.id, action.id, sample_ids, db_url),
        daemon=True,
    )
    thread.start()
    
    return _job_to_dict(job)


# ─── Job Status & Predictions ────────────────────────────────────────────────

@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = assert_ai_job_owner(db, job_id, current_user)
    return _job_to_dict(job)


@router.get("/jobs/project/{project_id}")
def list_jobs(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_project_owner(db, project_id, current_user)
    jobs = db.query(AILabelingJob).filter(
        AILabelingJob.project_id == project_id
    ).order_by(AILabelingJob.created_at.desc()).all()
    return [_job_to_dict(j) for j in jobs]


@router.get("/jobs/{job_id}/predictions")
def get_predictions(
    job_id: str,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get predictions for a job, optionally filtered by status."""
    assert_ai_job_owner(db, job_id, current_user)
    q = db.query(AIPrediction).filter(AIPrediction.job_id == job_id)
    if status:
        q = q.filter(AIPrediction.status == status)
    total = q.count()
    preds = q.offset(skip).limit(limit).all()
    return {
        "total": total,
        "items": [_pred_to_dict(p) for p in preds],
    }


@router.patch("/predictions/{prediction_id}")
def update_prediction(
    prediction_id: str,
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an AI prediction's bounding boxes and/or predicted label.

    Body may include: bounding_boxes (list), predicted_label (str).
    """
    pred = assert_ai_prediction_owner(db, prediction_id, current_user)
    if "bounding_boxes" in body:
        pred.bounding_boxes = body["bounding_boxes"] or []
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(pred, "bounding_boxes")
    if "predicted_label" in body:
        pred.predicted_label = body["predicted_label"]
    db.commit()
    db.refresh(pred)
    return _pred_to_dict(pred)


# ─── Apply / Reject ──────────────────────────────────────────────────────────

def _get_or_create_label(name: str, project_id: str, db: Session) -> str:
    """Get an existing label by name or create a new one. Returns label_id."""
    if _is_placeholder_label(name):
        raise ValueError("Placeholder labels must not be created as project labels")
    name = name.strip()
    label = db.query(Label).filter(Label.project_id == project_id, Label.name == name).first()
    if not label:
        label = Label(project_id=project_id, name=name)
        db.add(label)
        db.flush()
    return label.id


@router.post("/apply")
def apply_predictions(
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Apply approved predictions → convert to actual dataset labels.
    
    Body: {
        prediction_ids: [...],  (if empty, apply all pending for job_id)
        job_id: "...",
        project_id: "...",
    }
    """
    prediction_ids = body.get("prediction_ids", [])
    job_id = body.get("job_id")
    project_id = body.get("project_id")
    
    if not project_id:
        raise HTTPException(400, "project_id required")
    # Caller must own the target project that labels will be written into.
    assert_project_owner(db, project_id, current_user)

    # Only operate on predictions the caller owns (via job → project), so a
    # forged prediction_id from another tenant cannot be applied.
    base = (
        db.query(AIPrediction)
        .join(AILabelingJob, AIPrediction.job_id == AILabelingJob.id)
        .join(Project, AILabelingJob.project_id == Project.id)
        .filter(Project.owner_id == current_user.id)
    )
    if prediction_ids:
        preds = base.filter(AIPrediction.id.in_(prediction_ids)).all()
    elif job_id:
        preds = base.filter(
            AIPrediction.job_id == job_id,
            AIPrediction.status == "pending",
        ).all()
    else:
        raise HTTPException(400, "prediction_ids or job_id required")

    applied = 0
    skipped = 0
    background = 0
    for pred in preds:
        if pred.status == "rejected":
            continue

        # ── Zero-detection run → training negative ───────────────────────────
        # The labeling worker leaves predicted_label None and bounding_boxes []
        # when the detector found nothing (see the "leaving prediction
        # unlabeled" branch in run_ai_labeling).  Approving that used to be a
        # silent no-op; the user's "yes, there is nothing here" was thrown away.
        # Record it as an explicit background marker instead of manufacturing a
        # label_id the image does not deserve.
        #
        # Classification jobs never reach here with an empty box list — that
        # path always synthesises a full-image box — so this is detection-only.
        if not pred.bounding_boxes:
            sample = db.query(Sample).filter(
                Sample.id == pred.sample_id,
                Sample.project_id == project_id,
            ).first()
            if sample is None:
                skipped += 1
                continue
            meta = copy.deepcopy(sample.extra_metadata or {})
            meta[IS_BACKGROUND_KEY] = True
            meta["boundingBoxes"] = []
            sample.extra_metadata = meta
            sample.label_id = None
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(sample, "extra_metadata")
            pred.status = "approved"
            background += 1
            applied += 1
            continue

        if not pred.predicted_label:
            continue

        if _is_placeholder_label(pred.predicted_label):
            logger.info(f"Skipping prediction {pred.id} with placeholder label {pred.predicted_label!r}")
            skipped += 1
            continue
        
        # Create/find label in the project
        try:
            label_id = _get_or_create_label(pred.predicted_label, project_id, db)
        except ValueError:
            skipped += 1
            continue

        # Assign label to sample — restricted to the owned target project.
        sample = db.query(Sample).filter(
            Sample.id == pred.sample_id,
            Sample.project_id == project_id,
        ).first()
        if sample:
            sample.label_id = label_id

            # Enrich bounding boxes with label_id so the training worker can
            # resolve them via either 'label_id' (UUID) or 'label' (name).
            # Without label_id, _create_fomo_heatmap silently skips every box
            # because its label_map is UUID-keyed → empty GT heatmaps → TP=0.
            enriched_boxes = []
            for box in (pred.bounding_boxes or []):
                b = dict(box)
                # If the box already has a valid UUID, leave it; otherwise look
                # up the name string and inject the resolved UUID.
                if _is_placeholder_label(b.get("label")):
                    b["label"] = None
                    b["label_id"] = None
                    enriched_boxes.append(b)
                    continue
                if not b.get("label_id"):
                    box_label_name = b.get("label", "")
                    if box_label_name:
                        try:
                            resolved_id = _get_or_create_label(box_label_name, project_id, db)
                            if resolved_id:
                                b["label_id"] = resolved_id
                        except ValueError:
                            b["label"] = None
                            b["label_id"] = None
                enriched_boxes.append(b)

            # Always save bounding boxes into the sample's annotations
            meta = copy.deepcopy(sample.extra_metadata or {})
            meta["boundingBoxes"] = enriched_boxes
            # A sample can arrive here still carrying a stale background
            # marker (e.g. it was manually marked background earlier, and a
            # re-run of AI labeling now finds a real object). Real, labeled
            # boxes mean it's no longer a negative — mirrors the zero-box
            # branch above, which does the same in reverse.
            meta = enforce_background_invariant(meta, label_id=sample.label_id)
            sample.extra_metadata = meta
            # Force SQLAlchemy to detect the JSON mutation
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(sample, "extra_metadata")

            pred.status = "approved"
            applied += 1
    
    db.commit()
    return {
        "applied": applied,
        "skipped": skipped,
        "background": background,
        "total": len(preds),
    }


@router.post("/reject")
def reject_predictions(
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Reject predictions — marks them so they won't be applied.
    
    Body: {prediction_ids: [...]} or {job_id: "..."} to reject all pending
    """
    prediction_ids = body.get("prediction_ids", [])
    job_id = body.get("job_id")

    # Scope to predictions the caller owns (via job → project).
    base = (
        db.query(AIPrediction)
        .join(AILabelingJob, AIPrediction.job_id == AILabelingJob.id)
        .join(Project, AILabelingJob.project_id == Project.id)
        .filter(Project.owner_id == current_user.id)
    )
    if prediction_ids:
        preds = base.filter(AIPrediction.id.in_(prediction_ids)).all()
    elif job_id:
        preds = base.filter(
            AIPrediction.job_id == job_id,
            AIPrediction.status == "pending",
        ).all()
    else:
        raise HTTPException(400, "prediction_ids or job_id required")

    rejected = 0
    for pred in preds:
        pred.status = "rejected"
        rejected += 1
    
    db.commit()
    return {"rejected": rejected}
