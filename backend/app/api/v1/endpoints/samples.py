"""
Sample endpoints — ingest, list, label, delete sensor data
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from pydantic import BaseModel
from typing import Optional, List, Any
import asyncio
import csv
import json, io
import xml.etree.ElementTree as ET
from botocore.exceptions import ClientError, BotoCoreError

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.authz import assert_project_owner, assert_sample_owner
from app.core.storage import storage
from app.models.user import User, Sample, Project, Label, SampleType
from app.api.v1.endpoints.labels import _prune_orphans_in_project
from app.workers.sample_utils import IS_BACKGROUND_KEY, enforce_background_invariant
from app.services.labeling_status import is_sample_unlabeled, summarize
from app.services.sample_ingest import (
    create_sample_from_bytes,
    _resolve_sample_type,
    _extract_metadata,
    _parse_csv_feature_names,
)
from app.motion.services import signal_decode

router = APIRouter()


class SampleResponse(BaseModel):
    id: str
    filename: str
    sensor_type: Optional[str]
    frequency_hz: Optional[float]
    duration_ms: Optional[int]
    num_channels: int
    num_samples: Optional[int]
    sample_type: str
    label_id: Optional[str]
    label_name: Optional[str]
    file_size_bytes: Optional[int]
    created_at: str

    class Config:
        from_attributes = True


def _normalize_box_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"unlabeled", "unlabelled", "unknown"}:
        return ""
    return text


def _normalize_box_geometry(box: dict) -> dict:
    """Canonicalize bounding-box geometry keys to {x, y, w, h}.

    The Edge Impulse info.labels format uses width/height while every other
    import path (_box_xyxy_to_xywh) and the UI annotation tool both use w/h.
    Translating here keeps one canonical schema in extra_metadata.boundingBoxes
    so the Konva renderer never receives undefined dimensions.
    """
    box = dict(box)
    if "width" in box and "w" not in box:
        box["w"] = box.pop("width")
    if "height" in box and "h" not in box:
        box["h"] = box.pop("height")
    return box


def _build_label_index(db: Session, project_id: str) -> dict:
    """Load a project's labels once into ``{"by_id": ..., "by_name": ...}``.

    Passed into ``_sample_to_dict`` when serializing many samples so per-row
    label resolution is a dict hit instead of a SELECT (was an N+1 across the
    page / batch).
    """
    labels = db.query(Label).filter(Label.project_id == project_id).all()
    return {
        "by_id":   {l.id: l for l in labels},
        "by_name": {l.name: l for l in labels},
    }


def _resolve_bbox_display_label(
    s: Sample,
    db: Optional[Session] = None,
    label_index: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    meta = s.extra_metadata or {}
    boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
    if not isinstance(boxes, list):
        return None, None

    for box in boxes:
        if not isinstance(box, dict):
            continue

        box_label_name = _normalize_box_label(box.get("label"))
        box_label_id = box.get("label_id")

        if box_label_name:
            resolved_id = box_label_id
            if not resolved_id:
                # Prefer the preloaded index (no query); fall back to a per-row
                # SELECT only when no index was supplied (single-sample callers).
                if label_index is not None:
                    lbl = label_index["by_name"].get(box_label_name)
                    resolved_id = lbl.id if lbl else None
                elif db is not None:
                    lbl = (
                        db.query(Label)
                        .filter(Label.project_id == s.project_id, Label.name == box_label_name)
                        .first()
                    )
                    resolved_id = lbl.id if lbl else None
            return resolved_id, box_label_name

        if box_label_id:
            if label_index is not None:
                lbl = label_index["by_id"].get(box_label_id)
                if lbl:
                    return lbl.id, lbl.name
            elif db is not None:
                lbl = (
                    db.query(Label)
                    .filter(Label.project_id == s.project_id, Label.id == box_label_id)
                    .first()
                )
                if lbl:
                    return lbl.id, lbl.name

    return None, None


def _sample_to_dict(
    s: Sample,
    db: Optional[Session] = None,
    label_index: Optional[dict] = None,
) -> dict:
    meta = s.extra_metadata or {}
    box_label_id, box_label_name = _resolve_bbox_display_label(s, db, label_index)
    if label_index is not None:
        # Resolve the top-level label name from the index so we never trigger a
        # lazy `s.label` relationship load per row.
        _lbl = label_index["by_id"].get(s.label_id) if s.label_id else None
        top_level_label_name = _lbl.name if _lbl else None
    else:
        top_level_label_name = s.label.name if s.label else None
    return {
        "id": s.id,
        "filename": s.filename,
        "sensor_type": s.sensor_type,
        "frequency_hz": s.frequency_hz,
        "duration_ms": s.duration_ms,
        "num_channels": s.num_channels,
        "num_samples": s.num_samples,
        "sample_type": s.sample_type,
        "label_id": box_label_id or s.label_id,
        "label_name": box_label_name or top_level_label_name,
        "file_size_bytes": s.file_size_bytes,
        "created_at": s.created_at.isoformat(),
        "is_disabled": meta.get("is_disabled", False),
        # Explicit background-only ("negative") image — see IS_BACKGROUND_KEY.
        # Surfaced as a top-level field so the dataset page can badge it without
        # reaching into extra_metadata, exactly like is_disabled.
        "is_background": bool(meta.get(IS_BACKGROUND_KEY, False)),
        "extra_metadata": meta,
    }


def _set_sample_label(sample: Sample, label_id: Optional[str], label_name: Optional[str]) -> None:
    """Keep top-level and image-box labels aligned for dataset-page actions."""
    sample.label_id = label_id or None

    meta = sample.extra_metadata or {}
    if not isinstance(meta, dict):
        meta = {}
    boxes = meta.get("boundingBoxes")

    new_meta = dict(meta)
    if isinstance(boxes, list):
        updated_boxes = []
        normalized_name = _normalize_box_label(label_name)
        for box in boxes:
            if not isinstance(box, dict):
                updated_boxes.append(box)
                continue

            updated = _normalize_box_geometry(box)
            if normalized_name:
                updated["label"] = normalized_name
                updated["label_id"] = label_id
            else:
                updated["label"] = None
                updated["label_id"] = None
            updated_boxes.append(updated)
        new_meta["boundingBoxes"] = updated_boxes

    # A real label just got assigned — a background/negative assertion can no
    # longer hold. Runs even when there's no box list to touch, since a
    # background sample being given a plain classification label must still
    # drop the marker.
    new_meta = enforce_background_invariant(new_meta, label_id=sample.label_id)

    sample.extra_metadata = new_meta
    flag_modified(sample, "extra_metadata")


def _sample_matches_label_filter(sample: Sample, label_id: str, label_name: Optional[str]) -> bool:
    if sample.label_id == label_id:
        return True

    meta = sample.extra_metadata or {}
    boxes = meta.get("boundingBoxes") if isinstance(meta, dict) else None
    if not isinstance(boxes, list):
        return False

    normalized_target = _normalize_box_label(label_name)
    for box in boxes:
        if not isinstance(box, dict):
            continue
        if box.get("label_id") == label_id:
            return True
        if normalized_target and _normalize_box_label(box.get("label")) == normalized_target:
            return True
    return False


def _clear_associated_sample_records(db: Session, sample_id: str) -> None:
    from app.models.model_testing import ModelTestSample
    from app.models.user import AIPrediction, FeatureSet

    db.query(AIPrediction).filter(AIPrediction.sample_id == sample_id).delete()
    db.query(FeatureSet).filter(FeatureSet.sample_id == sample_id).delete()
    db.query(ModelTestSample).filter(ModelTestSample.sample_id == sample_id).update(
        {ModelTestSample.sample_id: None},
        synchronize_session=False,
    )


# Max number of object-storage uploads kept in flight at once during a batch
# upload. Bounds both event-loop fan-out and peak in-flight memory.
_BATCH_UPLOAD_CONCURRENCY = 16


def _get_or_create_label_cached(
    name: str, project_id: str, db: Session, cache: dict
) -> Optional[str]:
    """Cache-backed variant of ``_get_or_create_label`` for batch loops.

    ``cache`` maps ``label_name -> Label`` and is mutated in place when a new
    label is created, so a batch that references the same label across thousands
    of files issues at most one INSERT per distinct new label and zero repeat
    SELECTs. Behaviour is otherwise identical to ``_get_or_create_label``.
    """
    if not name or not name.strip():
        return None
    name = name.strip()
    if name.startswith(("{", "[")):
        return None
    label = cache.get(name)
    if label is None:
        label = Label(project_id=project_id, name=name)
        db.add(label)
        db.flush()  # assign id without committing
        cache[name] = label
    return label.id


def _clean_rel_path(path: str) -> str:
    return str(path or "").strip().lstrip("/").replace("\\", "/")


def _path_variants(path: str) -> list[str]:
    clean = _clean_rel_path(path)
    if not clean:
        return []
    parts = clean.split("/")
    variants = [clean, parts[-1]]
    return list(dict.fromkeys([v for v in variants if v]))


def _splitext_lower(path: str) -> tuple[str, str]:
    if "." not in path:
        return path, ""
    idx = path.rfind(".")
    return path[:idx], path[idx:].lower()


def _image_file_paths(file_data: dict[str, tuple[bytes, str]]) -> set[str]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return {
        path for path in file_data
        if _splitext_lower(path)[1] in image_exts
    }


def _find_matching_image_path(ref_path: str, image_paths: set[str]) -> Optional[str]:
    clean = _clean_rel_path(ref_path)
    if not clean:
        return None
    stem, _ = _splitext_lower(clean)
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    candidates = [clean, *(stem + ext for ext in image_exts), clean.split("/")[-1]]

    # Sibling-folder convention (Ultralytics/Roboflow/Darknet exports):
    # annotations live in a `labels/` (or `annotations/`) dir next to a
    # sibling `images/` dir at the same split level, e.g.
    # train/labels/0001.txt <-> train/images/0001.jpg. Try the swapped path
    # before the global filename search below, since that fallback can't
    # tell train/0001 from valid/0001 when both splits reuse filenames.
    parts = stem.split("/")
    for i, part in enumerate(parts):
        if part.lower() in ("labels", "annotations"):
            swapped = "/".join(parts[:i] + ["images"] + parts[i + 1:])
            candidates.extend(swapped + ext for ext in image_exts)
            break

    for candidate in candidates:
        if candidate in image_paths:
            return candidate

    # Fuzzy fallback: match by bare filename stem. `image_paths` is an
    # unordered set, so when more than one image shares that stem (e.g.
    # duplicate filenames reused across train/valid/test splits) iterating
    # it directly returns whichever the set happens to yield first —
    # silently attaching one split's annotation to another split's image.
    # Instead, prefer whichever candidate shares the longest directory-path
    # prefix with the annotation, with a deterministic tie-break.
    bare_stem = parts[-1]
    ann_dir_parts = parts[:-1]
    matches = [
        image_path for image_path in image_paths
        if (lambda s: s == stem or s.split("/")[-1] == bare_stem)(_splitext_lower(image_path)[0])
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    def _dir_overlap(image_path: str) -> int:
        image_dir_parts = _splitext_lower(image_path)[0].split("/")[:-1]
        overlap = 0
        for a, b in zip(ann_dir_parts, image_dir_parts):
            if a != b:
                break
            overlap += 1
        return overlap

    matches.sort(key=lambda p: (-_dir_overlap(p), p))
    return matches[0]


def _read_image_size(image_bytes: bytes) -> tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        return img.size
    except Exception:
        return None, None


def _box_xyxy_to_xywh(label: str, x1: float, y1: float, x2: float, y2: float) -> Optional[dict]:
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    if w <= 0 or h <= 0:
        return None
    return {
        "label": _normalize_box_label(label),
        "x": float(x1),
        "y": float(y1),
        "w": float(w),
        "h": float(h),
    }


def _record_annotation_label(store: dict[str, str], path: str, label: str) -> None:
    normalized = _normalize_box_label(label)
    if not normalized:
        return
    for key in _path_variants(path):
        store[key] = normalized


def _record_annotation_boxes(store: dict[str, list], path: str, boxes: list[dict]) -> None:
    # `boxes == []` is meaningful and must be recorded: it is how EI's
    # info.labels and Darknet's zero-length .txt declare "this image is
    # background".  Only a missing annotation (None) is a no-op.  Callers read
    # this store by key *presence*, never by truthiness, for the same reason.
    if boxes is None:
        return
    normalized = [_normalize_box_geometry(dict(b)) for b in boxes if isinstance(b, dict)]
    for key in _path_variants(path):
        # Empty never overwrites non-empty. Two sidecars in one upload can name
        # the same image and disagree (an info.labels entry with boxes next to
        # an empty .txt); without this the winner would be whichever file the
        # request happened to iterate last. An annotated object is real
        # evidence, an empty list is the absence of it, so boxes win.
        if not normalized and store.get(key):
            continue
        store[key] = list(normalized)


_SPLIT_FOLDERS = {
    "training", "train",
    "testing", "test",
    "valid", "validation", "val",
    "post-processing", "postprocessing", "post_processing",
}

# Maps each split-folder name (lowercase) to the SampleType value stored in the
# DB. Includes this app's own export folder names ("training"/"testing") plus
# the conventional names external tools (YOLO/Roboflow/Ultralytics/Kaggle)
# use, so a folder dataset imports into the right split without requiring it
# to have been round-tripped through this app's exporter first.
_SPLIT_FOLDER_TO_TYPE: dict[str, str] = {
    "training": "training",
    "train": "training",
    "testing": "testing",
    "test": "testing",
    # SampleType has no separate validation bucket — this app's split is
    # binary train/test — so an external dataset's val/validation folder
    # maps onto "testing".
    "valid": "testing",
    "validation": "testing",
    "val": "testing",
    "post-processing": "postprocessing",
    "postprocessing": "postprocessing",
    "post_processing": "postprocessing",
}


def _resolve_folder_label_from_path(clean_name: str) -> Optional[str]:
    parts = [p for p in _clean_rel_path(clean_name).split("/") if p]
    if len(parts) < 2:
        return None
    candidate = parts[-2].strip()
    if not candidate:
        return None
    # Any recognised split-folder name used as the immediate parent means the
    # file is directly inside that split root — there is no label sub-directory
    # between the split and the file. The `len == 2` guard that was here
    # previously only fired for 2-part paths (split/file.ext); browser folder
    # uploads via webkitRelativePath include the root directory name, making
    # every path one level deeper (root/split/file.ext, len == 3), so the guard
    # never fired and "training" / "testing" / "post-processing" leaked through
    # as label names. Removing the depth constraint fixes all layouts.
    if candidate.lower() in _SPLIT_FOLDERS:
        return None
    return candidate


def _resolve_split_from_path(clean_name: str) -> Optional[str]:
    """Return the sample_type implied by the first recognised split-folder in
    clean_name (skipping the root directory name), or None.

    Handles both bare paths (training/file.jpg) and webkitRelativePath-style
    paths that include the root folder (export-folder/training/file.jpg).
    """
    parts = [p for p in _clean_rel_path(clean_name).split("/") if p]
    for part in parts[:-1]:   # exclude the filename itself
        mapped = _SPLIT_FOLDER_TO_TYPE.get(part.lower())
        if mapped:
            return mapped
    return None


def _parse_yolo_name_file(content: bytes) -> list[str]:
    names: list[str] = []
    for raw_line in content.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def _parse_yolo_yaml_names(content: bytes) -> dict[int, str]:
    text = content.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    names: dict[int, str] = {}
    in_names = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_names:
            if line.startswith("names:"):
                after = line.split(":", 1)[1].strip()
                if after.startswith("[") and after.endswith("]"):
                    items = [x.strip().strip("'\"") for x in after[1:-1].split(",") if x.strip()]
                    return {idx: item for idx, item in enumerate(items)}
                in_names = True
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip().isdigit():
                names[int(k.strip())] = v.strip().strip("'\"")
                continue
        if line.startswith("-"):
            names[len(names)] = line[1:].strip().strip("'\"")
            continue
        if names:
            break
    return names


def _build_yolo_class_map(file_data: dict[str, tuple[bytes, str]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    preferred_name_files = ("classes.txt", "labels.txt", "obj.names")
    for path, (content, _ct) in file_data.items():
        lower = path.lower()
        if any(lower.endswith(name) for name in preferred_name_files):
            names = _parse_yolo_name_file(content)
            if names:
                return {idx: name for idx, name in enumerate(names)}
    for path, (content, _ct) in file_data.items():
        lower = path.lower()
        if lower.endswith("data.yaml") or lower.endswith("data.yml") or lower.endswith("dataset.yaml") or lower.endswith("dataset.yml"):
            parsed = _parse_yolo_yaml_names(content)
            if parsed:
                return parsed
    return mapping


_OPEN_IMAGES_BOX_FIELDS = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}


def _build_open_images_class_map(file_data: dict[str, tuple[bytes, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path, (content, _ct) in file_data.items():
        lower = path.lower()
        if not lower.endswith(".csv"):
            continue
        text = content.decode("utf-8", errors="ignore")
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            continue
        # Skip the box-annotation CSV itself. Its header doesn't match the
        # two-column LabelName/DisplayName class-map shape below, so without
        # this it falls into the `else` branch and every row (including the
        # header) gets folded into the map as {ImageID: <2nd column>} pairs.
        if _OPEN_IMAGES_BOX_FIELDS.issubset({h.strip() for h in rows[0]}):
            continue
        if rows[0][:2] == ["LabelName", "DisplayName"]:
            data_rows = rows[1:]
        else:
            data_rows = rows
        built = {}
        for row in data_rows:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                built[row[0].strip()] = row[1].strip()
        if built:
            mapping.update(built)
    return mapping


@router.post("/upload", status_code=201)
async def upload_sample(
    project_id: str = Form(...),
    label_id: Optional[str] = Form(None),
    sample_type: str = Form("training"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a raw sensor data file (CSV, JSON, WAV, or binary)."""
    project = db.query(Project).filter(
        Project.id == project_id, Project.owner_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    content = await file.read()
    sample = create_sample_from_bytes(
        db,
        project_id=project_id,
        filename=file.filename,
        content=content,
        content_type=file.content_type,
        sample_type=sample_type,
        label_id=label_id,
        uploaded_by=current_user.id,
    )
    return _sample_to_dict(sample, db)


@router.post("/upload-batch", status_code=201)
async def upload_batch(
    project_id: str = Form(...),
    sample_type: str = Form("training"),
    label_name: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Batch upload endpoint supporting:
    - Multiple files in a single request
    - Folder upload (relative path preserved via webkitRelativePath → filename)
    - info.labels JSON annotation file parsing (classification labels + bounding boxes)
    - Automatic 80/20 split when sample_type='automatic'
    - Label auto-creation from folder name or annotation file

    File naming conventions (from folder uploads):
        label_name/filename.ext           → label from immediate parent folder
        split/label_name/filename.ext     → label from immediate parent folder

    Label resolution priority per file:
        1. info.labels full path match
        2. info.labels bare filename match
        3. Immediate parent folder name
        4. Explicit label_name form field
    """
    project = db.query(Project).filter(
        Project.id == project_id, Project.owner_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # ── Step 1: Read all file contents and separate info.labels files ────────
    file_data: dict[str, tuple[bytes, str]] = {}   # filename → (bytes, content_type)
    annotations_labels: dict[str, str] = {}         # filename → label_name (classification)
    # Lower-priority labels parsed from EI-canonical dict form
    # ({type:"label", label:"foo"}). Kept separate because some annotation
    # tools dump a per-image hash into that field, producing one class per
    # image when folder inference would have given the correct label. We
    # therefore consult this bucket only AFTER folder inference.
    weak_annotations_labels: dict[str, str] = {}
    annotations_boxes: dict[str, list] = {}         # filename → bounding box list
    # Filenames the annotation sidecar explicitly declared to be background-only
    # (EI `boundingBoxes: []`, zero-length Darknet .txt).  Kept separate from
    # annotations_boxes so "the sidecar said empty" is never confused with "the
    # sidecar said nothing" — the two used to collapse into the same falsy value
    # and the background declaration was silently discarded.
    background_paths: set[str] = set()
    # EI-canonical "category" field from info.labels — "training", "testing",
    # "postprocessing". Highest-priority source for sample_type: the exporter
    # wrote it explicitly so it overrides both the form field and path inference.
    annotations_category: dict[str, str] = {}       # filename → sample_type
    annotation_sidecars: set[str] = set()

    for f in files:
        raw_name = f.filename or ""
        content = await f.read()
        # Strip leading slashes / normalise separators
        clean_name = _clean_rel_path(raw_name)

        # Detect Edge Impulse annotation sidecar files. Some exports use
        # names like "bounding_boxes.labels" rather than literally "info.labels".
        if clean_name.lower().endswith(".labels"):
            try:
                parsed = json.loads(content.decode("utf-8"))
                for entry in parsed.get("files", []):
                    path = entry.get("path", "").replace("\\", "/").lstrip("/")
                    if not path:
                        continue
                    bare = path.split("/")[-1]
                    # Parse EI "category" field → sample_type. Normalise
                    # "post-processing" (hyphenated in the export) to the
                    # "postprocessing" value stored in the DB SampleType enum.
                    _raw_cat = entry.get("category", "")
                    if isinstance(_raw_cat, str) and _raw_cat:
                        _mapped = _SPLIT_FOLDER_TO_TYPE.get(_raw_cat.lower().strip())
                        if _mapped:
                            annotations_category[path] = _mapped
                            annotations_category[bare] = _mapped
                    # Cast to str defensively — malformed JSON might have a non-string label field.
                    # The EI canonical info.labels stores `label` as an object {type:"label",label:"foo"};
                    # accept both shapes so exports round-trip cleanly. Dict-form goes into the
                    # weak bucket so folder inference still wins for hash-label-per-image exports.
                    _raw_lbl = entry.get("label")
                    is_dict_label = False
                    if isinstance(_raw_lbl, str):
                        lbl = _raw_lbl.strip()
                    elif isinstance(_raw_lbl, dict):
                        inner = _raw_lbl.get("label")
                        lbl = inner.strip() if isinstance(inner, str) else ""
                        is_dict_label = bool(lbl)
                    else:
                        lbl = ""
                    boxes = entry.get("boundingBoxes")

                    if isinstance(boxes, list) and not boxes:
                        # Explicit `boundingBoxes: []` — EI's background marker.
                        # Falls before the truthy branch because an empty list is
                        # falsy and would otherwise drop through to the label
                        # branch and lose the declaration entirely.
                        _record_annotation_boxes(annotations_boxes, path, [])
                        background_paths.update(_path_variants(path))
                    elif boxes and isinstance(boxes, list):
                        # Object-detection entry: store boxes under both keys.
                        # Derive the classification label from the first bounding
                        # box's class — that is the intentional annotation. The
                        # entry-level `label` is only a fallback: Roboflow / EI
                        # object-detection exports stuff the per-image filename
                        # stem (e.g. "101_jpg") into it, which would otherwise
                        # become a bogus one-class-per-image top-level label and
                        # diverge from the boxes the data table renders from.
                        _record_annotation_boxes(annotations_boxes, path, boxes)
                        _box0_lbl = boxes[0].get("label") if boxes else None
                        first_box_lbl = _box0_lbl.strip() if isinstance(_box0_lbl, str) else ""
                        derived_lbl = first_box_lbl or lbl
                        _is_weak = not first_box_lbl and is_dict_label
                        _placeholder = (
                            _is_weak
                            and derived_lbl.strip().lower() == _splitext_lower(bare)[0].strip().lower()
                        )
                        if derived_lbl and not _placeholder:
                            # Bounding-box derived labels are strong — boxes are intentional
                            # per-image metadata, not the hash-label anti-pattern.
                            target = weak_annotations_labels if _is_weak else annotations_labels
                            for key in (path, bare):
                                target[key] = derived_lbl
                    elif lbl:
                        # A dict-form label that just echoes the image's own
                        # filename stem (e.g. "screenshot_000.jpg" labeled
                        # "screenshot_000") is the placeholder this app's own
                        # exporter — and other EI-family tools — write for a
                        # sample that has no real label and no boxes, not a
                        # human-assigned class name. Treating it as real would
                        # mint one bogus per-image Label on every reimport of
                        # an unlabeled dataset.
                        _bare_stem = _splitext_lower(bare)[0].strip().lower()
                        _is_filename_placeholder = (
                            is_dict_label and lbl.strip().lower() == _bare_stem
                        )
                        if not _is_filename_placeholder:
                            target = weak_annotations_labels if is_dict_label else annotations_labels
                            for key in (path, bare):
                                target[key] = lbl
            except Exception:
                pass
            annotation_sidecars.add(clean_name)
            continue   # don't store annotation files as samples

        file_data[clean_name] = (content, f.content_type or "application/octet-stream")

    image_paths = _image_file_paths(file_data)
    yolo_class_map = _build_yolo_class_map(file_data)
    open_images_class_map = _build_open_images_class_map(file_data)
    for path in file_data:
        lower = path.lower()
        if lower.endswith(("classes.txt", "labels.txt", "obj.names", "data.yaml", "data.yml", "dataset.yaml", "dataset.yml")):
            annotation_sidecars.add(path)
        if lower.endswith(".csv") and "class-descriptions" in lower:
            annotation_sidecars.add(path)

    # Additional object-detection import formats.
    for clean_name, (content, _ct) in list(file_data.items()):
        stem, ext = _splitext_lower(clean_name)
        bare_stem = stem.split("/")[-1]

        if ext == ".xml":
            try:
                root = ET.fromstring(content.decode("utf-8", errors="ignore"))
                if root.tag != "annotation":
                    continue
                filename_tag = root.findtext("filename", default="").strip()
                image_ref = _find_matching_image_path(filename_tag or bare_stem, image_paths)
                if not image_ref:
                    image_ref = _find_matching_image_path(bare_stem, image_paths)
                if not image_ref:
                    continue
                boxes: list[dict] = []
                for obj in root.findall("object"):
                    label = (obj.findtext("name", default="") or "").strip()
                    bnd = obj.find("bndbox")
                    if bnd is None:
                        continue
                    try:
                        x1 = float(bnd.findtext("xmin", default="0"))
                        y1 = float(bnd.findtext("ymin", default="0"))
                        x2 = float(bnd.findtext("xmax", default="0"))
                        y2 = float(bnd.findtext("ymax", default="0"))
                    except ValueError:
                        continue
                    box = _box_xyxy_to_xywh(label, x1, y1, x2, y2)
                    if box:
                        boxes.append(box)
                if boxes:
                    _record_annotation_boxes(annotations_boxes, image_ref, boxes)
                    _record_annotation_label(annotations_labels, image_ref, boxes[0].get("label", ""))
                    annotation_sidecars.add(clean_name)
            except Exception:
                pass
            continue

        if ext == ".txt":
            image_ref = _find_matching_image_path(clean_name, image_paths) or _find_matching_image_path(bare_stem, image_paths)
            if not image_ref or image_ref == clean_name:
                # Plain .txt files in folder uploads are most commonly YOLO
                # sidecars or misc metadata like README.txt, not raw samples.
                annotation_sidecars.add(clean_name)
                continue
            # An empty sidecar is Darknet's explicit "this image is background"
            # declaration and carries no coordinates, so it is handled before
            # the dimension read: requiring decodable image dimensions to
            # interpret a file that contains no numbers would drop the
            # declaration for any image PIL cannot open.
            _txt_lines = [
                ln.strip()
                for ln in content.decode("utf-8", errors="ignore").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            if not _txt_lines:
                _record_annotation_boxes(annotations_boxes, image_ref, [])
                background_paths.update(_path_variants(image_ref))
                annotation_sidecars.add(clean_name)
                continue

            img_bytes = file_data.get(image_ref, (b"", ""))[0]
            img_w, img_h = _read_image_size(img_bytes)
            if not img_w or not img_h:
                annotation_sidecars.add(clean_name)
                continue
            boxes: list[dict] = []
            valid = True
            for raw_line in content.decode("utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    valid = False
                    break
                try:
                    cls_idx = int(float(parts[0]))
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h
                except ValueError:
                    valid = False
                    break
                label = yolo_class_map.get(cls_idx, str(cls_idx))
                boxes.append({
                    "label": _normalize_box_label(label),
                    "x": cx - (bw / 2.0),
                    "y": cy - (bh / 2.0),
                    "w": bw,
                    "h": bh,
                })
            # `boxes` cannot be empty here — the no-content case returned above —
            # so `if valid` is equivalent to the old `if valid and boxes`.
            if valid:
                _record_annotation_boxes(annotations_boxes, image_ref, boxes)
                _record_annotation_label(annotations_labels, image_ref, boxes[0].get("label", ""))
                annotation_sidecars.add(clean_name)
            elif ext == ".txt":
                annotation_sidecars.add(clean_name)
            continue

        if ext == ".json":
            try:
                parsed = json.loads(content.decode("utf-8"))
            except Exception:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("images"), list) and isinstance(parsed.get("annotations"), list):
                categories = {
                    int(cat.get("id")): str(cat.get("name", "")).strip()
                    for cat in parsed.get("categories", [])
                    if isinstance(cat, dict) and cat.get("id") is not None
                }
                image_lookup: dict[Any, str] = {}
                for image in parsed.get("images", []):
                    if not isinstance(image, dict):
                        continue
                    image_id = image.get("id")
                    file_name = _clean_rel_path(image.get("file_name", ""))
                    if image_id is None or not file_name:
                        continue
                    image_ref = _find_matching_image_path(file_name, image_paths)
                    if image_ref:
                        image_lookup[image_id] = image_ref
                boxes_by_path: dict[str, list[dict]] = {}
                for ann in parsed.get("annotations", []):
                    if not isinstance(ann, dict):
                        continue
                    image_ref = image_lookup.get(ann.get("image_id"))
                    bbox = ann.get("bbox")
                    if not image_ref or not isinstance(bbox, list) or len(bbox) < 4:
                        continue
                    try:
                        x, y, w, h = map(float, bbox[:4])
                    except ValueError:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    label = categories.get(int(ann.get("category_id", -1)), str(ann.get("category_id", "")))
                    boxes_by_path.setdefault(image_ref, []).append({
                        "label": _normalize_box_label(label),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    })
                if boxes_by_path:
                    for image_ref, boxes in boxes_by_path.items():
                        _record_annotation_boxes(annotations_boxes, image_ref, boxes)
                        _record_annotation_label(annotations_labels, image_ref, boxes[0].get("label", ""))
                    annotation_sidecars.add(clean_name)
            continue

        if ext == ".csv":
            text = content.decode("utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            fieldnames = set(reader.fieldnames or [])
            required = {"ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"}
            if not required.issubset(fieldnames):
                continue
            boxes_by_path: dict[str, list[dict]] = {}
            for row in reader:
                image_id = _clean_rel_path(row.get("ImageID", ""))
                image_ref = _find_matching_image_path(image_id, image_paths) or _find_matching_image_path(image_id + ".jpg", image_paths)
                if not image_ref:
                    continue
                img_bytes = file_data.get(image_ref, (b"", ""))[0]
                img_w, img_h = _read_image_size(img_bytes)
                if not img_w or not img_h:
                    continue
                try:
                    x1 = float(row.get("XMin", "0")) * img_w
                    x2 = float(row.get("XMax", "0")) * img_w
                    y1 = float(row.get("YMin", "0")) * img_h
                    y2 = float(row.get("YMax", "0")) * img_h
                except ValueError:
                    continue
                label_key = (row.get("LabelName", "") or "").strip()
                label = open_images_class_map.get(label_key, label_key)
                box = _box_xyxy_to_xywh(label, x1, y1, x2, y2)
                if box:
                    boxes_by_path.setdefault(image_ref, []).append(box)
            if boxes_by_path:
                for image_ref, boxes in boxes_by_path.items():
                    _record_annotation_boxes(annotations_boxes, image_ref, boxes)
                    _record_annotation_label(annotations_labels, image_ref, boxes[0].get("label", ""))
                annotation_sidecars.add(clean_name)
            continue

    # ── Step 2: Create samples ───────────────────────────────────────────────
    # Resolve all of the project's labels once. Per-file/per-box label lookups
    # then hit this dict instead of issuing a SELECT each time (was an N+1 the
    # size of the batch).
    label_cache: dict[str, Label] = {
        l.name: l for l in db.query(Label).filter(Label.project_id == project_id).all()
    }

    loop = asyncio.get_event_loop()

    async def _flush_uploads(batch: list) -> None:
        # boto3 is blocking; running it in the threadpool keeps the event loop
        # free, and awaiting a bounded batch at a time means at most
        # _BATCH_UPLOAD_CONCURRENCY objects are uploading (and held) at once.
        if not batch:
            return
        try:
            await asyncio.gather(*(
                loop.run_in_executor(
                    None,
                    lambda d=data, k=key, c=ctype: storage.upload_bytes(d, k, content_type=c),
                )
                for data, key, ctype in batch
            ))
        except (ClientError, BotoCoreError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Object storage upload failed: {exc}",
            )
        batch.clear()

    created: list = []
    pending_uploads: list = []
    # Materialise the data-file names up front so we can pop each file's bytes
    # out of `file_data` as it is processed, releasing the buffer slot instead of
    # holding every file until the request returns.
    data_names = [n for n in list(file_data.keys()) if n not in annotation_sidecars]

    for clean_name in data_names:
        content, ct = file_data.pop(clean_name)
        parts = clean_name.split("/")
        bare_filename = parts[-1]

        # Folder-based label: always use the immediate parent folder (parts[-2]).
        # This handles both label_name/file.ext and split/label_name/file.ext
        # layouts because parts[-2] is the label folder in either case.
        folder_label: Optional[str] = _resolve_folder_label_from_path(clean_name)

        # Label resolution priority:
        #  1. info.labels annotation  (by full path)
        #  2. info.labels annotation  (by bare filename)
        #  3. Folder prefix           (immediate parent directory name)
        #  4. EI dict-form label      (weak_annotations_labels — see below)
        #  5. Explicit label_name form field
        # weak_annotations_labels is deprioritized below folder inference
        # because some annotation tools dump a per-image filename hash into
        # the dict-form label field instead of a real class name — folder
        # structure is trusted over that when both are present. But it must
        # still be consulted somewhere, otherwise a plain classification
        # sample with no folder subdir and no box-derived label (exactly
        # what this app's own dataset exporter produces) loses its label on
        # reimport, since the exporter always writes dict-form labels.
        # Samples with no label from any of these sources remain Unlabeled.
        inferred_label: Optional[str] = (
            annotations_labels.get(clean_name)
            or annotations_labels.get(bare_filename)
            or folder_label
            or weak_annotations_labels.get(clean_name)
            or weak_annotations_labels.get(bare_filename)
            or label_name
        )

        # Bounding boxes from info.labels for this file (object detection).
        # Presence-based, NOT truthiness: `[]` is a recorded empty annotation
        # set (the background convention) and `or` would fall through it.
        if clean_name in annotations_boxes:
            inferred_boxes: Optional[list] = annotations_boxes[clean_name]
        elif bare_filename in annotations_boxes:
            inferred_boxes = annotations_boxes[bare_filename]
        else:
            inferred_boxes = None

        # Real boxes always win over a background declaration: two sidecars in
        # the same upload can disagree (an info.labels entry with boxes next to
        # an empty .txt), and an image with an annotated object is not a
        # negative whatever else was said about it.
        _is_background = (
            not inferred_boxes
            and (clean_name in background_paths or bare_filename in background_paths)
        )

        # A background image asserts "none of the project's classes are here",
        # so it must not pick up a top-level class from its folder name or the
        # form's label_name — that would turn the negative back into a positive
        # with no boxes, which is exactly the state Phase 0 had to stop treating
        # as a centre-of-frame object.
        resolved_label_id = (
            _get_or_create_label_cached(inferred_label, project_id, db, label_cache)
            if inferred_label and not _is_background else None
        )
        # sample_type priority:
        #  1. EI "category" field from info.labels (most explicit — exporter wrote it)
        #  2. First recognised split folder in the path (handles both bare and
        #     webkitRelativePath-style paths that include the root folder name)
        #  3. Form-field value (user's UI selection, used for single-type uploads)
        _cat_type = (
            annotations_category.get(clean_name)
            or annotations_category.get(bare_filename)
            or _resolve_split_from_path(clean_name)
        )
        effective_sample_type = _cat_type if _cat_type else sample_type
        resolved_type = _resolve_sample_type(effective_sample_type, bare_filename)
        sensor_type, freq, duration = _extract_metadata(bare_filename, content)

        # Build extra_metadata; normalize geometry keys and resolve label_id per box.
        extra: dict = {}
        if inferred_boxes:
            resolved_boxes = []
            for box in inferred_boxes:
                box_copy = _normalize_box_geometry(dict(box))
                box_lbl_name = _normalize_box_label(box_copy.get("label"))
                if box_lbl_name:
                    box_copy["label_id"] = _get_or_create_label_cached(
                        box_lbl_name, project_id, db, label_cache
                    )
                resolved_boxes.append(box_copy)
            extra["boundingBoxes"] = resolved_boxes
        elif _is_background:
            # Store the marker plus the empty box list the sidecar declared, so
            # a re-export reproduces `boundingBoxes: []` and the round-trip is
            # lossless in both directions.
            extra[IS_BACKGROUND_KEY] = True
            extra["boundingBoxes"] = []
        elif inferred_boxes is not None:
            # Recorded-but-empty without a background declaration: keep the
            # empty list (faithful to the sidecar) but do NOT infer background.
            extra["boundingBoxes"] = []

        if bare_filename.lower().endswith(".csv"):
            feature_names = _parse_csv_feature_names(content)
            if feature_names:
                extra["feature_names"] = feature_names

        key = storage.sample_key(project_id, bare_filename)
        # Queue the (blocking) S3 upload; it runs in the threadpool, in parallel,
        # bounded to _BATCH_UPLOAD_CONCURRENCY in flight at a time.
        pending_uploads.append((content, key, ct))

        sample = Sample(
            project_id=project_id,
            label_id=resolved_label_id,
            filename=clean_name,       # preserve relative path for context
            storage_key=key,
            sensor_type=sensor_type,
            frequency_hz=freq,
            duration_ms=duration,
            sample_type=resolved_type,
            file_size_bytes=len(content),
            extra_metadata=extra,
            uploaded_by=current_user.id,
        )
        db.add(sample)
        created.append(sample)

        if len(pending_uploads) >= _BATCH_UPLOAD_CONCURRENCY:
            await _flush_uploads(pending_uploads)

    # Upload any remaining files past the last full batch.
    await _flush_uploads(pending_uploads)

    # flush (not commit) so the new rows + labels get their ids and Python-side
    # defaults (created_at) populated while the objects stay un-expired — we can
    # then serialize straight from them without a refresh()/lazy-load per row.
    db.flush()
    label_index = {
        "by_id":   {l.id: l for l in label_cache.values()},
        "by_name": dict(label_cache),
    }
    items = [_sample_to_dict(s, db, label_index=label_index) for s in created]
    db.commit()

    return {
        "uploaded": len(created),
        "items": items,
    }


@router.post("/ingest/{project_id}", status_code=201)
async def ingest_sample(
    project_id: str,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Ingest sensor data from device SDK in PetalEdge-compatible JSON format.

    Payload:
    {
      "protected": {"ver": "v1", "alg": "none"},
      "signature": "...",
      "payload": {
        "device_type": "ESP32",
        "interval_ms": 10,
        "sensors": [{"name": "accX", "units": "m/s2"}, ...],
        "values": [[0.1, 0.2, 0.3], ...]
      }
    }
    """
    project = assert_project_owner(db, project_id, current_user)

    inner = payload.get("payload", payload)
    data_json = json.dumps(inner).encode()
    filename = f"ingest_{inner.get('device_type', 'device')}.json"
    key = storage.sample_key(project_id, filename)
    storage.upload_bytes(data_json, key, content_type="application/json")

    interval_ms = inner.get("interval_ms", 10)
    values = inner.get("values", [])
    sensors = inner.get("sensors", [])

    extra_metadata: dict = {}
    feature_names = [s.get("name") for s in sensors if isinstance(s, dict) and s.get("name")]
    if feature_names:
        extra_metadata["feature_names"] = feature_names

    sample = Sample(
        project_id=project_id,
        filename=filename,
        storage_key=key,
        sensor_type=inner.get("device_type", "unknown"),
        frequency_hz=1000.0 / interval_ms if interval_ms else 100.0,
        num_channels=len(sensors),
        num_samples=len(values),
        file_size_bytes=len(data_json),
        extra_metadata=extra_metadata,
        uploaded_by=current_user.id,
    )
    db.add(sample)
    db.commit()
    db.refresh(sample)
    return _sample_to_dict(sample, db)


@router.get("/project/{project_id}")
def list_samples(
    project_id: str,
    label_id: Optional[str] = Query(None),
    sample_type: Optional[SampleType] = Query(None),
    labeled_only: bool = Query(False),
    unlabeled_only: bool = Query(False),
    skip: int = Query(0),
    limit: int = Query(50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    assert_project_owner(db, project_id, current_user)
    # Deterministic order: makes "match the Dataset page's order" (see the
    # Labeling Queue plan, §2.3) a guarantee instead of an accident of
    # Postgres' scan strategy. id is a tiebreaker for same-millisecond
    # uploads.
    q = (
        db.query(Sample)
        .filter(Sample.project_id == project_id)
        .order_by(Sample.created_at.asc(), Sample.id.asc())
    )

    if labeled_only:
        from sqlalchemy import and_
        q = q.filter(and_(Sample.label_id.isnot(None), Sample.label_id != ""))

    if sample_type:
        q = q.filter(Sample.sample_type == sample_type)

    if label_id:
        target_label = db.query(Label).filter(Label.id == label_id).first()
        if not target_label:
            return {"total": 0, "items": []}

        matching_samples = [
            sample for sample in q.all()
            if _sample_matches_label_filter(sample, label_id, target_label.name)
        ]
        total = len(matching_samples)
        if limit == 0:
            return {"total": total, "items": []}
        samples = matching_samples[skip: skip + limit]
        label_index = _build_label_index(db, project_id)
        return {"total": total, "items": [_sample_to_dict(s, db, label_index=label_index) for s in samples]}

    if unlabeled_only:
        # is_sample_unlabeled is a Python predicate (it inspects
        # extra_metadata.boundingBoxes, which the plain-JSON column here
        # can't cheaply express as a WHERE clause — see
        # app/services/labeling_status.py). Same shape as the label_id
        # branch above: evaluate over the filtered set, then slice. This is
        # the canonical "unlabeled" definition shared with the
        # labeling-status-summary endpoint and AI Labeling's skip_labeled,
        # so this filter, that endpoint's count, and AI Labeling's sample
        # selection can never disagree.
        matching_samples = [s for s in q.all() if is_sample_unlabeled(s)]
        total = len(matching_samples)
        if limit == 0:
            return {"total": total, "items": []}
        samples = matching_samples[skip: skip + limit]
        label_index = _build_label_index(db, project_id)
        return {"total": total, "items": [_sample_to_dict(s, db, label_index=label_index) for s in samples]}

    total = q.count()

    # limit=0 means "count only" — skip the expensive row fetch (used for tab badge counts)
    if limit == 0:
        return {"total": total, "items": []}
    samples = q.offset(skip).limit(limit).all()
    label_index = _build_label_index(db, project_id)
    return {"total": total, "items": [_sample_to_dict(s, db, label_index=label_index) for s in samples]}


@router.get("/project/{project_id}/labeling-status-summary")
def labeling_status_summary(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Canonical labeling-status counts for a project (Labeling Queue plan,
    Phase 1 §1.F) — powers the Dataset page's outstanding-work readouts.

    Projects a narrow column set and evaluates the shared predicate in
    Python rather than fetching full sample rows, so this is a small JSON
    object rather than the 10k-row metadata fetch the Dataset page used to
    do for the same numbers. No sample_type/label_id filter params: this is
    always the whole project (see the plan's §2.4 for why a filtered
    variant would be actively misleading here).
    """
    assert_project_owner(db, project_id, current_user)
    rows = (
        db.query(Sample.filename, Sample.sample_type, Sample.extra_metadata)
        .filter(Sample.project_id == project_id)
        .all()
    )
    return summarize(rows)


@router.get("/project/{project_id}/unlabeled-queue")
def unlabeled_queue(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Ordered id list backing the Labeling Queue (Labeling Queue plan, Phase 3
    §3.A) — everything in the project the shared predicate
    (app/services/labeling_status.py) still considers unlabeled, across all
    three splits at once (see the plan's §2.4: the queue has no
    sample_type/label_id filter, unlike list_samples/unlabeled_only).

    Same created_at/id ordering as list_samples (§2.3) and the same narrow
    per-row projection as labeling_status_summary — evaluates the predicate
    in Python over a small column set rather than a SQL filter the plain-JSON
    extra_metadata column can't cheaply express. Deliberately omits
    extra_metadata from the response: the queue only needs ids and display
    fields up front, and fetches a sample's boxes/image via the existing
    GET /samples/{id} and GET /samples/{id}/download only once that sample
    becomes current (§3.C).
    """
    assert_project_owner(db, project_id, current_user)
    rows = (
        db.query(Sample.id, Sample.filename, Sample.sample_type, Sample.created_at, Sample.extra_metadata)
        .filter(Sample.project_id == project_id)
        .order_by(Sample.created_at.asc(), Sample.id.asc())
        .all()
    )
    items = [
        {
            "id": row.id,
            "filename": row.filename,
            "sample_type": row.sample_type,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
        if is_sample_unlabeled(row)
    ]
    return {"total": len(items), "items": items}


@router.get("/project/{project_id}/feature-axes")
def get_feature_axes(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The dataset's feature/axis names, as parsed from the most recently
    uploaded CSV (or device-declared sensor names) that carries them — the
    values a Motion impulse's first processing block should treat as its
    input features. Empty when no sample in the project has feature metadata
    yet (e.g. only image samples, or CSVs with no readable header)."""
    assert_project_owner(db, project_id, current_user)

    recent_samples = (
        db.query(Sample)
        .filter(Sample.project_id == project_id)
        .order_by(Sample.created_at.desc())
        .limit(25)
        .all()
    )
    features: list[str] = []
    for s in recent_samples:
        meta = s.extra_metadata or {}
        names = meta.get("feature_names") if isinstance(meta, dict) else None
        if isinstance(names, list) and names:
            features = [str(n) for n in names]
            break

    return {"features": features}


@router.get("/project/{project_id}/labels-summary")
def labels_summary(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Distinct *assigned* labels for a project — used by the impulse builder
    to populate the Learning block and Output features cards without having to
    download every sample row (the previous client-side aggregation pulled up
    to 10k samples with full metadata just to count classes, which made the
    cards stall behind a multi-second JSON payload).

    Two sources merged, then deduped + skip-set applied:
      1. Classification samples — `Sample.label_id` joined to `Label.name`.
         (The Sample model has no `label_name` column; that field on the
         list response is derived via the `s.label` relationship — so we
         JOIN, not column-select.)
      2. Object-detection samples — `extra_metadata.boundingBoxes[*].label`
         (canonical key is camelCase `boundingBoxes`, confirmed by the
         dataset code and `_sample_to_dict`).

    Returned shape: `{count: int, names: list[str]}`."""
    import logging
    logger = logging.getLogger(__name__)

    assert_project_owner(db, project_id, current_user)

    SKIP = {"unlabeled", "unlabelled", "unknown"}
    seen: dict[str, str] = {}  # lowercase key → first-seen display casing
    skipped: list[str] = []

    # 1) Classification labels — JOIN through Label so we read the actual name.
    name_rows = (
        db.query(Label.name)
        .join(Sample, Sample.label_id == Label.id)
        .filter(Sample.project_id == project_id)
        .distinct()
        .all()
    )
    for (raw,) in name_rows:
        text = (raw or "").strip()
        if not text:
            continue
        if text.lower() in SKIP:
            skipped.append(text)
            continue
        seen.setdefault(text.lower(), text)

    # 2) Bounding-box labels — scan extra_metadata for object-detection samples.
    box_rows = (
        db.query(Sample.extra_metadata)
        .filter(Sample.project_id == project_id)
        .filter(Sample.extra_metadata.isnot(None))
        .all()
    )
    box_samples_with_boxes = 0
    for (meta,) in box_rows:
        if not isinstance(meta, dict):
            continue
        boxes = meta.get("boundingBoxes")
        if not isinstance(boxes, list) or not boxes:
            continue
        box_samples_with_boxes += 1
        for b in boxes:
            if not isinstance(b, dict):
                continue
            raw = b.get("label")
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue
            if text.lower() in SKIP:
                skipped.append(text)
                continue
            seen.setdefault(text.lower(), text)

    names = list(seen.values())
    logger.info(
        "[labels-summary] project=%s classification_distinct=%d bbox_samples=%d "
        "merged_distinct=%d skipped=%s names=%s",
        project_id, len(name_rows), box_samples_with_boxes,
        len(names), skipped, names,
    )
    return {"count": len(names), "names": names}


@router.get("/{sample_id}")
def get_sample(
    sample_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sample = assert_sample_owner(db, sample_id, current_user)
    return _sample_to_dict(sample, db)


@router.get("/{sample_id}/download")
def download_sample(
    sample_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sample = assert_sample_owner(db, sample_id, current_user)
    url = storage.get_presigned_url(sample.storage_key)
    return {"url": url}


@router.get("/{sample_id}/signal")
def get_sample_signal(
    sample_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Decoded {axes, values, frequency_hz, duration_ms, num_channels,
    num_samples} for a time-series recording — same decoder Phase 3's window
    segmentation uses (app/motion/dsp/payloads.py), so this view and training
    can never disagree on what the stored bytes mean. Powers the frontend's
    waveform preview and expanded Signal Viewer (sampleSignalCache.ts)."""
    sample = assert_sample_owner(db, sample_id, current_user)
    raw_bytes = storage.download_bytes(sample.storage_key)
    try:
        return signal_decode.decode(raw_bytes, sample)
    except signal_decode.SignalDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.patch("/{sample_id}/label")
def assign_label(
    sample_id: str,
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sample = assert_sample_owner(db, sample_id, current_user)
    label_id = body.get("label_id") or None
    if label_id:
        # The label must belong to the same project as the sample.
        label = db.query(Label).filter(
            Label.id == label_id,
            Label.project_id == sample.project_id,
        ).first()
        if not label:
            raise HTTPException(status_code=404, detail="Label not found")
    _set_sample_label(sample, label_id, label.name if label_id else None)
    db.commit()
    return _sample_to_dict(sample, db)


@router.patch("/{sample_id}/split")
def update_split(
    sample_id: str,
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Move sample between training / testing sets."""
    sample = assert_sample_owner(db, sample_id, current_user)
    sample.sample_type = body.get("sample_type", "training")
    db.commit()
    return _sample_to_dict(sample, db)


@router.patch("/{sample_id}")
def update_sample(
    sample_id: str,
    body: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generic update endpoint for samples (rename, disable, extra_metadata)."""
    sample = assert_sample_owner(db, sample_id, current_user)
    meta_touched = False

    if "filename" in body:
        sample.filename = body["filename"]

    if "is_disabled" in body:
        import copy
        meta = copy.deepcopy(sample.extra_metadata or {})
        meta["is_disabled"] = body["is_disabled"]
        sample.extra_metadata = meta
        meta_touched = True

    if IS_BACKGROUND_KEY in body:
        # Mirrors is_disabled above: a single JSON key, no schema change.
        # Setting it False removes the key entirely so "absent" and "explicitly
        # not background" stay one state — otherwise the export/round-trip
        # tests below would have to distinguish two encodings of the same thing.
        import copy
        meta = copy.deepcopy(sample.extra_metadata or {})
        if body[IS_BACKGROUND_KEY]:
            # A background marker asserts "this image contains none of the
            # project's classes" — the same invariant the AI-labeling
            # apply-predictions path enforces (see ai_labeling.py). Setting
            # it on a sample that still carries real box annotations or a
            # label would leave contradictory state on disk: the training
            # loader treats the background flag as authoritative and wipes
            # the boxes at load time (see training_worker.py), so the boxes
            # would silently stop training the model while still looking
            # "labeled" everywhere else. Clear them here instead of leaving
            # that footgun for later.
            meta[IS_BACKGROUND_KEY] = True
            meta["boundingBoxes"] = []
            sample.label_id = None
        else:
            meta.pop(IS_BACKGROUND_KEY, None)
        sample.extra_metadata = meta
        meta_touched = True

    if "extra_metadata" in body and isinstance(body["extra_metadata"], dict):
        # Full replace — caller sends the entire metadata dict they want stored.
        # This is how the dataset page's annotation editor persists drawn
        # boxes (samplesApi.update({extra_metadata: {...}})), so a box saved
        # onto a previously-background-marked sample lands here, not in the
        # IS_BACKGROUND_KEY branch above.
        # Using a copy ensures SQLAlchemy detects the mutation on the JSON column.
        import copy
        sample.extra_metadata = copy.deepcopy(body["extra_metadata"])
        meta_touched = True

    if meta_touched and isinstance(sample.extra_metadata, dict):
        # Whichever branch above produced the final metadata, it must not
        # leave is_background sitting next to real boxes/label — e.g. drawing
        # a box on a background-marked sample must clear the marker even
        # though this request never touched IS_BACKGROUND_KEY directly.
        sample.extra_metadata = enforce_background_invariant(
            sample.extra_metadata, label_id=sample.label_id
        )

    flag_modified(sample, "extra_metadata")
    db.commit()
    db.refresh(sample)
    return _sample_to_dict(sample, db)

class BulkUpdateRequest(BaseModel):
    sample_ids: List[str]
    action: str
    value: Optional[Any] = None

@router.post("/bulk-update")
def bulk_update_samples(
    body: BulkUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generic bulk processor handling Edge Impulse bulk actions."""
    if not body.sample_ids or not body.action:
         raise HTTPException(status_code=400, detail="Missing sample_ids or action")
         
    from sqlalchemy.orm.attributes import flag_modified
    import copy

    updated = 0
    failed_ids = []
    
    for sid in body.sample_ids:
        try:
            s = (
                db.query(Sample)
                .join(Project, Sample.project_id == Project.id)
                .filter(Sample.id == sid, Project.owner_id == current_user.id)
                .first()
            )
            if not s:
                failed_ids.append(sid)
                continue
                
            if body.action == "label":
                val_id = body.value or None
                if val_id:
                    label = db.query(Label).filter(Label.id == val_id).first()
                    if label:
                        _set_sample_label(s, val_id, label.name)
                        db.commit()
                        updated += 1
                    else:
                        failed_ids.append(sid)
                        db.rollback()
                else:
                    _set_sample_label(s, None, None)
                    db.commit()
                    updated += 1
                
            elif body.action == "split":
                s.sample_type = body.value
                db.commit()
                updated += 1
                
            elif body.action == "disable":
                m = copy.deepcopy(s.extra_metadata or {})
                m["is_disabled"] = True
                s.extra_metadata = m
                flag_modified(s, "extra_metadata")
                db.commit()
                updated += 1
                
            elif body.action == "enable":
                m = copy.deepcopy(s.extra_metadata or {})
                m["is_disabled"] = False
                s.extra_metadata = m
                flag_modified(s, "extra_metadata")
                db.commit()
                updated += 1
                
            elif body.action == "metadata_add":
                m = copy.deepcopy(s.extra_metadata or {})
                if isinstance(body.value, dict):
                    m.update(body.value)
                    # Same invariant as the single-sample PATCH endpoint: a
                    # background marker must not coexist with real box/label
                    # annotations, or the training loader silently wipes the
                    # boxes while everything else still displays as labeled.
                    # The dataset page's "Mark as background" bulk action
                    # rides this generic path (no dedicated endpoint), so it
                    # has to be enforced here too.
                    if body.value.get(IS_BACKGROUND_KEY):
                        m["boundingBoxes"] = []
                        s.label_id = None
                    else:
                        # Reverse direction — an arbitrary metadata_add
                        # payload that adds real boxes to a sample already
                        # marked background must clear the marker.
                        m = enforce_background_invariant(m, label_id=s.label_id)
                s.extra_metadata = m
                flag_modified(s, "extra_metadata")
                db.commit()
                updated += 1
                
            elif body.action == "metadata_clear_key":
                m = copy.deepcopy(s.extra_metadata or {})
                if isinstance(body.value, str) and body.value in m:
                    del m[body.value]
                s.extra_metadata = m
                flag_modified(s, "extra_metadata")
                db.commit()
                updated += 1
                
            elif body.action == "metadata_clear_all":
                # Preserve the flags that are dataset *state* rather than
                # user-authored metadata: disabling a sample and marking it a
                # training negative both survive a metadata wipe, because
                # neither is something the user entered in the Metadata dialog.
                _meta = s.extra_metadata or {}
                is_dis = _meta.get("is_disabled", False)
                kept: dict = {"is_disabled": is_dis}
                if _meta.get(IS_BACKGROUND_KEY):
                    kept[IS_BACKGROUND_KEY] = True
                s.extra_metadata = kept
                flag_modified(s, "extra_metadata")
                db.commit()
                updated += 1
        except Exception:
            failed_ids.append(sid)
            db.rollback()

    return {"status": "success", "updated": updated, "failed": failed_ids}


class BulkDeleteRequest(BaseModel):
    sample_ids: List[str]

@router.post("/bulk-delete")
def bulk_delete_samples(
    body: BulkDeleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Bulk deletion for action bar."""
    import logging
    logger = logging.getLogger(__name__)

    if not body.sample_ids:
        return {"status": "success", "deleted": 0, "failed": []}

    deleted = 0
    failed_ids = []
    affected_project_ids = set()

    for sid in body.sample_ids:
        try:
            s = (
                db.query(Sample)
                .join(Project, Sample.project_id == Project.id)
                .filter(Sample.id == sid, Project.owner_id == current_user.id)
                .first()
            )
            if not s:
                failed_ids.append(sid)
                continue

            # Capture the project before deletion so we can prune its orphans.
            if s.project_id:
                affected_project_ids.add(s.project_id)

            try:
                storage.delete_file(s.storage_key)
            except Exception as e:
                logger.warning(f"Storage deletion failed for {s.storage_key}: {e}")

            # Clear associated records before deleting to prevent IntegrityError.
            _clear_associated_sample_records(db, sid)

            db.delete(s)
            db.commit()
            deleted += 1

        except Exception as e:
            failed_ids.append(sid)
            db.rollback()

    # Auto-prune labels left orphaned by the deletions. Never fail the delete.
    for pid in affected_project_ids:
        try:
            _prune_orphans_in_project(db, pid)
        except Exception as e:
            logger.warning(f"Orphan-label prune failed for project {pid}: {e}")
            db.rollback()

    return {"status": "success", "deleted": deleted, "failed": failed_ids}


@router.delete("/{sample_id}", status_code=204)
def delete_sample(
    sample_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sample = assert_sample_owner(db, sample_id, current_user)

    # Capture the project before deletion so we can prune its orphans.
    project_id = sample.project_id

    storage.delete_file(sample.storage_key)

    # Clear associated records before deleting.
    _clear_associated_sample_records(db, sample_id)

    db.delete(sample)
    db.commit()

    # Auto-prune labels left orphaned by the deletion. Never fail the delete.
    if project_id:
        try:
            _prune_orphans_in_project(db, project_id)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Orphan-label prune failed for project {project_id}: {e}"
            )
            db.rollback()
    return Response(status_code=204)
