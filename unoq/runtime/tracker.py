"""Standalone IoU-based multi-object tracker for the PXE runtime.

Operates on detection dicts with shape:
    {"label": str, "confidence": float,
     "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}}

Returns detection dicts augmented with:
    "object_id":  int   — stable cross-frame identifier
    "is_carried": bool  — True when emitted as a keep-grace carry-forward

This is a self-contained re-implementation of the same algorithm in
backend/app/services/post_processing/tracker.py.  It is intentionally
kept separate so the UNO Q device package has no dependency on backend code.

State dict shape (JSON-serializable, pass back as next-frame input):
    {
        "tracks": {
            "<id>": {
                "object_id": int,
                "x1": float, "y1": float, "x2": float, "y2": float,
                "label": str,
                "confidence": float,
                "missed_frames": int,
                "observations": int,
                "box_history": [[x1, y1, x2, y2], ...]
            }
        },
        "next_id": int
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─── Internal track record ─────────────────────────────────────────────────────

@dataclass
class _Track:
    object_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float = 0.0
    missed_frames: int = 0
    observations: int = 1
    box_history: list = field(default_factory=list)


# ─── State encode / decode ─────────────────────────────────────────────────────

def _decode_state(state: dict) -> Tuple[Dict[int, _Track], int]:
    tracks: Dict[int, _Track] = {}
    for k, v in state.get("tracks", {}).items():
        t = _Track(**v)
        if not t.box_history:
            t.box_history = [[t.x1, t.y1, t.x2, t.y2]]
        tracks[int(k)] = t
    return tracks, int(state.get("next_id", 1))


def _encode_state(tracks: Dict[int, _Track], next_id: int) -> dict:
    return {
        "tracks": {
            str(tid): {
                "object_id":    t.object_id,
                "x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2,
                "label":        t.label,
                "confidence":   t.confidence,
                "missed_frames": t.missed_frames,
                "observations": t.observations,
                "box_history":  t.box_history,
            }
            for tid, t in tracks.items()
        },
        "next_id": next_id,
    }


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def _iou(ax1: float, ay1: float, ax2: float, ay2: float,
         bx1: float, by1: float, bx2: float, by2: float) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


def _center_dist(ax1: float, ay1: float, ax2: float, ay2: float,
                 bx1: float, by1: float, bx2: float, by2: float) -> float:
    """Euclidean distance between box centers (normalized coords → [0, √2])."""
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _smoothed_box(history: list) -> Tuple[float, float, float, float]:
    n = len(history)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        sum(b[0] for b in history) / n,
        sum(b[1] for b in history) / n,
        sum(b[2] for b in history) / n,
        sum(b[3] for b in history) / n,
    )


def _det_bbox(det: Dict) -> Tuple[float, float, float, float]:
    b = det.get("bbox", {})
    return b.get("x1", 0.0), b.get("y1", 0.0), b.get("x2", 0.0), b.get("y2", 0.0)


# ─── Tracker ───────────────────────────────────────────────────────────────────

class PxeTracker:
    """
    Greedy two-pass multi-object tracker for PXE runtime.

    Matching strategy (same as backend SimpleTracker):
      Pass 1 — primary IoU:  pairs where IoU >= iou_threshold, sorted descending.
      Pass 2 — center-dist fallback: remaining pairs where center-to-center
               distance < center_dist_threshold, sorted ascending.
               Catches fast-moving objects that drift below the IoU threshold.

    Both passes enforce same-label matching.
    State is a plain JSON-serializable dict; instantiate per-call with the
    persisted state or reuse the instance across frames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        keep_grace: int = 3,
        max_observations: int = 5,
        center_dist_threshold: float = 0.2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.keep_grace = keep_grace
        self.max_observations = max_observations
        self.center_dist_threshold = center_dist_threshold

    def update(
        self,
        detections: List[Dict],
        state: dict,
    ) -> Tuple[List[Dict], dict]:
        """
        Match detections to existing tracks and assign stable object IDs.

        Parameters
        ----------
        detections : list of dicts
            Each dict must have 'label', 'confidence', and
            'bbox': {'x1', 'y1', 'x2', 'y2'} keys.
        state : dict
            Tracker state from the previous frame (empty dict on first call).

        Returns
        -------
        (augmented_detections, new_state)
            augmented_detections: same list with 'object_id' and 'is_carried' added.
            new_state: updated state to pass into the next frame.
        """
        tracks, next_id = _decode_state(state)
        existing_ids = set(tracks.keys())

        matched_det: Dict[int, int] = {}
        matched_track: set = set()

        if tracks and detections:
            # ── Pass 1: greedy IoU ────────────────────────────────────────────
            iou_candidates: list = []
            for det_idx, det in enumerate(detections):
                dx1, dy1, dx2, dy2 = _det_bbox(det)
                for tid, track in tracks.items():
                    if track.label != det.get("label"):
                        continue
                    score = _iou(dx1, dy1, dx2, dy2, track.x1, track.y1, track.x2, track.y2)
                    if score >= self.iou_threshold:
                        iou_candidates.append((score, det_idx, tid))

            iou_candidates.sort(reverse=True)
            for _, det_idx, tid in iou_candidates:
                if det_idx not in matched_det and tid not in matched_track:
                    matched_det[det_idx] = tid
                    matched_track.add(tid)

            # ── Pass 2: center-distance fallback ─────────────────────────────
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
            unmatched_tracks = [t for t in existing_ids if t not in matched_track]
            if unmatched_dets and unmatched_tracks:
                dist_candidates: list = []
                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    dx1, dy1, dx2, dy2 = _det_bbox(det)
                    for tid in unmatched_tracks:
                        track = tracks[tid]
                        if track.label != det.get("label"):
                            continue
                        dist = _center_dist(dx1, dy1, dx2, dy2,
                                            track.x1, track.y1, track.x2, track.y2)
                        if dist < self.center_dist_threshold:
                            dist_candidates.append((dist, det_idx, tid))

                dist_candidates.sort()
                for _, det_idx, tid in dist_candidates:
                    if det_idx not in matched_det and tid not in matched_track:
                        matched_det[det_idx] = tid
                        matched_track.add(tid)

        # ── Build result for matched / new detections ─────────────────────────
        result: List[Dict] = []
        for det_idx, det in enumerate(detections):
            dx1, dy1, dx2, dy2 = _det_bbox(det)
            if det_idx in matched_det:
                tid = matched_det[det_idx]
                track = tracks[tid]
                track.box_history.append([dx1, dy1, dx2, dy2])
                if len(track.box_history) > self.max_observations:
                    track.box_history = track.box_history[-self.max_observations:]
                track.x1, track.y1, track.x2, track.y2 = dx1, dy1, dx2, dy2
                track.confidence = det.get("confidence", 0.0)
                track.missed_frames = 0
                track.observations = min(track.observations + 1, self.max_observations)
                sx1, sy1, sx2, sy2 = _smoothed_box(track.box_history)
                result.append({
                    **det,
                    "bbox":      {"x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2},
                    "object_id": track.object_id,
                    "is_carried": False,
                })
            else:
                obj_id = next_id
                next_id += 1
                tracks[obj_id] = _Track(
                    object_id=obj_id,
                    x1=dx1, y1=dy1, x2=dx2, y2=dy2,
                    label=det.get("label", ""),
                    confidence=det.get("confidence", 0.0),
                    box_history=[[dx1, dy1, dx2, dy2]],
                )
                result.append({**det, "object_id": obj_id, "is_carried": False})

        # ── Expire unmatched tracks or emit carry-forwards ────────────────────
        for tid in existing_ids - matched_track:
            track = tracks[tid]
            track.missed_frames += 1
            if track.missed_frames > self.keep_grace:
                del tracks[tid]
            else:
                sx1, sy1, sx2, sy2 = _smoothed_box(track.box_history)
                result.append({
                    "label":      track.label,
                    "confidence": track.confidence,
                    "bbox":       {"x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2},
                    "object_id":  track.object_id,
                    "is_carried": True,
                })

        return result, _encode_state(tracks, next_id)
