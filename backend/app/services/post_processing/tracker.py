"""
IoU-based multi-object tracker.

Assigns stable object IDs across sequential frames using greedy IoU matching
with a center-distance fallback pass to reduce ID breaks on fast-moving objects.
All mutable state lives in a plain serializable dict so it can be stored in a
DB job row or passed over JSON without any special pickling.

Public API
----------
SimpleTracker(iou_threshold, keep_grace, max_observations, center_dist_threshold)
    .update(detections, state) -> (List[Detection], state_dict)

State dict shape
----------------
{
    "tracks": {
        "<id>": {
            "object_id": int,
            "x1": float, "y1": float, "x2": float, "y2": float,  # last RAW box
            "class_name": str,
            "confidence": float,
            "missed_frames": int,
            "observations": int,
            "box_history": [[x1, y1, x2, y2], ...]   # last N raw boxes
        },
        ...
    },
    "next_id": int
}

Matching strategy (two passes, greedy assignment)
-------------------------------------------------
Pass 1 — primary IoU: pairs where IoU >= iou_threshold, sorted descending.
Pass 2 — center-distance fallback: remaining unmatched pairs where
          center-to-center distance < center_dist_threshold, sorted ascending.
          This catches fast-moving objects that drift just below the IoU
          threshold in one frame without spawning a spurious new track.

Matching is geometry-first rather than class-gated. Real detections can
oscillate between nearby labels (for example tiger ↔ zebra) across frames,
and requiring the class name to stay identical causes one physical object to
fragment into many object IDs.

Smoothing
---------
The raw detection box is stored in track.x1..y2 (used for future IoU
matching so there is no lag on moving objects).  The output Detection box is
the mean of the last max_observations raw boxes (jitter reduction only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .types import Detection


# ─── Internal track record ────────────────────────────────────────────────────

@dataclass
class _Track:
    object_id: int
    x1: float       # last raw detection x1 (used for matching, not smoothed)
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float = 0.0
    missed_frames: int = 0
    observations: int = 1
    box_history: list = field(default_factory=list)  # [[x1,y1,x2,y2], ...]


# ─── State serialisation helpers ─────────────────────────────────────────────

def _decode_state(state: dict) -> tuple[dict[int, _Track], int]:
    tracks: dict[int, _Track] = {}
    for k, v in state.get("tracks", {}).items():
        t = _Track(**v)
        if not t.box_history:
            t.box_history = [[t.x1, t.y1, t.x2, t.y2]]
        tracks[int(k)] = t
    return tracks, int(state.get("next_id", 1))


def _encode_state(tracks: dict[int, _Track], next_id: int) -> dict:
    return {
        "tracks": {
            str(tid): {
                "object_id": t.object_id,
                "x1": t.x1,
                "y1": t.y1,
                "x2": t.x2,
                "y2": t.y2,
                "class_name": t.class_name,
                "confidence": t.confidence,
                "missed_frames": t.missed_frames,
                "observations": t.observations,
                "box_history": t.box_history,
            }
            for tid, t in tracks.items()
        },
        "next_id": next_id,
    }


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _iou_coords(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


def _center_dist(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    """Euclidean distance between box centers (normalized coords → [0, √2])."""
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _smoothed_box(history: list) -> tuple[float, float, float, float]:
    """Mean of all boxes in history."""
    n = len(history)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        sum(b[0] for b in history) / n,
        sum(b[1] for b in history) / n,
        sum(b[2] for b in history) / n,
        sum(b[3] for b in history) / n,
    )


# ─── Tracker ──────────────────────────────────────────────────────────────────

class SimpleTracker:
    """
    Greedy two-pass multi-object tracker (IoU primary, center-distance fallback).

    Stateless class — all mutable data lives in the 'state' dict passed to
    and returned from update(), making it safe to instantiate once and reuse
    across frames or re-create per call with a persisted state dict.
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
        detections: List[Detection],
        state: dict,
    ) -> tuple[List[Detection], dict]:
        """
        Match detections to existing tracks and assign stable object IDs.

        Matching (two greedy passes, geometry-first):
          Pass 1 — IoU >= iou_threshold, sorted by IoU descending.
          Pass 2 — center-distance < center_dist_threshold for any pair that
                   survived pass 1 unmatched, sorted by distance ascending.
                   Prevents a fast-moving object from simultaneously spawning
                   a new track and emitting a carry-forward for the old one.

        Class names are allowed to change across frames for the same track.
        This keeps IDs stable when the detector briefly flips between visually
        similar classes on one physical object.

        Track.x1..y2 stores the last *raw* detection position so IoU matching
        is always against the true last-known position with no mean-lag.
        Output Detection coordinates are the mean of the last max_observations
        raw boxes (smoothed for rendering, not fed back to the match loop).

        Unmatched tracks within keep_grace emit carry-forward detections
        (is_carried=True) so they remain visible.  Tracks beyond keep_grace
        are expired.

        Returns (detections_with_ids, new_state).  Matched/new detections
        come first (input order preserved); carry-forwards are appended.
        """
        tracks, next_id = _decode_state(state)
        existing_ids = set(tracks.keys())

        matched_det: dict[int, int] = {}   # det_idx -> track_id
        matched_track: set[int] = set()

        if tracks and detections:
            # ── Pass 1: greedy IoU ────────────────────────────────────────────
            iou_candidates: list[tuple[float, int, int]] = []
            for det_idx, det in enumerate(detections):
                for tid, track in tracks.items():
                    iou = _iou_coords(
                        det.x1, det.y1, det.x2, det.y2,
                        track.x1, track.y1, track.x2, track.y2,
                    )
                    if iou >= self.iou_threshold:
                        iou_candidates.append((iou, det_idx, tid))

            iou_candidates.sort(reverse=True)
            for _, det_idx, tid in iou_candidates:
                if det_idx not in matched_det and tid not in matched_track:
                    matched_det[det_idx] = tid
                    matched_track.add(tid)

            # ── Pass 2: center-distance fallback ─────────────────────────────
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_det]
            unmatched_tracks = [tid for tid in existing_ids if tid not in matched_track]

            if unmatched_dets and unmatched_tracks:
                dist_candidates: list[tuple[float, int, int]] = []
                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    for tid in unmatched_tracks:
                        track = tracks[tid]
                        dist = _center_dist(
                            det.x1, det.y1, det.x2, det.y2,
                            track.x1, track.y1, track.x2, track.y2,
                        )
                        if dist < self.center_dist_threshold:
                            dist_candidates.append((dist, det_idx, tid))

                dist_candidates.sort()
                for _, det_idx, tid in dist_candidates:
                    if det_idx not in matched_det and tid not in matched_track:
                        matched_det[det_idx] = tid
                        matched_track.add(tid)

        # ── Build result for matched / new detections ─────────────────────────
        result: list[Detection] = []
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det:
                tid = matched_det[det_idx]
                track = tracks[tid]
                # Append raw box to sliding window history
                track.box_history.append([det.x1, det.y1, det.x2, det.y2])
                if len(track.box_history) > self.max_observations:
                    track.box_history = track.box_history[-self.max_observations:]
                # Raw position stored for future matching (no mean-lag on moving objects)
                track.x1, track.y1, track.x2, track.y2 = det.x1, det.y1, det.x2, det.y2
                track.class_name = det.class_name
                track.confidence = det.confidence
                track.missed_frames = 0
                track.observations = min(track.observations + 1, self.max_observations)
                # Smoothed box used for output only
                sx1, sy1, sx2, sy2 = _smoothed_box(track.box_history)
                result.append(Detection(
                    x1=sx1, y1=sy1, x2=sx2, y2=sy2,
                    class_name=det.class_name, confidence=det.confidence,
                    object_id=track.object_id,
                    observations=track.observations,
                ))
            else:
                # New track — first box becomes its entire history
                obj_id = next_id
                next_id += 1
                tracks[obj_id] = _Track(
                    object_id=obj_id,
                    x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    box_history=[[det.x1, det.y1, det.x2, det.y2]],
                )
                result.append(Detection(
                    x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                    class_name=det.class_name, confidence=det.confidence,
                    object_id=obj_id,
                    observations=1,
                ))

        # ── Expire unmatched tracks or emit carry-forwards ────────────────────
        for tid in existing_ids - matched_track:
            track = tracks[tid]
            track.missed_frames += 1
            if track.missed_frames > self.keep_grace:
                del tracks[tid]
            else:
                # Still within grace window — output uses smoothed box (jitter-free
                # rendering); raw track.x1..y2 is kept intact for future IoU matching.
                sx1, sy1, sx2, sy2 = _smoothed_box(track.box_history)
                result.append(Detection(
                    x1=sx1, y1=sy1, x2=sx2, y2=sy2,
                    class_name=track.class_name, confidence=track.confidence,
                    object_id=track.object_id,
                    observations=track.observations,
                    is_carried=True,
                ))

        return result, _encode_state(tracks, next_id)
