"""
Video post-processing service — Phase 6.

Reads frames from an input video stored in S3, runs per-frame inference
(when an inference callable is supplied) or uses pre-computed detections
(legacy / debug path), applies the Phase 3 detection pipeline per frame
(threshold filter → class filter → NMS → tracking), renders surviving
detections as annotated bounding boxes, writes the output video back to S3,
and returns summary metadata.

Design constraints
------------------
- The model-loading concern lives in the worker; this module only receives a
  callable inference_fn(frame_bgr) → List[Detection].
- Reads/writes via storage.download_bytes / storage.upload_bytes so the
  worker never touches the host filesystem permanently.
- cv2 is required (opencv-python-headless).  A clear ImportError is raised
  at call-time if the package is missing rather than at module import time,
  so the rest of the app can still load.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from app.core.storage import storage
from app.services.post_processing.detection_postprocess import PipelineConfig, run_pipeline
from app.services.post_processing.types import Detection

logger = logging.getLogger(__name__)


# ─── Cancellation ─────────────────────────────────────────────────────────────

class ProcessingCancelledError(Exception):
    """Raised by VideoProcessor.process() when cancel_check() reports True.

    Kept distinct from a generic failure so the worker can leave the job's
    status as 'cancelled' (set by the cancel endpoint) instead of overwriting
    it with 'failed'.
    """


# ─── Summary ──────────────────────────────────────────────────────────────────

@dataclass
class ProcessingSummary:
    frames_processed: int = 0
    frames_with_detections: int = 0   # frames where ≥1 detection survived pipeline
    total_input_detections: int = 0
    total_final_detections: int = 0
    processing_ms: float = 0.0


# ─── Rendering helpers ────────────────────────────────────────────────────────

# Colour palette for up to 20 object IDs (BGR)
_PALETTE = [
    (  0, 200,   0), (200,   0,   0), (  0,   0, 200), (200, 200,   0),
    (  0, 200, 200), (200,   0, 200), (255, 128,   0), (128,   0, 255),
    (  0, 128, 255), (255,   0, 128), ( 64, 200,  64), (200,  64,  64),
    ( 64,  64, 200), (200, 200,  64), ( 64, 200, 200), (200,  64, 200),
    (255, 200,   0), (  0, 255, 200), (200,   0, 255), (128, 128, 128),
]

def _colour_for(object_id: Optional[int]) -> tuple:
    if object_id is None:
        return (0, 220, 0)
    return _PALETTE[object_id % len(_PALETTE)]


_LABEL_BG = (30, 30, 30)   # near-black background for all labels
_LABEL_PAD = 10            # pixels of padding inside label background on every side


def _render_detections(frame, detections: List[Detection], h: int, w: int):
    """Draw bounding boxes + labels onto a frame in-place. Returns the frame."""
    import cv2  # local import — fails clearly if package missing

    # Carry-forward tracker outputs keep state continuity but tend to read as
    # duplicate predictions in the rendered video, so only live detections are drawn.
    detections = [det for det in detections if not getattr(det, "is_carried", False)]
    if not detections:
        return frame

    # Font scale relative to frame height — clearly readable after browser downscaling.
    # h/420 gives 1.71 at 720p and 2.57 at 1080p; floor of 1.2 keeps small frames legible.
    font_scale = max(1.2, h / 420.0)
    font_face  = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = 2

    # Box thickness scales with resolution — no hairlines at any size.
    # h//240: 480p→2, 720p→3, 1080p→4.  Floor of 3 ensures visible strokes.
    live_thick = max(3, h // 240)

    for det in detections:
        x1 = int(det.x1 * w)
        y1 = int(det.y1 * h)
        x2 = int(det.x2 * w)
        y2 = int(det.y2 * h)

        colour = _colour_for(det.object_id)
        carried = getattr(det, "is_carried", False)

        # Grace-period tracks: one stroke thinner to hint they are carry-forwards
        box_thickness = max(1, live_thick - 1) if carried else live_thick
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, box_thickness)

        # Label text: "class_name #id confidence%" or "class_name confidence%"
        conf_str = f"{det.confidence * 100:.0f}%"
        if det.object_id is not None:
            label = f"{det.class_name} #{det.object_id} {conf_str}"
            if carried:
                label += " ~"
        else:
            label = f"{det.class_name} {conf_str}"

        (tw, th), baseline = cv2.getTextSize(label, font_face, font_scale, font_thick)

        bg_w = tw + 2 * _LABEL_PAD
        bg_h = th + baseline + 2 * _LABEL_PAD

        # Prefer label above the box (2 px gap).  When there is not enough room
        # above, place the label 2 px inside the box top — not flush against the
        # frame edge — so it reads as part of the annotation, not a screen border.
        if y1 - bg_h - 2 >= 0:
            bg_top = y1 - bg_h - 2
        else:
            bg_top = max(0, min(y1 + 2, h - bg_h))

        # Clamp left edge so label stays within frame width
        bg_left = max(0, min(x1, w - bg_w))

        bg_right  = min(bg_left + bg_w, w)
        bg_bottom = min(bg_top  + bg_h, h)

        # Dark background, then white text
        cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), _LABEL_BG, -1)
        cv2.putText(
            frame, label,
            (bg_left + _LABEL_PAD, bg_top + _LABEL_PAD + th),
            font_face, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA,
        )

    return frame


# ─── H.264 re-encoder ────────────────────────────────────────────────────────

def _reencode_h264(raw_path: str, out_path: str) -> str:
    """
    Re-encode *raw_path* (OpenCV mp4v) to H.264 MP4 at *out_path*.

    Requires ffmpeg on PATH.  Raises RuntimeError on any failure so the
    caller never silently uploads a non-playable file.  Returns *out_path*.
    """
    raw_size = os.path.getsize(raw_path)
    if raw_size == 0:
        raise RuntimeError("Raw OpenCV output is empty; cannot re-encode to H.264")

    logger.info("VideoProcessor: starting H.264 re-encode (raw size %d bytes)", raw_size)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",      # broadest browser/device support
                "-movflags", "+faststart",   # moov atom at front for streaming
                "-an",                       # no audio track (silent output acceptable)
                out_path,
            ],
            capture_output=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH — cannot re-encode output "
            "to browser-compatible H.264. Install ffmpeg on the worker and retry."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg re-encode timed out after 300 s")

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(
            f"ffmpeg re-encode failed (exit {result.returncode}): {stderr}"
        )

    out_size = os.path.getsize(out_path)
    if out_size == 0:
        raise RuntimeError(
            "ffmpeg produced an empty output file — H.264 re-encode may have silently failed"
        )

    logger.info(
        "VideoProcessor: H.264 re-encode complete — %d bytes -> %d bytes",
        raw_size, out_size,
    )
    return out_path


# ─── Processor ────────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Static-method interface to keep the worker thin.

    process() is the single entry point:
      1. Download input video from S3 to a temp file.
      2. Open with cv2.VideoCapture.
      3. For each frame obtain raw detections via inference_fn (preferred) or
         from the pre-computed detections_per_frame list (legacy/debug path).
      4. Run the Phase 3 post-processing pipeline.
      5. Render final_detections onto the frame.
      6. Encode the output video and upload to S3.
      7. Return a ProcessingSummary.
    """

    @staticmethod
    def process(
        input_key: str,
        output_key: str,
        config: PipelineConfig,
        detections_per_frame: Optional[List[List[Detection]]] = None,
        tracker_state: Optional[dict] = None,
        inference_fn: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessingSummary:
        """
        Parameters
        ----------
        inference_fn:
            Callable that takes a BGR numpy frame and returns List[Detection].
            When provided this is the primary detection source and
            detections_per_frame is ignored.
        detections_per_frame:
            Legacy/debug fallback: one list of Detection objects per frame.
            Used when inference_fn is None.
        cancel_check:
            Optional no-arg callable polled once per frame; raises
            ProcessingCancelledError as soon as it returns True. Lets the
            caller stop a long-running render early (e.g. user clicked
            Cancel) without waiting for the whole video to finish.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python-headless is required for video processing. "
                "Add it to requirements.txt."
            ) from exc

        summary = ProcessingSummary()
        t_start = time.monotonic()
        detections_per_frame = detections_per_frame or []
        current_tracker_state = tracker_state or {}

        with tempfile.TemporaryDirectory() as tmpdir:
            # ── 1. Download input ──────────────────────────────────────────────
            input_bytes = storage.download_bytes(input_key)
            input_path = os.path.join(tmpdir, "input.mp4")
            with open(input_path, "wb") as fh:
                fh.write(input_bytes)

            # ── 2. Open video ──────────────────────────────────────────────────
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video from key {input_key!r}")

            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ── 3. Set up writer ───────────────────────────────────────────────
            output_path = os.path.join(tmpdir, "output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_idx = 0
            try:
                while True:
                    if cancel_check is not None and cancel_check():
                        raise ProcessingCancelledError("Cancelled by user")

                    ok, frame = cap.read()
                    if not ok:
                        break

                    # ── 4. Obtain raw detections ───────────────────────────────
                    if inference_fn is not None:
                        raw_dets: List[Detection] = inference_fn(frame)
                    else:
                        raw_dets = (
                            detections_per_frame[frame_idx]
                            if frame_idx < len(detections_per_frame)
                            else []
                        )
                    summary.total_input_detections += len(raw_dets)

                    # ── 5. Run pipeline ────────────────────────────────────────
                    result = run_pipeline(raw_dets, config, tracker_state=current_tracker_state)
                    current_tracker_state = result.tracker_state
                    summary.total_final_detections += len(result.final_detections)
                    if result.final_detections:
                        summary.frames_with_detections += 1

                    # ── 6. Render and write ────────────────────────────────────
                    annotated = _render_detections(frame, result.final_detections, height, width)
                    writer.write(annotated)

                    frame_idx += 1
            finally:
                cap.release()
                writer.release()

            summary.frames_processed = frame_idx

            # ── 7. Log raw output ─────────────────────────────────────────────
            raw_size = os.path.getsize(output_path)
            logger.info(
                "VideoProcessor: raw OpenCV output written — %d bytes", raw_size
            )

            # ── 8. Re-encode to H.264 for browser playback ────────────────────
            reencoded_path = os.path.join(tmpdir, "output_h264.mp4")
            _reencode_h264(output_path, reencoded_path)

            # ── 9. Read re-encoded file and log upload details ─────────────────
            final_size = os.path.getsize(reencoded_path)
            logger.info(
                "VideoProcessor: uploading final output — key=%s  size=%d bytes",
                output_key, final_size,
            )
            with open(reencoded_path, "rb") as fh:
                output_bytes = fh.read()

        storage.upload_bytes(output_bytes, output_key, content_type="video/mp4")
        summary.processing_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "VideoProcessor: processed %d frames in %.0f ms "
            "(%d input → %d final detections)",
            summary.frames_processed,
            summary.processing_ms,
            summary.total_input_detections,
            summary.total_final_detections,
        )
        return summary
