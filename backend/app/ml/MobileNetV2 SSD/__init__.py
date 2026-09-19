"""
MobileNetV2 SSD FPN-Lite 320×320 — package shim.

The modules in this folder use bare-name imports (e.g. ``from primitives import CFG``)
because they were written as standalone scripts, not as a Python package.
This __init__.py injects the folder's own directory onto sys.path so that
those bare imports resolve correctly when the folder is imported as a package
from anywhere in the project.

Nothing in the SSD source files is modified.  All integration logic lives in
app/ml/mobilenetv2_ssd_worker.py.

Public API (mirrors the yolo_pro package surface):
    from app.ml.mobilenetv2_ssd_worker import (
        is_mobilenetv2_ssd,
        run_mobilenetv2_ssd_training,
    )
"""

import sys
import os as _os

# Insert this folder at the *front* of sys.path so bare-name imports inside
# the SSD modules (primitives, backbone, fpn_lite, …) resolve to the files
# here and not to any identically-named top-level module that might exist.
# This folder lives at:  backend/app/ml/MobileNetV2 SSD/__init__.py
# _SSD_DIR resolves to:  .../backend/app/ml/MobileNetV2 SSD/
_SSD_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SSD_DIR not in sys.path:
    sys.path.insert(0, _SSD_DIR)

# ---------------------------------------------------------------------------
# Re-export the public build / export surface so callers can use:
#   from app.ml.mobilenetv2_ssd import build_model, export_tflite
# ---------------------------------------------------------------------------
from model import build_model                                   # noqa: E402
from export import export_tflite, validate_tflite               # noqa: E402
from anchor_generator import generate_anchors                   # noqa: E402
from postprocess import postprocess                             # noqa: E402
from losses import ssd_loss                                     # noqa: E402

__all__ = [
    "build_model",
    "export_tflite",
    "validate_tflite",
    "generate_anchors",
    "postprocess",
    "ssd_loss",
]