"""
yolo_pro/
─────────
YOLO-Pro: anchor-free multi-scale object detection for industrial edge devices.

Quick start:
    from yolo_pro import build_yolo_pro, TrainConfig, export_tflite_int8
"""

from .config         import YoloProConfig, CONFIGS, get_config
from .primitives     import silu, conv_bn, dw_conv
from .stem           import build_stem
from .csp_irb        import csp_irb
from .stage          import make_stage
from .backbone       import build_backbone, build_backbone_model
from .fpn_top_down   import fpn_fuse, build_top_down
from .pan_bottom_up  import pan_fuse, build_bottom_up
from .head           import detection_head
from .model          import build_yolo_pro
from .losses         import (
    varifocal_loss, ciou_loss, dfl_loss, yolo_pro_loss,
    WEIGHT_CLS, WEIGHT_BOX, WEIGHT_DFL, WEIGHT_O2O_PATH, BOX_WEIGHT_FLOOR,
)
from .train_config   import TrainConfig, ModelEMA
from .export         import (
    RepresentativeDataset,
    export_tflite_int8,
    export_tflite_float32,
    export_tflite_float32_decoded,
    verify_tflite_output,
)
from .decode         import (
    DecodeDetectionsLayer,
    build_yolo_pro_decoded,
    decode_raw_outputs_np,
    total_anchor_count,
)

__all__ = [
    "build_yolo_pro",
    "TrainConfig", "ModelEMA",
    "YoloProConfig", "CONFIGS", "get_config",
    "silu", "conv_bn", "dw_conv",
    "build_stem", "csp_irb", "make_stage",
    "build_backbone", "build_backbone_model",
    "fpn_fuse", "build_top_down",
    "pan_fuse", "build_bottom_up",
    "detection_head",
    "varifocal_loss", "ciou_loss", "dfl_loss", "yolo_pro_loss",
    "WEIGHT_CLS", "WEIGHT_BOX", "WEIGHT_DFL", "WEIGHT_O2O_PATH", "BOX_WEIGHT_FLOOR",
    "RepresentativeDataset",
    "export_tflite_int8",
    "export_tflite_float32",
    "export_tflite_float32_decoded",
    "verify_tflite_output",
    "DecodeDetectionsLayer",
    "build_yolo_pro_decoded",
    "decode_raw_outputs_np",
    "total_anchor_count",
]
