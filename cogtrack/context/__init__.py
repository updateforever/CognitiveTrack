"""VLM 跟踪所需的视觉上下文构造。"""

from .builder import ContextBuildResult, TrackingContextBuilder
from .visual import (
    DEFAULT_VISUAL_MARKER_STYLE,
    REFERENCE_MODE_BBOX_TEXT,
    REFERENCE_MODE_VISUAL_BOX,
    REFERENCE_MODES,
    VISUAL_MARKER_VERSION,
    VisualMarkerStyle,
    build_history_mosaic,
    draw_reference_box,
    validate_reference_mode,
)

__all__ = [
    "ContextBuildResult",
    "DEFAULT_VISUAL_MARKER_STYLE",
    "REFERENCE_MODE_BBOX_TEXT",
    "REFERENCE_MODE_VISUAL_BOX",
    "REFERENCE_MODES",
    "TrackingContextBuilder",
    "VISUAL_MARKER_VERSION",
    "VisualMarkerStyle",
    "build_history_mosaic",
    "draw_reference_box",
    "validate_reference_mode",
]
