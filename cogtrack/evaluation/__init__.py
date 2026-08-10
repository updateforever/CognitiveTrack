"""CognitiveTrack 统一评测接口。"""

from .evaluator import (
    evaluate_frames,
    evaluate_jsonl_files,
    summarize_pytracking_curves,
    write_evaluation_outputs,
)
from .io import (
    CanonicalFrame,
    discover_jsonl_files,
    is_debug_limited,
    load_frame_records,
    partition_debug_limited,
)
from .metrics import (
    aggregate_benchmark_standard,
    bbox_iou,
    evaluate_benchmark_sequence,
    evaluate_cognitive_visible_only,
)
from .pytracking_metrics import (
    calc_err_center,
    calc_iou_overlap,
    calc_seq_err_robust,
    extract_results_from_canonical_frames,
)

__all__ = [
    "CanonicalFrame",
    "aggregate_benchmark_standard",
    "bbox_iou",
    "calc_err_center",
    "calc_iou_overlap",
    "calc_seq_err_robust",
    "discover_jsonl_files",
    "evaluate_benchmark_sequence",
    "evaluate_cognitive_visible_only",
    "evaluate_frames",
    "evaluate_jsonl_files",
    "extract_results_from_canonical_frames",
    "is_debug_limited",
    "load_frame_records",
    "partition_debug_limited",
    "summarize_pytracking_curves",
    "write_evaluation_outputs",
]
