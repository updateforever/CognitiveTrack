"""CognitiveTrack 的 ms-swift 数据与训练辅助接口。"""

from .model_family import (
    detect_qwen_model_family,
    detect_training_view_family,
    validate_model_dataset_family,
)
from .qwen_grounding import (
    QWEN_FAMILY_COORDINATE_DESCRIPTIONS,
    QWEN_FAMILY_PROTOCOLS,
    QWEN_MODEL_FAMILIES,
    QwenGroundingExportReport,
    export_qwen_grounding_records,
    qwen_bbox_protocol,
    to_qwen_grounding_record,
    validate_qwen_model_family,
)
from .release import DatasetReleaseReport, ReleaseShard, package_dataset_release
from .swift_dataset import (
    ValidationIssue,
    ValidationReport,
    read_jsonl,
    split_records_by_sequence,
    to_grpo_record,
    to_ms_swift_record,
    validate_ms_swift_record,
    validate_records,
    write_jsonl,
)
from .temporal_sampling import (
    SequenceCasePlan,
    TemporalCaseSamplingPlan,
    plan_temporal_presence_cases,
    sequence_sampling_key,
)
from .tracking_samples import (
    SOURCE_SCHEMA_VERSION,
    TrackingSampleBuildReport,
    TrackingSampleConfig,
    build_tracking_samples,
)

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "SOURCE_SCHEMA_VERSION",
    "TrackingSampleBuildReport",
    "TrackingSampleConfig",
    "build_tracking_samples",
    "SequenceCasePlan",
    "TemporalCaseSamplingPlan",
    "plan_temporal_presence_cases",
    "sequence_sampling_key",
    "read_jsonl",
    "split_records_by_sequence",
    "to_grpo_record",
    "to_ms_swift_record",
    "validate_ms_swift_record",
    "validate_records",
    "write_jsonl",
    "DatasetReleaseReport",
    "ReleaseShard",
    "package_dataset_release",
    "QWEN_FAMILY_COORDINATE_DESCRIPTIONS",
    "QWEN_FAMILY_PROTOCOLS",
    "QWEN_MODEL_FAMILIES",
    "QwenGroundingExportReport",
    "export_qwen_grounding_records",
    "qwen_bbox_protocol",
    "to_qwen_grounding_record",
    "validate_qwen_model_family",
    "detect_qwen_model_family",
    "detect_training_view_family",
    "validate_model_dataset_family",
]
