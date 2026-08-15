from cogtrack.context import REFERENCE_MODE_BBOX_TEXT, REFERENCE_MODE_VISUAL_BOX
from cogtrack.training import (
    MEMORY_SUPERVISION_DISABLED,
    MEMORY_SUPERVISION_EXPLICIT,
    MEMORY_SUPERVISION_MASKED_NULL,
)
from tracking.synthesize_stage1_dataset import _parser


def test_legacy_and_visual_synthesis_profiles_have_safe_defaults() -> None:
    legacy = _parser("legacy_stage1").parse_args(["--output-dir", "legacy"])
    visual = _parser("visual_v5").parse_args(["--output-dir", "visual"])
    vlt = _parser("vlt_v6").parse_args(["--output-dir", "vlt"])

    assert legacy.context_mode == "pair"
    assert legacy.reference_mode == REFERENCE_MODE_BBOX_TEXT
    assert legacy.memory_supervision == MEMORY_SUPERVISION_DISABLED

    assert visual.context_mode == "both"
    assert visual.reference_mode == REFERENCE_MODE_VISUAL_BOX
    assert visual.memory_supervision == MEMORY_SUPERVISION_EXPLICIT
    assert visual.history_size == 4
    assert visual.qwen_model_families == ["qwen3_vl"]

    assert vlt.context_mode == "mosaic"
    assert vlt.reference_mode == REFERENCE_MODE_VISUAL_BOX
    assert vlt.memory_supervision == MEMORY_SUPERVISION_MASKED_NULL
    assert vlt.history_size == 3
    assert vlt.qwen_model_families == ["qwen3_vl"]
