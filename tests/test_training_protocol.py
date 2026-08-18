import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cogtrack.training.grpo_rewards import (
    CognitiveConsistencyReward,
    CognitiveFormatReward,
)
from cogtrack.training.swift_dataset import (
    TRACKING_OUTPUT_KEYS,
    parse_training_tracking_answer,
    validate_ms_swift_record,
)


def _payload() -> dict:
    return {
        "status": "present",
        "bbox_2d": [100, 120, 400, 520],
    }


def _record(payload: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>Track the initialized target."},
            {"role": "assistant", "content": json.dumps(payload)},
        ],
        "images": ["images/current.jpg"],
        "metadata": {"dataset": "unit", "sequence": "seq-1"},
    }


def _protocol_errors(payload: dict) -> list:
    return [
        issue
        for issue in validate_ms_swift_record(_record(payload), check_images=False)
        if issue.code == "answer.protocol"
    ]


def test_presence_only_answer_requires_exact_two_fields_and_valid_norm1000_bbox():
    valid = _payload()
    assert set(parse_training_tracking_answer(valid)) == TRACKING_OUTPUT_KEYS
    assert not _protocol_errors(valid)

    with_extra = {**valid, "bbox_format": "norm1000_xyxy"}
    assert _protocol_errors(with_extra)

    missing = deepcopy(valid)
    missing.pop("bbox_2d")
    assert _protocol_errors(missing)

    legacy_confidence = {**valid, "presence_confidence": 0.9}
    assert _protocol_errors(legacy_confidence)

    outside_norm1000 = deepcopy(valid)
    outside_norm1000["bbox_2d"] = [-1, 100, 400, 500]
    assert _protocol_errors(outside_norm1000)


def test_memory_labeled_answer_accepts_the_versioned_three_field_protocol():
    with_null = {**_payload(), "memory_update": None}
    with_delta = {
        **_payload(),
        "memory_update": "Rear view reveals two stable white stripes.",
    }
    assert set(parse_training_tracking_answer(with_null)) == {
        *TRACKING_OUTPUT_KEYS,
        "memory_update",
    }
    assert not _protocol_errors(with_null)
    assert not _protocol_errors(with_delta)
    assert CognitiveFormatReward()([json.dumps(with_delta)]) == [1.0]
    assert CognitiveConsistencyReward()([json.dumps(with_delta)]) == [1.0]

    absent_with_memory = {
        "status": "absent",
        "bbox_2d": None,
        "memory_update": "The target changed appearance.",
    }
    assert not _protocol_errors(absent_with_memory)


@pytest.mark.parametrize(
    ("status", "bbox"),
    [
        ("absent", None),
        ("present", [100, 120, 400, 520]),
    ],
)
def test_training_accepts_the_same_binary_states_as_inference(status: str, bbox: list[int] | None):
    payload = _payload()
    payload.update(
        status=status,
        bbox_2d=bbox,
    )
    assert not _protocol_errors(payload)
    text = json.dumps(payload)
    assert CognitiveFormatReward()([text]) == [1.0]
    assert CognitiveConsistencyReward()([text]) == [1.0]


@pytest.mark.parametrize(
    ("status", "bbox"),
    [
        ("absent", [100, 120, 400, 520]),
        ("present", None),
        ("uncertain", None),
    ],
)
def test_sft_and_grpo_reject_the_same_inconsistent_states(
    status: str,
    bbox: list[int] | None,
):
    payload = _payload()
    payload.update(
        status=status,
        bbox_2d=bbox,
    )
    assert _protocol_errors(payload)
    text = json.dumps(payload)
    assert CognitiveFormatReward()([text]) == [0.0]
    assert CognitiveConsistencyReward()([text]) == [0.0]


def test_training_yaml_matches_launchers_and_contains_no_absolute_path():
    root = Path(__file__).resolve().parents[1]
    sft = yaml.safe_load((root / "configs/training/sft.yaml").read_text(encoding="utf-8"))
    grpo = yaml.safe_load((root / "configs/training/grpo.yaml").read_text(encoding="utf-8"))

    assert sft["tuner_type"] == "lora"
    assert sft["num_train_epochs"] == 3
    assert sft["output_dir"] == "outputs/sft_qwen_vl"
    assert not Path(sft["model"]).is_absolute()

    assert grpo["rlhf_type"] == "grpo"
    assert grpo["gradient_accumulation_steps"] == 8
    assert grpo["max_completion_length"] == 256
    assert grpo["reward_funcs"] == [
        "cogtrack_format",
        "cogtrack_presence",
        "cogtrack_bbox",
        "cogtrack_consistency",
    ]
    assert grpo["reward_weights"] == [0.5, 1.5, 2.0, 0.5]
