import json
from pathlib import Path

import pytest

from cogtrack.training.loss_mask import split_tracking_core_response
from tracking.validate_sft_supervision import validate_dataset


def test_tracking_core_masks_only_memory_value() -> None:
    response = (
        '{"target_status":"present","bbox_norm1000_xyxy":<bbox>,'
        '"memory_update":null}'
    )

    parts, weights = split_tracking_core_response(
        response,
        require_memory_field=True,
    )

    assert "".join(parts) == response
    assert weights == [1.0, 0.0, 1.0]
    assert parts[0].endswith('"memory_update":')
    assert parts[1] == "null"
    assert parts[2] == "}"
    assert "<bbox>" in parts[0]


def test_tracking_core_rejects_noncanonical_memory_boundary() -> None:
    response = (
        '{"target_status":"absent","bbox_norm1000_xyxy":null,'
        '"memory_update" : null}'
    )
    with pytest.raises(ValueError, match="canonical memory_update"):
        split_tracking_core_response(response, require_memory_field=True)


def test_tracking_core_rejects_extra_field_after_memory() -> None:
    response = (
        '{"target_status":"absent","bbox_norm1000_xyxy":null,'
        '"memory_update":null,"extra":1}'
    )
    with pytest.raises(ValueError, match="最后一个"):
        split_tracking_core_response(response, require_memory_field=True)


def test_two_field_history_remains_full_loss_in_non_strict_mode() -> None:
    response = '{"target_status":"absent","bbox_norm1000_xyxy":null}'
    assert split_tracking_core_response(response) == ([response], [1.0])


def test_supervision_preflight_accepts_masked_null_and_rejects_wrong_profile(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "train.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": "track"},
            {"role": "user", "content": "<image><image><image>"},
            {
                "role": "assistant",
                "content": (
                    '{"target_status":"absent","bbox_norm1000_xyxy":null,'
                    '"memory_update":null}'
                ),
            },
        ],
        "images": ["a.jpg", "b.jpg", "c.jpg"],
        "metadata": {
            "prompt_profile": "vlt_v6",
            "memory_supervision": "masked_null",
            "sft_supervision_profile": "tracking_core",
            "memory_loss_masked": True,
        },
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert validate_dataset(dataset, profile="tracking_core") == 1
    with pytest.raises(ValueError, match="训练请求"):
        validate_dataset(dataset, profile="full")

    row["metadata"].update(
        sft_supervision_profile="full",
        memory_loss_masked=False,
        memory_supervision="feasibility_null",
    )
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="禁止用于 SFT"):
        validate_dataset(dataset, profile="full")
