import json
from pathlib import Path

import pytest

from cogtrack.training import (
    detect_qwen_model_family,
    detect_training_view_family,
    validate_model_dataset_family,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_detects_official_huggingface_model_types(tmp_path: Path) -> None:
    qwen25 = tmp_path / "model-a"
    qwen3 = tmp_path / "model-b"
    _write_json(qwen25 / "config.json", {"model_type": "qwen2_5_vl"})
    _write_json(qwen3 / "config.json", {"model_type": "qwen3_vl"})

    assert detect_qwen_model_family(qwen25) == "qwen2_5_vl"
    assert detect_qwen_model_family(qwen3) == "qwen3_vl"


def test_rejects_cross_family_training_view(tmp_path: Path) -> None:
    model = tmp_path / "Qwen2.5-VL-7B"
    data = tmp_path / "qwen3_vl" / "train.jsonl"
    _write_json(model / "config.json", {"model_type": "qwen2_5_vl"})
    _write_json(data.parent / "dataset_info.json", {"model_family": "qwen3_vl"})
    _write_json(data, {"metadata": {"qwen_model_family": "qwen3_vl"}})

    assert detect_training_view_family(data) == "qwen3_vl"
    with pytest.raises(ValueError, match="模型/数据族不匹配"):
        validate_model_dataset_family(model, data)


def test_accepts_matching_family_from_record_metadata(tmp_path: Path) -> None:
    model = tmp_path / "Qwen3-VL-32B"
    data = tmp_path / "train.jsonl"
    _write_json(model / "config.json", {"model_type": "qwen3_vl"})
    _write_json(data, {"metadata": {"qwen_model_family": "qwen3_vl"}})

    assert validate_model_dataset_family(model, data) == "qwen3_vl"
