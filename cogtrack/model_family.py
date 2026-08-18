"""Lightweight Qwen model/data-family preflight without training imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

QWEN_MODEL_FAMILIES = ("qwen2_5_vl", "qwen3_vl")


def validate_qwen_model_family(value: str) -> str:
    if value not in QWEN_MODEL_FAMILIES:
        raise ValueError(f"不支持的 Qwen 模型族：{value!r}")
    return value


def _family_from_text(value: str) -> Optional[str]:
    normalized = value.lower().replace("-", "_").replace(".", "_")
    if "qwen2_5_vl" in normalized or "qwen25vl" in normalized:
        return "qwen2_5_vl"
    if "qwen3_vl" in normalized or "qwen3vl" in normalized:
        return "qwen3_vl"
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return payload


def detect_qwen_model_family(model: str | Path) -> str:
    """Detect the Qwen-VL generation from local config or path text."""

    model_path = Path(model).expanduser()
    candidates: list[str] = [str(model)]
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if config_path.is_file():
            config = _read_json_object(config_path)
            candidates.extend(
                [
                    str(config.get("model_type", "")),
                    json.dumps(config.get("architectures", []), ensure_ascii=False),
                ]
            )
        adapter_path = model_path / "adapter_config.json"
        if adapter_path.is_file():
            adapter = _read_json_object(adapter_path)
            candidates.append(str(adapter.get("base_model_name_or_path", "")))

    families = {family for value in candidates if (family := _family_from_text(value))}
    if len(families) != 1:
        raise ValueError(
            f"无法唯一识别模型族：{model}；应能识别为 {list(QWEN_MODEL_FAMILIES)} 之一"
        )
    return families.pop()


def detect_training_view_family(dataset: str | Path) -> str:
    """Detect the training-view family from dataset_info or the first row."""

    dataset_path = Path(dataset).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"训练 JSONL 不存在：{dataset_path}")
    info_path = dataset_path.parent / "dataset_info.json"
    family: Any = None
    if info_path.is_file():
        family = _read_json_object(info_path).get("model_family")
    if family is None:
        with dataset_path.open("r", encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), "")
        if not first_line:
            raise ValueError(f"训练 JSONL 为空：{dataset_path}")
        row = json.loads(first_line)
        if not isinstance(row, Mapping):
            raise ValueError(f"训练 JSONL 首条样本必须是对象：{dataset_path}")
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            family = metadata.get("qwen_model_family")
    if not isinstance(family, str):
        raise ValueError(
            f"训练视图缺少 qwen model_family 元数据：{dataset_path}；禁止把旧通用 JSONL 用于训练"
        )
    return validate_qwen_model_family(family)


def validate_model_dataset_family(
    model: str | Path,
    dataset: str | Path,
    *,
    expected_family: Optional[str] = None,
) -> str:
    """Return the matching family and reject any model/data mismatch."""

    detected_model = detect_qwen_model_family(model)
    detected_data = detect_training_view_family(dataset)
    if expected_family is not None:
        expected = validate_qwen_model_family(expected_family)
        if detected_model != expected:
            raise ValueError(f"显式模型族 {expected} 与模型配置识别结果 {detected_model} 不一致")
    if detected_model != detected_data:
        raise ValueError(
            f"模型/数据族不匹配：model={detected_model}, dataset={detected_data}。"
            "Qwen2.5-VL 与 Qwen3-VL 坐标协议不同，禁止交叉使用训练视图。"
        )
    return detected_model


__all__ = [
    "detect_qwen_model_family",
    "detect_training_view_family",
    "validate_model_dataset_family",
    "validate_qwen_model_family",
]
