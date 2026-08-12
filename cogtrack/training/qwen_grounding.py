"""把 canonical 跟踪样本导出为 Qwen 官方 grounding 坐标范式。

Qwen2.5-VL 与 Qwen3-VL 的坐标协议不同：

* Qwen2.5-VL 使用 processor resize 后图像空间的绝对像素坐标；
* Qwen3-VL 使用与图像尺寸无关的 ``[0, 1000]`` 相对坐标。

本模块不自行复刻两套 resize。它把导出图片上的真实像素框放入 ms-swift
``objects.bbox``，并在消息中使用 ``<bbox>``。ms-swift 会根据实际模型模板在
图像预处理后完成官方转换，因而同一 canonical 数据可以安全派生两种训练视图。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from cogtrack.prompts import build_mosaic_prompt, build_pair_prompt
from cogtrack.protocol import (
    BBOX_PROTOCOL_NORM1000,
    BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    norm1000_xyxy_to_pixel_xywh,
    xywh_to_xyxy,
)

QWEN_MODEL_FAMILIES = ("qwen2_5_vl", "qwen3_vl")
QWEN_FAMILY_PROTOCOLS = {
    "qwen2_5_vl": BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    "qwen3_vl": BBOX_PROTOCOL_NORM1000,
}
QWEN_FAMILY_COORDINATE_DESCRIPTIONS = {
    "qwen2_5_vl": "processor-resized absolute pixel xyxy (official Qwen2.5-VL)",
    "qwen3_vl": "relative 0-to-1000 xyxy (official Qwen3-VL)",
}


@dataclass(frozen=True)
class QwenGroundingExportReport:
    """一批模型专属训练视图的统计。"""

    schema_version: str
    model_family: str
    bbox_protocol: str
    coordinate_description: str
    sample_count: int
    present_count: int
    absent_count: int
    bbox_placeholder_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_qwen_model_family(model_family: str) -> str:
    """拒绝模糊的 ``qwen`` 名称，强制调用者明确模型代际。"""

    if model_family not in QWEN_MODEL_FAMILIES:
        raise ValueError(
            f"model_family 必须是 {list(QWEN_MODEL_FAMILIES)} 之一，实际为 {model_family!r}"
        )
    return model_family


def qwen_bbox_protocol(model_family: str) -> str:
    return QWEN_FAMILY_PROTOCOLS[validate_qwen_model_family(model_family)]


@lru_cache(maxsize=131072)
def _image_size(path: Path) -> tuple[int, int]:
    # canonical pair/mosaic assets keep reference/current in one sequence directory and
    # resize every full frame with the same long-side rule.  Reading the JPEG header once
    # per sequence avoids reopening the same NAS files for thousands of repeated cases.
    if path.name.startswith(("reference_", "current_")):
        return _sequence_asset_size(path.parent)
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸非法：{path} -> {(width, height)}")
    return width, height


@lru_cache(maxsize=4096)
def _sequence_asset_size(directory: Path) -> tuple[int, int]:
    candidates = sorted(directory.glob("reference_*.jpg")) or sorted(directory.glob("current_*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"序列图片目录没有 reference/current 资产：{directory}")
    with Image.open(candidates[0]) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸非法：{candidates[0]} -> {(width, height)}")
    return width, height


def _real_xyxy_from_norm1000(
    bbox: Sequence[float],
    *,
    image_path: Path,
) -> list[float]:
    """将 canonical norm1000 框映射到实际导出图片，供 ms-swift 再处理。"""

    width, height = _image_size(image_path)
    xywh = norm1000_xyxy_to_pixel_xywh(bbox, width, height)
    return [round(value, 4) for value in xywh_to_xyxy(xywh)]


def _build_prompt(row: Mapping[str, Any], *, model_family: str):
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("canonical 样本缺少 metadata")
    effective_mode = str(metadata.get("effective_mode", "pair"))
    protocol = qwen_bbox_protocol(model_family)
    common = {
        "target_text": "",
        "semantic_memory": "",
        "reference_bbox": "<bbox>",
        "bbox_protocol": protocol,
        "include_memory_update": False,
    }
    if effective_mode == "pair":
        return build_pair_prompt(reference_has_box=False, **common)
    if effective_mode == "mosaic":
        history_ids = metadata.get("history_frame_ids")
        if not isinstance(history_ids, list) or not history_ids:
            raise ValueError("mosaic 样本缺少非空 history_frame_ids")
        return build_mosaic_prompt(len(history_ids), **common)
    raise ValueError(f"不支持的 effective_mode：{effective_mode!r}")


def _answer_text(*, status: str, bbox_protocol: str) -> str:
    bbox_key = (
        "bbox_norm1000_xyxy"
        if bbox_protocol == BBOX_PROTOCOL_NORM1000
        else "bbox_pixel_xyxy"
    )
    bbox_value = "<bbox>" if status == "present" else "null"
    return f'{{"target_status":"{status}","{bbox_key}":{bbox_value}}}'


def to_qwen_grounding_record(
    row: Mapping[str, Any],
    *,
    image_root: str | Path,
    model_family: str,
) -> dict[str, Any]:
    """将一条 v4 canonical 样本转换为模型族专属 ms-swift Grounding 样本。

    ``objects.bbox`` 始终保存导出 JPEG 上的真实绝对坐标。不要在这里预先执行
    Qwen smart-resize；这是 ms-swift 模型模板的职责，也是避免训练/推理漂移的
    关键边界。
    """

    family = validate_qwen_model_family(model_family)
    protocol = qwen_bbox_protocol(family)
    root = Path(image_root).expanduser().resolve()
    images_value = row.get("images")
    if not isinstance(images_value, list) or len(images_value) < 2:
        raise ValueError("跟踪 Grounding 样本至少需要初始化图和当前图")
    images = [str(value) for value in images_value]
    image_paths = [root / value for value in images]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")

    metadata_value = row.get("metadata")
    if not isinstance(metadata_value, Mapping):
        raise ValueError("canonical 样本缺少 metadata")
    metadata = dict(metadata_value)
    reference_norm = metadata.get("reference_bbox_norm1000_xyxy")
    if not isinstance(reference_norm, list):
        raise ValueError("canonical 样本缺少 reference_bbox_norm1000_xyxy")
    reference_real = _real_xyxy_from_norm1000(reference_norm, image_path=image_paths[0])

    status = str(row.get("target_status", ""))
    if status not in {"present", "absent"}:
        raise ValueError(f"target_status 非法：{status!r}")
    boxes = [reference_real]
    image_ids = [0]
    current_norm = row.get("bbox_norm1000_xyxy")
    if status == "present":
        if not isinstance(current_norm, list):
            raise ValueError("present 样本缺少 bbox_norm1000_xyxy")
        boxes.append(_real_xyxy_from_norm1000(current_norm, image_path=image_paths[-1]))
        image_ids.append(len(images) - 1)
    elif current_norm is not None:
        raise ValueError("absent 样本的 bbox_norm1000_xyxy 必须为 null")

    prompt = _build_prompt(row, model_family=family)
    if prompt.expected_image_count != len(images):
        raise ValueError(
            f"Prompt {prompt.name} 期望 {prompt.expected_image_count} 张图，实际为 {len(images)}"
        )
    assistant = _answer_text(status=status, bbox_protocol=protocol)
    metadata.update(
        qwen_model_family=family,
        bbox_protocol=protocol,
        coordinate_description=QWEN_FAMILY_COORDINATE_DESCRIPTIONS[family],
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        ms_swift_bbox_format="new",
        canonical_bbox_format="norm1000_xyxy",
    )
    record = {
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": "<image>" * len(images) + "\n" + prompt.user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "images": images,
        "objects": {
            "bbox": boxes,
            "bbox_type": "real",
            "image_id": image_ids,
        },
        "metadata": metadata,
        "target_status": status,
        # canonical 框只给审计和 GRPO reward 使用，不会进入模型消息。
        "bbox_norm1000_xyxy": current_norm,
        "bbox_format": "norm1000_xyxy",
    }
    if "id" in row:
        record["id"] = row["id"]
    return record


def export_qwen_grounding_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    image_root: str | Path,
    model_family: str,
) -> tuple[list[dict[str, Any]], QwenGroundingExportReport]:
    """批量导出并返回可写入 JSON 的统计报告。"""

    family = validate_qwen_model_family(model_family)
    records = [
        to_qwen_grounding_record(row, image_root=image_root, model_family=family)
        for row in rows
    ]
    present = sum(record["target_status"] == "present" for record in records)
    absent = len(records) - present
    placeholder_count = sum(
        message["content"].count("<bbox>")
        for record in records
        for message in record["messages"]
    )
    report = QwenGroundingExportReport(
        schema_version="cogtrack.qwen_grounding_export.v1",
        model_family=family,
        bbox_protocol=qwen_bbox_protocol(family),
        coordinate_description=QWEN_FAMILY_COORDINATE_DESCRIPTIONS[family],
        sample_count=len(records),
        present_count=present,
        absent_count=absent,
        bbox_placeholder_count=placeholder_count,
    )
    return records, report


def answer_with_materialized_bbox(answer: str, bbox: Sequence[int]) -> dict[str, Any]:
    """测试/审计辅助：将单个 assistant ``<bbox>`` 替换为实际坐标后解析。"""

    text = answer.replace("<bbox>", json.dumps(list(bbox), separators=(",", ":")))
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("materialized assistant 不是 JSON 对象")
    return payload


__all__ = [
    "QWEN_FAMILY_COORDINATE_DESCRIPTIONS",
    "QWEN_FAMILY_PROTOCOLS",
    "QWEN_MODEL_FAMILIES",
    "QwenGroundingExportReport",
    "answer_with_materialized_bbox",
    "export_qwen_grounding_records",
    "qwen_bbox_protocol",
    "to_qwen_grounding_record",
    "validate_qwen_model_family",
]
