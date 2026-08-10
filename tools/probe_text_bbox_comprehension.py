#!/usr/bin/env python3
"""用可控合成图检查 VLM 是否真正理解 Prompt 中的初始化框坐标。

该探针刻意把两个问题拆开：

1. 单图区域指认只检查“文本坐标 -> 图像区域 -> 目标身份”；
2. 双图跟踪再检查“初始化坐标确定身份 -> 当前帧重新定位”。

四个目标同时出现在图中，所有测试仅改变 Prompt 中的坐标。若模型忽略坐标，
就不可能在四个查询上稳定返回四个不同且正确的目标。合成图只用于能力诊断，
不会混入训练数据。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.vlm import (  # noqa: E402
    GenerationConfig,
    HuggingFaceQwenVLBackend,
    QwenVLConfig,
)

CANVAS_SIZE = 1000
TARGETS = ("red_circle", "blue_square", "green_triangle", "yellow_star")

REFERENCE_BOXES: Dict[str, list[int]] = {
    "red_circle": [100, 100, 350, 350],
    "blue_square": [650, 100, 900, 350],
    "green_triangle": [100, 650, 350, 900],
    "yellow_star": [650, 650, 900, 900],
}

CURRENT_BOXES: Dict[str, list[int]] = {
    "yellow_star": [100, 100, 350, 350],
    "green_triangle": [650, 100, 900, 350],
    "blue_square": [100, 650, 350, 900],
    "red_circle": [650, 650, 900, 900],
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs/models/qwen25vl_7b.yaml"),
        help="Qwen-VL 模型配置 YAML。",
    )
    parser.add_argument(
        "--env-config",
        default=str(PROJECT_ROOT / "configs/env.local.yaml"),
        help="用于解析 model_root 的本机环境配置。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs/coordinate_comprehension_probe"),
        help="保存合成图和完整报告的目录。",
    )
    return parser.parse_args()


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"配置顶层必须是 mapping: {resolved}")
    return dict(payload)


def _resolve_model_config(model_config: str | Path, env_config: str | Path) -> QwenVLConfig:
    payload = _load_yaml(model_config)
    environment = _load_yaml(env_config)
    model_path = Path(str(payload["model_path"])).expanduser()
    if not model_path.is_absolute():
        model_root = environment.get("model_root")
        if not model_root:
            raise ValueError("相对 model_path 需要 env-config 提供 model_root")
        model_path = Path(str(model_root)).expanduser() / model_path
    payload["model_path"] = str(model_path.resolve())
    # 探针输出很短，缩短生成长度可显著降低重复测试耗时。
    payload["max_new_tokens"] = 96
    return QwenVLConfig.from_mapping(payload)


def _star_points(box: Sequence[int]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    outer = min(x2 - x1, y2 - y1) * 0.48
    inner = outer * 0.43
    points = []
    for index in range(10):
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        radius = outer if index % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _draw_target(draw: ImageDraw.ImageDraw, target_id: str, box: Sequence[int]) -> None:
    x1, y1, x2, y2 = box
    if target_id == "red_circle":
        draw.ellipse((x1, y1, x2, y2), fill=(220, 35, 45), outline=(80, 0, 0), width=8)
    elif target_id == "blue_square":
        draw.rectangle((x1, y1, x2, y2), fill=(35, 85, 220), outline=(0, 20, 90), width=8)
    elif target_id == "green_triangle":
        draw.polygon(
            ((x1 + x2) / 2.0, y1, x2, y2, x1, y2),
            fill=(30, 175, 75),
            outline=(0, 70, 20),
            width=8,
        )
    elif target_id == "yellow_star":
        draw.polygon(_star_points(box), fill=(245, 195, 20), outline=(95, 70, 0), width=8)
    else:  # pragma: no cover - 常量表错误才会进入
        raise KeyError(target_id)


def _make_scene(boxes: Mapping[str, Sequence[int]]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    # 淡网格只帮助观察坐标方向，不直接标出目标名称或查询框。
    draw.line((500, 0, 500, 1000), fill=(190, 190, 190), width=3)
    draw.line((0, 500, 1000, 500), fill=(190, 190, 190), width=3)
    for target_id, box in boxes.items():
        _draw_target(draw, target_id, box)
    return image


def _extract_json(text: str) -> Dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(
        0.0, by2 - by1
    ) - intersection
    return intersection / union if union > 0.0 else 0.0


def _is_valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def _region_prompt(box: Sequence[int], protocol: str, image_size: Sequence[int]) -> str:
    if protocol == "norm1000":
        coordinate_text = (
            "The image coordinate system is normalized from 0 to 1000 on both axes, "
            "with (0,0) at the top-left and (1000,1000) at the bottom-right."
        )
    else:
        coordinate_text = (
            f"The image shown to you is {image_size[0]} pixels wide and {image_size[1]} pixels high. "
            "Use absolute image pixel coordinates with (0,0) at the top-left."
        )
    return f"""{coordinate_text}
Inspect only the region inside bbox {list(box)} in xyxy coordinates.
Identify the object centered in that region.
Allowed target_id values: red_circle, blue_square, green_triangle, yellow_star, none.
Return exactly one JSON object and no other text:
{{"target_id":"one_allowed_value"}}"""


def _tracking_prompt(box: Sequence[int], protocol: str, image_size: Sequence[int]) -> str:
    if protocol == "norm1000":
        coordinate_text = "normalized 0-to-1000 xyxy coordinates"
        bbox_key = "bbox_norm1000_xyxy"
    else:
        coordinate_text = (
            f"absolute xyxy pixels on the {image_size[0]} by {image_size[1]} image shown to you"
        )
        bbox_key = "bbox_pixel_xyxy"
    return f"""Image 1 is the unmodified full initialization frame.
The exact target is specified only by bbox {list(box)} in {coordinate_text} on Image 1.
Image 2 is the current full frame; the four objects have moved.
First use the initialization bbox to determine the target identity, then locate that exact
object in Image 2. Return exactly one JSON object and no other text:
Allowed target_id values: red_circle, blue_square, green_triangle, yellow_star.
{{"target_id":"one_allowed_value", "{bbox_key}":[x1,y1,x2,y2]}}"""


def _scale_box(box: Sequence[int], image_size: Sequence[int]) -> list[int]:
    """把 norm1000 坐标换成模型实际看到的绝对像素坐标。"""

    width, height = image_size
    return [
        round(box[0] * width / 1000),
        round(box[1] * height / 1000),
        round(box[2] * width / 1000),
        round(box[3] * height / 1000),
    ]


def _run_protocol_cases(
    backend: HuggingFaceQwenVLBackend,
    reference: Image.Image,
    current: Image.Image,
    *,
    protocol: str,
    image_size: Sequence[int],
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    generation = GenerationConfig(max_new_tokens=96, do_sample=False, temperature=0.0)
    system_prompt = "Follow the coordinate reference exactly. Output strict JSON only."

    for target_id in TARGETS:
        norm_box = REFERENCE_BOXES[target_id]
        box = norm_box if protocol == "norm1000" else _scale_box(norm_box, image_size)
        response = backend.generate(
            [reference],
            _region_prompt(box, protocol, image_size),
            system_prompt=system_prompt,
            generation_config=generation,
        )
        parsed = _extract_json(response.text)
        predicted_id = parsed.get("target_id") if parsed else None
        results.append(
            {
                "test": "single_image_region",
                "input_bbox_protocol": protocol,
                "expected_target_id": target_id,
                "reference_bbox_xyxy": box,
                "raw_response": response.text,
                "parsed_response": parsed,
                "identity_correct": predicted_id == target_id,
                "latency_ms": response.latency_ms,
                "model_image_sizes": response.image_sizes,
            }
        )

    for target_id in TARGETS:
        norm_box = REFERENCE_BOXES[target_id]
        box = norm_box if protocol == "norm1000" else _scale_box(norm_box, image_size)
        expected_norm_bbox = CURRENT_BOXES[target_id]
        expected_bbox = (
            expected_norm_bbox if protocol == "norm1000" else _scale_box(expected_norm_bbox, image_size)
        )
        bbox_key = "bbox_norm1000_xyxy" if protocol == "norm1000" else "bbox_pixel_xyxy"
        response = backend.generate(
            [reference, current],
            _tracking_prompt(box, protocol, image_size),
            system_prompt=system_prompt,
            generation_config=generation,
        )
        parsed = _extract_json(response.text)
        predicted_id = parsed.get("target_id") if parsed else None
        predicted_bbox = parsed.get(bbox_key) if parsed else None
        iou = _bbox_iou(predicted_bbox, expected_bbox) if _is_valid_bbox(predicted_bbox) else 0.0
        results.append(
            {
                "test": "two_image_tracking",
                "input_bbox_protocol": protocol,
                "expected_target_id": target_id,
                "reference_bbox_xyxy": box,
                "expected_current_bbox_xyxy": expected_bbox,
                "raw_response": response.text,
                "parsed_response": parsed,
                "identity_correct": predicted_id == target_id,
                "bbox_iou": iou,
                "bbox_correct_at_0_5": iou >= 0.5,
                "latency_ms": response.latency_ms,
                "model_image_sizes": response.image_sizes,
            }
        )
    return results


def _protocol_summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    region_cases = [case for case in cases if case["test"] == "single_image_region"]
    tracking_cases = [case for case in cases if case["test"] == "two_image_tracking"]
    return {
        "single_image_identity_accuracy": _mean(float(case["identity_correct"]) for case in region_cases),
        "tracking_identity_accuracy": _mean(float(case["identity_correct"]) for case in tracking_cases),
        "tracking_bbox_accuracy_at_iou_0_5": _mean(
            float(case["bbox_correct_at_0_5"]) for case in tracking_cases
        ),
        "tracking_mean_bbox_iou": _mean(float(case["bbox_iou"]) for case in tracking_cases),
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def main() -> int:
    args = _args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _make_scene(REFERENCE_BOXES)
    current = _make_scene(CURRENT_BOXES)
    reference_path = output_dir / "reference.png"
    current_path = output_dir / "current.png"
    reference.save(reference_path)
    current.save(current_path)

    config = _resolve_model_config(args.model_config, args.env_config)
    backend = HuggingFaceQwenVLBackend(config)
    try:
        # 第一组先按 Stage-1 数据协议查询，并从 processor 返回值取得模型真实图像尺寸。
        norm_cases = _run_protocol_cases(
            backend,
            reference,
            current,
            protocol="norm1000",
            image_size=(1000, 1000),
        )
        reported_sizes = norm_cases[0].get("model_image_sizes")
        if not reported_sizes:
            raise RuntimeError("Qwen processor 未返回模型图像尺寸，无法构造绝对像素对照组")
        model_image_size = reported_sizes[0]
        native_cases = _run_protocol_cases(
            backend,
            reference,
            current,
            protocol="qwen_abs_pixel",
            image_size=model_image_size,
        )
        cases = norm_cases + native_cases
    finally:
        backend.unload()

    summary = {
        "model": config.model_name or Path(config.model_path).name,
        "model_image_size": model_image_size,
        "norm1000": _protocol_summary(norm_cases),
        "qwen_abs_pixel": _protocol_summary(native_cases),
        "case_count": len(cases),
    }
    report = {
        "probe": "text_bbox_protocol_comprehension_v2",
        "reference_image": str(reference_path),
        "current_image": str(current_path),
        "summary": summary,
        "cases": cases,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[报告] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
