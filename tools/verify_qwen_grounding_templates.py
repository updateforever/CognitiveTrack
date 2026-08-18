#!/usr/bin/env python3
"""用真实 ms-swift 模板回放一条样本，验证 Qwen 两代坐标转换。

该工具不加载模型权重，只加载本地 processor/tokenizer。可只验证一个已安装模型族，
也可同时验证两代。它分别断言：

* Qwen2.5-VL 的 `<bbox>` 被转换为 processor-resize 后绝对像素；
* Qwen3-VL 的 `<bbox>` 被转换为 0-to-1000 相对坐标；
* 多图 `image_id` 把 assistant 框绑定到当前帧而不是初始化帧。

传入 ``--verify-tracking-core-mask`` 时，还会加载项目的 ms-swift 插件并断言：
bbox 坐标 token 保持监督，只有末尾 ``memory_update`` 的值被标为 ``-100``。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _first_present(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("target_status") == "present":
                return row
    raise ValueError(f"JSONL 中没有 present 样本：{path}")


def _assistant_json(decoded: str) -> dict[str, Any]:
    # 模型可见协议以 bbox 字段开头：{"bbox_2d":[...],"status":"present",...}
    candidates = re.findall(r'\{"bbox_(?:2d|pixel_xyxy)":[^\n]*\}', decoded)
    for text in reversed(candidates):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("无法从模板解码文本中找到 assistant JSON")


def _rounded_box(values: list[float]) -> list[int]:
    return [round(value) for value in values]


def _expected_qwen25(
    real_box: list[float],
    *,
    original_size: tuple[int, int],
    grid: list[int],
    patch_size: int,
) -> list[int]:
    original_width, original_height = original_size
    processed_height = grid[-2] * patch_size
    processed_width = grid[-1] * patch_size
    return _rounded_box(
        [
            real_box[0] * processed_width / original_width,
            real_box[1] * processed_height / original_height,
            real_box[2] * processed_width / original_width,
            real_box[3] * processed_height / original_height,
        ]
    )


def _expected_qwen3(real_box: list[float], *, original_size: tuple[int, int]) -> list[int]:
    width, height = original_size
    return _rounded_box(
        [
            real_box[0] * 1000 / width,
            real_box[1] * 1000 / height,
            real_box[2] * 1000 / width,
            real_box[3] * 1000 / height,
        ]
    )


def _verify_family(
    *,
    family: str,
    model_path: Path,
    dataset_root: Path,
    split: str,
    max_pixels: int,
    verify_tracking_core_mask: bool,
) -> dict[str, Any]:
    try:
        from swift.model import get_processor
        from swift.template import StdTemplateInputs, TemplateInputs, get_template
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 ms-swift") from exc

    row = _first_present(dataset_root / "ms_swift" / family / f"{split}.jsonl")
    image_paths = [dataset_root / value for value in row["images"]]
    processor = get_processor(
        str(model_path),
        model_type=family,
        download_model=False,
        use_fast=False,
    )
    def template_inputs() -> TemplateInputs:
        chosen = StdTemplateInputs(
            system=row["messages"][0]["content"],
            messages=deepcopy(row["messages"][1:]),
            images=[str(path) for path in image_paths],
            objects=deepcopy(row["objects"]),
        )
        return TemplateInputs(chosen=chosen)

    template = get_template(processor, max_pixels=max_pixels)
    encoded = template.encode(template_inputs())
    decoded = processor.tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    if "<bbox>" in decoded:
        raise AssertionError(f"{family} 模板编码后仍残留 <bbox>")

    payload = _assistant_json(decoded)
    current_object_index = len(row["objects"]["bbox"]) - 1
    current_image_index = row["objects"]["image_id"][current_object_index]
    real_box = row["objects"]["bbox"][current_object_index]
    with Image.open(image_paths[current_image_index]) as image:
        original_size = image.size
    grids = encoded["image_grid_thw"].tolist()

    if family == "qwen2_5_vl":
        field = "bbox_pixel_xyxy"
        patch_size = int(processor.image_processor.patch_size)
        expected = _expected_qwen25(
            real_box,
            original_size=original_size,
            grid=grids[current_image_index],
            patch_size=patch_size,
        )
        expected_norm_bbox = "none"
    elif family == "qwen3_vl":
        field = "bbox_2d"
        expected = _expected_qwen3(real_box, original_size=original_size)
        expected_norm_bbox = "norm1000"
    else:  # pragma: no cover - argparse 已限制
        raise ValueError(f"未知模型族：{family}")

    actual = payload.get(field)
    if actual != expected:
        raise AssertionError(f"{family} 坐标不一致：expected={expected}, actual={actual}")
    if template.norm_bbox != expected_norm_bbox:
        raise AssertionError(
            f"{family} norm_bbox 不一致：expected={expected_norm_bbox}, actual={template.norm_bbox}"
        )
    report = {
        "model_family": family,
        "sample_id": row.get("id"),
        "bbox_field": field,
        "template_norm_bbox": template.norm_bbox,
        "original_current_size": list(original_size),
        "image_grid_thw": grids,
        "objects_current_real_xyxy": real_box,
        "expected_model_bbox": expected,
        "decoded_model_bbox": actual,
        "ok": True,
    }
    if verify_tracking_core_mask:
        metadata = row.get("metadata", {})
        if metadata.get("sft_supervision_profile") != "tracking_core":
            raise AssertionError("待检查样本未声明 sft_supervision_profile=tracking_core")
        if metadata.get("memory_loss_masked") is not True:
            raise AssertionError("待检查样本未声明 memory_loss_masked=true")
        # 复用 swift sft 的真实外部插件导入方式；不修改 ms-swift 安装目录。
        from swift.utils import import_external_file

        import_external_file(
            str(PROJECT_ROOT / "cogtrack" / "training" / "ms_swift_plugin.py")
        )

        default_train = get_template(processor, max_pixels=max_pixels, loss_scale="default")
        masked_train = get_template(
            processor,
            max_pixels=max_pixels,
            loss_scale="cogtrack_tracking_core",
        )
        default_train.set_mode("train")
        masked_train.set_mode("train")
        default_encoded = default_train.encode(template_inputs())
        masked_encoded = masked_train.encode(template_inputs())
        if default_encoded["input_ids"] != masked_encoded["input_ids"]:
            raise AssertionError("启用 loss mask 后 input_ids 发生变化")
        masked_indices = [
            index
            for index, (default_label, masked_label) in enumerate(
                zip(default_encoded["labels"], masked_encoded["labels"], strict=True)
            )
            if default_label != -100 and masked_label == -100
        ]
        masked_text = processor.tokenizer.decode(
            [masked_encoded["input_ids"][index] for index in masked_indices],
            skip_special_tokens=False,
        )
        if masked_text != "null":
            raise AssertionError(f"memory loss mask 范围错误：masked={masked_text!r}")
        supervised_text = processor.tokenizer.decode(
            [
                token_id
                for token_id, label in zip(
                    masked_encoded["input_ids"],
                    masked_encoded["labels"],
                    strict=True,
                )
                if label != -100
            ],
            skip_special_tokens=False,
        )
        expected_bbox_text = f'"{field}":{json.dumps(expected)}'
        if expected_bbox_text not in supervised_text:
            raise AssertionError("bbox 坐标 token 被错误屏蔽")
        if '"memory_update":}' not in supervised_text:
            raise AssertionError("memory_update 键或 JSON 闭合符未保持监督")
        report["tracking_core_loss_mask"] = {
            "masked_text": masked_text,
            "masked_token_count": len(masked_indices),
            "bbox_tokens_supervised": True,
            "memory_key_and_json_close_supervised": True,
            "ok": True,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--qwen25-model")
    parser.add_argument("--qwen3-model")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--verify-tracking-core-mask",
        action="store_true",
        help="额外验证 VLT-v6 核心 SFT 只屏蔽 memory_update 值。",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    os.environ["QWENVL_BBOX_FORMAT"] = "new"
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    requested = []
    if args.qwen25_model:
        requested.append(("qwen2_5_vl", args.qwen25_model))
    if args.qwen3_model:
        requested.append(("qwen3_vl", args.qwen3_model))
    if not requested:
        parser.error("至少提供 --qwen25-model 或 --qwen3-model 之一")
    reports = [
        _verify_family(
            family=family,
            model_path=Path(model_path).expanduser().resolve(),
            dataset_root=dataset_root,
            split=args.split,
            max_pixels=args.max_pixels,
            verify_tracking_core_mask=args.verify_tracking_core_mask,
        )
        for family, model_path in requested
    ]
    result = {
        "schema_version": "cogtrack.qwen_template_verification.v1",
        "qwen_vl_bbox_format": "new",
        "max_pixels": args.max_pixels,
        "reports": reports,
        "ok": all(report["ok"] for report in reports),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
