#!/usr/bin/env python3
"""从 canonical Stage-1 JSONL 导出 Qwen 官方 Grounding 训练视图。

默认同时生成：

* ``qwen2_5_vl``：processor resize 后绝对像素坐标；
* ``qwen3_vl``：0–1000 相对坐标。

两者都使用 ms-swift ``<bbox> + objects.bbox + image_id``，不把待转换坐标作为
普通文本硬编码。这样坐标变换由实际模型 processor 完成，并能严格对应两张图。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training import (  # noqa: E402
    QWEN_MODEL_FAMILIES,
    export_qwen_grounding_records,
    read_jsonl,
    split_records_by_sequence,
    validate_records,
    write_jsonl,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="canonical source_samples.jsonl。")
    parser.add_argument("--dataset-root", required=True, help="相对 images 路径的数据集根目录。")
    parser.add_argument("--output-dir", required=True, help="模型专属 train/val JSONL 输出目录。")
    parser.add_argument(
        "--model-families",
        nargs="+",
        choices=QWEN_MODEL_FAMILIES,
        default=list(QWEN_MODEL_FAMILIES),
    )
    parser.add_argument(
        "--split-from",
        help="可选：已有 train.jsonl/val.jsonl 目录；按 id 原样复用其序列划分。",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def _source_seed(base_seed: int, source: str) -> int:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], byteorder="big")


def _stratified_split(
    records: list[dict[str, Any]], *, val_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("metadata", {}).get("source_dataset", "unknown"))].append(record)
    result = {"train": [], "val": []}
    for source in sorted(grouped):
        split = split_records_by_sequence(
            grouped[source],
            val_ratio=val_ratio,
            test_ratio=0.0,
            seed=_source_seed(seed, source),
        )
        result["train"].extend(split["train"])
        result["val"].extend(split["val"])
    return result


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id")
    if value is None and isinstance(record.get("metadata"), Mapping):
        value = record["metadata"].get("id")
    if value is None:
        raise ValueError("复用已有 split 需要每条样本包含 id")
    return str(value)


def _reuse_split(
    records: list[dict[str, Any]], *, split_root: Path
) -> dict[str, list[dict[str, Any]]]:
    split_by_id: dict[str, str] = {}
    for split in ("train", "val"):
        path = split_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"已有 split 文件不存在：{path}")
        for record in read_jsonl(path):
            sample_id = _record_id(record)
            previous = split_by_id.setdefault(sample_id, split)
            if previous != split:
                raise ValueError(f"样本 {sample_id} 同时出现在 {previous}/{split}")
    result = {"train": [], "val": []}
    for record in records:
        sample_id = _record_id(record)
        split = split_by_id.get(sample_id)
        if split is None:
            raise ValueError(f"canonical 样本未出现在已有 split：{sample_id}")
        result[split].append(record)
    if len(split_by_id) != len(records):
        raise ValueError("已有 split 与 canonical 样本数量不一致")
    return result


def _sequence_count(records: list[dict[str, Any]]) -> int:
    return len(
        {
            (
                str(record["metadata"].get("source_dataset")),
                str(record["metadata"].get("source_sequence")),
            )
            for record in records
        }
    )


def _update_root_dataset_info(
    dataset_root: Path,
    *,
    output_root: Path,
    families: list[str],
) -> None:
    """登记模型专属训练视图，避免把 canonical 坐标误作模型训练协议。"""

    path = dataset_root / "dataset_info.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"根 dataset_info.json 必须是 JSON 对象：{path}")
    legacy_bbox = payload.pop("bbox_format", None)
    legacy_reference_bbox = payload.pop("reference_bbox_format", None)
    payload.setdefault("canonical_bbox_format", legacy_bbox or "norm1000_xyxy")
    payload.setdefault(
        "reference_canonical_bbox_format",
        legacy_reference_bbox or "norm1000_xyxy",
    )
    payload["training_coordinate_protocols"] = {
        "qwen2_5_vl": "processor-resized absolute pixel xyxy",
        "qwen3_vl": "relative 0-to-1000 xyxy",
    }
    payload["coordinate_conversion_owner"] = "ms-swift model template"
    payload["qwen_vl_bbox_format"] = "new"
    payload["training_views"] = {
        family: (output_root / family).relative_to(dataset_root).as_posix()
        for family in families
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _args()
    source_path = Path(args.input).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    rows = list(read_jsonl(source_path))
    if not rows:
        raise ValueError("canonical JSONL 中没有样本")
    if args.split_from:
        canonical_splits = _reuse_split(rows, split_root=Path(args.split_from).expanduser().resolve())
        split_method = "reused_by_sample_id"
    else:
        canonical_splits = _stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
        split_method = "source_stratified_complete_sequence"

    reports = {}
    for family in dict.fromkeys(args.model_families):
        family_root = output_root / family
        family_stats = {}
        aggregate = None
        for split, split_rows in canonical_splits.items():
            records, report = export_qwen_grounding_records(
                split_rows,
                image_root=dataset_root,
                model_family=family,
            )
            validation = validate_records(records, image_root=dataset_root, check_images=True)
            if not validation.ok:
                preview = "; ".join(issue.message for issue in validation.errors[:5])
                raise ValueError(f"{family}/{split} 校验失败：{preview}")
            write_jsonl(family_root / f"{split}.jsonl", records)
            (family_root / f"validation_{split}.json").write_text(
                json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            family_stats[split] = {
                "samples": len(records),
                "sequences": _sequence_count(records),
                "present": report.present_count,
                "absent": report.absent_count,
            }
            if aggregate is None:
                aggregate = report.to_dict()
            else:
                aggregate["sample_count"] += report.sample_count
                aggregate["present_count"] += report.present_count
                aggregate["absent_count"] += report.absent_count
                aggregate["bbox_placeholder_count"] += report.bbox_placeholder_count
        reports[family] = {"export": aggregate, "splits": family_stats}
        (family_root / "dataset_info.json").write_text(
            json.dumps(
                {
                    "schema_version": "cogtrack.qwen_grounding_dataset.v1",
                    "model_family": family,
                    "qwen_vl_bbox_format": "new",
                    "coordinate_conversion_owner": "ms-swift model template",
                    "canonical_bbox_type": "real exported-image xyxy",
                    "split_method": split_method,
                    "source": source_path.name,
                    **reports[family],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "cogtrack.qwen_grounding_multi_family.v1",
        # 该文件会随数据集上传，禁止记录构建机绝对路径。
        "source": source_path.relative_to(dataset_root).as_posix(),
        "dataset_root": ".",
        "split_method": split_method,
        "qwen_vl_bbox_format": "new",
        "families": reports,
    }
    (output_root / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _update_root_dataset_info(
        dataset_root,
        output_root=output_root,
        families=list(dict.fromkeys(args.model_families)),
    )
    print("[CognitiveTrack] Qwen 官方 Grounding 训练视图导出完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
