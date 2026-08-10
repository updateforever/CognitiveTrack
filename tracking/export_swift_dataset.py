#!/usr/bin/env python3
"""转换、校验并按序列划分 ms-swift 多模态训练 JSONL。

本工具不绑定某个数据集。上游样本既可以已经是 ``messages + images``，也可
使用简化字段 ``prompt/user_prompt + answer/assistant/target + images``。

示例：

    python tracking/export_swift_dataset.py \
        --input generated_samples.jsonl \
        --output-dir data/cognitive_sft \
        --image-root data/cognitive_sft \
        --val-ratio 0.05 --mode sft

生成 GRPO 数据时使用 ``--mode grpo``，工具会把 assistant 参考答案移到
``solution`` 并从输入消息中删除，避免标签泄漏。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training import (  # noqa: E402
    read_jsonl,
    split_records_by_sequence,
    to_grpo_record,
    to_ms_swift_record,
    validate_records,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个源 JSONL。")
    parser.add_argument("--output-dir", required=True, help="train/val/test JSONL 输出目录。")
    parser.add_argument(
        "--image-root",
        default=None,
        help="相对图片路径的根目录；单输入时默认使用输入文件所在目录。",
    )
    parser.add_argument("--mode", choices=("sft", "grpo"), default="sft")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--default-system-prompt",
        default="You are a cognitive visual tracking assistant.",
    )
    parser.add_argument("--no-check-images", action="store_true", help="不检查图片是否存在。")
    parser.add_argument("--allow-absolute-images", action="store_true", help="允许不可移植的绝对图片路径。")
    parser.add_argument("--no-check-answer-json", action="store_true", help="不检查 assistant JSON schema。")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="跳过非法样本继续导出；默认遇到非法样本即失败。",
    )
    return parser.parse_args()


def _sequence_count(rows: list[dict]) -> int:
    keys = set()
    for row in rows:
        metadata = row.get("metadata", {})
        dataset = metadata.get("source_dataset", metadata.get("dataset", "unknown"))
        sequence = metadata.get("source_sequence", metadata.get("sequence"))
        if sequence is not None:
            keys.add((str(dataset), str(sequence)))
    return len(keys)


def main() -> None:
    args = parse_args()
    inputs = [Path(value).resolve() for value in args.input]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.image_root:
        image_root = Path(args.image_root).resolve()
    elif len(inputs) == 1:
        image_root = inputs[0].parent
    else:
        image_root = Path.cwd().resolve()

    rows = []
    for path in inputs:
        for source_row in read_jsonl(path):
            record = to_ms_swift_record(
                source_row,
                default_system_prompt=args.default_system_prompt,
            )
            if args.mode == "grpo":
                record = to_grpo_record(record)
            rows.append(record)
    if not rows:
        raise ValueError("输入 JSONL 中没有样本")

    report = validate_records(
        rows,
        image_root=image_root,
        check_images=not args.no_check_images,
        allow_absolute_images=args.allow_absolute_images,
        check_answer_json=not args.no_check_answer_json,
    )
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    invalid_indices = {issue.sample_index for issue in report.errors if issue.sample_index is not None}
    if invalid_indices and not args.skip_invalid:
        preview = "; ".join(issue.message for issue in report.errors[:5])
        raise ValueError(
            f"发现 {len(invalid_indices)} 条非法样本，详情见 {report_path}。"
            f"前几项：{preview}"
        )
    valid_rows = [row for index, row in enumerate(rows) if index not in invalid_indices]
    if not valid_rows:
        raise ValueError("没有可导出的合法样本")

    splits = split_records_by_sequence(
        valid_rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    stats = {}
    for split, split_rows in splits.items():
        if not split_rows and split == "test" and args.test_ratio == 0:
            continue
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        stats[split] = {
            "samples": len(split_rows),
            "sequences": _sequence_count(split_rows),
            "path": path.name,
        }

    info = {
        "schema_version": "cogtrack.ms_swift.v4",
        "mode": args.mode,
        "format": "ms-swift messages+images JSONL",
        "image_root": str(image_root),
        "image_paths_portable": not args.allow_absolute_images,
        "split_unit": "sequence",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "input_files": [str(path) for path in inputs],
        "source_samples": len(rows),
        "exported_samples": len(valid_rows),
        "skipped_invalid_samples": len(invalid_indices),
        "splits": stats,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CognitiveTrack] {args.mode.upper()} 数据导出完成："
        f"source={len(rows)} exported={len(valid_rows)} output={output_dir}"
    )
    for split, values in stats.items():
        print(
            f"[CognitiveTrack] {split}: samples={values['samples']} "
            f"sequences={values['sequences']} file={output_dir / values['path']}"
        )
    print(f"[CognitiveTrack] 校验报告：{report_path}")


if __name__ == "__main__":
    main()
