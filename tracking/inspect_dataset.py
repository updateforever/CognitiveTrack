#!/usr/bin/env python3
"""快速检查数据集结构、标注长度与关键帧信息。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytracking.datasets.registry import list_datasets, load_dataset  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError("数据集配置顶层必须是 mapping")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 CognitiveTrack 数据集 loader")
    parser.add_argument("--dataset", choices=list_datasets(), help="数据集注册名")
    parser.add_argument("--dataset-config", help="包含 name/split/root/source_roots 的 YAML")
    parser.add_argument("--config", "--env-config", dest="env_config", help="本机环境 YAML")
    parser.add_argument("--split", help="覆盖 split")
    parser.add_argument("--sequence", action="append", help="仅检查指定序列，可重复传入")
    parser.add_argument("--limit", type=int, default=5, help="未指定序列时展示前 N 个")
    parser.add_argument("--check-files", action="store_true", help="逐帧检查图像路径是否存在（可能较慢）")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        dataset_config = _read_yaml(args.dataset_config) if args.dataset_config else {}
        dataset_name = args.dataset or dataset_config.get("name")
        if not dataset_name:
            raise ValueError("请提供 --dataset 或 --dataset-config")
        split = args.split or dataset_config.get("split")
        overrides: dict[str, str] = {}
        if dataset_config.get("root"):
            overrides[str(dataset_name).lower()] = str(dataset_config["root"])
        source_roots = dataset_config.get("source_roots", {}) or {}
        if not isinstance(source_roots, dict):
            raise TypeError("source_roots 必须是 mapping")
        overrides.update({str(name).lower(): str(path) for name, path in source_roots.items()})
        environment = load_environment(args.env_config, overrides=overrides)

        # 显式序列和 limit 均走 loader 的按名称快路径，避免仅为结构检查就展开
        # CognitiveBench 上百万个帧路径；无论哪种模式都不会读取图像像素。
        dataset_kwargs = {"split": str(split)} if split is not None else {}
        sequences = load_dataset(
            str(dataset_name),
            environment=environment,
            sequence_names=args.sequence,
            limit=None if args.sequence else max(args.limit, 0),
            **dataset_kwargs,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    print(f"dataset={dataset_name} split={split or 'registry_default'} inspected_sequences={len(sequences)}")
    for sequence in sequences:
        visible_count = int(np.count_nonzero(sequence.target_visible)) if sequence.target_visible is not None else None
        ground_truth_count = len(sequence.ground_truth_rect) if sequence.ground_truth_rect is not None else 0
        missing = 0
        if args.check_files:
            missing = sum(not Path(path).is_file() for path in sequence.frames)
        print(
            f"- {sequence.name}: frames={len(sequence)}, gt={ground_truth_count}, "
            f"visible={visible_count if visible_count is not None else 'unknown'}, "
            f"keyframes={len(sequence.keyframe_indices)}, language={'yes' if sequence.language_query else 'no'}, "
            f"missing_images={missing if args.check_files else 'not_checked'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
