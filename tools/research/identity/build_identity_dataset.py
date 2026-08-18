#!/usr/bin/env python3
"""构建同类别跨序列的候选级身份困难负样本。

示例：

    python tools/research/identity/build_identity_dataset.py \
        --dataset lasot --split test \
        --env-config configs/env.local.yaml \
        --output-dir data/identity_hard_negatives \
        --max-candidate-frames 8 --frame-stride 20

该工具属于隔离的未来 identity 研究线。CognitiveTrack v4 主训练协议只接受
``target_status + bbox``，因此当前导出器会主动拒绝这里的 identity 样本，避免
两类监督误混。恢复身份训练前应单独设计并版本化 identity 协议。

安全约束：构造器只接受显式且相同的 ``object_class``，只把不同实例来源的
序列标为 ``different``。每个实例最多属于一个配对组，现有按 source_sequence
划分器可防止本负样本集内部泄漏。若还要和普通逐序列 tracking 样本合并，应先
按实例生成统一 split 清单，再在各 split 内合并，不能重新做逐行随机划分。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 本工具已从正式 tracking CLI 隔离到 tools/research/identity/；向上三级才是仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.identity_samples import (  # noqa: E402
    IdentitySampleConfig,
    build_identity_samples,
)
from pytracking.datasets.registry import list_datasets, load_dataset  # noqa: E402
from pytracking.evaluation.data import SequenceList  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建可追溯的同类别跨序列身份困难负样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        choices=list_datasets(),
        help="pytracking 数据集注册名；可重复指定。",
    )
    parser.add_argument("--split", help="覆盖全部输入数据集的 split。")
    parser.add_argument(
        "--sequence",
        action="append",
        help="只加载指定序列；仅允许和单个 --dataset 一起使用，可重复。",
    )
    parser.add_argument("--limit-sequences", type=int, help="每个数据集最多加载前 N 个序列。")
    parser.add_argument("--env-config", help="本机环境 YAML；默认按框架环境规则发现。")
    parser.add_argument("--output-dir", required=True, help="源 JSONL 与图片资产输出目录。")
    parser.add_argument("--frame-stride", type=int, default=1, help="候选帧步长。")
    parser.add_argument(
        "--max-candidate-frames",
        type=int,
        default=8,
        help="每个配对方向最多候选帧数；默认 8。",
    )
    parser.add_argument("--seed", type=int, default=20260805, help="配对与帧采样的固定种子。")
    parser.add_argument("--one-way", action="store_true", help="每个实例对只生成一个方向；默认双向。")
    parser.add_argument("--keyframes-only", action="store_true", help="候选帧仅取关键帧。")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--force", action="store_true", help="覆盖同名 JSONL 与图片资产。")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.limit_sequences is not None and args.limit_sequences <= 0:
            raise ValueError("--limit-sequences 必须为正整数")
        if args.sequence and len(args.dataset) != 1:
            raise ValueError("--sequence 与多个 --dataset 组合时语义不明确，请分开构建")

        environment = load_environment(args.env_config)
        sequences = SequenceList()
        dataset_kwargs = {"split": args.split} if args.split else {}
        for dataset_name in args.dataset:
            current = load_dataset(
                dataset_name,
                environment=environment,
                sequence_names=args.sequence,
                limit=args.limit_sequences if args.sequence is None else None,
                **dataset_kwargs,
            )
            if args.sequence is not None and args.limit_sequences is not None:
                current = SequenceList(current[: args.limit_sequences])
            sequences.extend(current)

        config = IdentitySampleConfig(
            frame_stride=args.frame_stride,
            max_candidate_frames=args.max_candidate_frames,
            seed=args.seed,
            bidirectional=not args.one_way,
            keyframes_only=args.keyframes_only,
            jpeg_quality=args.jpeg_quality,
        )
        report = build_identity_samples(
            sequences,
            args.output_dir,
            config=config,
            overwrite=args.force,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    print("[CognitiveTrack] 身份困难负样本构建完成")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"[隔离输出] {Path(args.output_dir) / report.source_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
