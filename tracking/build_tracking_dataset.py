#!/usr/bin/env python3
"""从标准 pytracking 数据集构建 ms-swift 跟踪训练源数据。

Pair 示例：

    python tracking/build_tracking_dataset.py \
        --dataset cognitivebench --split test \
        --env-config configs/env.local.yaml \
        --output-dir data/cognitive_pair --mode pair \
        --frame-stride 10 --max-samples-per-sequence 64

Mosaic 示例（无可信历史的早期帧会按在线推理规则自动退化为 pair）：

    python tracking/build_tracking_dataset.py \
        --dataset cognitivebench --env-config configs/env.local.yaml \
        --output-dir data/cognitive_mosaic --mode mosaic \
        --history-size 4 --keyframes-only

构建完成后，执行下面任一命令生成最终训练划分：

    python tracking/export_swift_dataset.py --input data/cognitive_pair/source_samples.jsonl \
        --output-dir data/cognitive_pair/sft --image-root data/cognitive_pair --mode sft
    python tracking/export_swift_dataset.py --input data/cognitive_pair/source_samples.jsonl \
        --output-dir data/cognitive_pair/grpo --image-root data/cognitive_pair --mode grpo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.tracking_samples import (  # noqa: E402
    TrackingSampleConfig,
    build_tracking_samples,
)
from pytracking.datasets.registry import list_datasets, load_dataset  # noqa: E402
from pytracking.evaluation.data import SequenceList  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建自包含的认知跟踪 pair/mosaic 训练源 JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=list_datasets(), help="pytracking 数据集注册名。")
    parser.add_argument("--split", help="覆盖数据集 split；默认使用注册表配置。")
    parser.add_argument("--sequence", action="append", help="只构建指定序列；可重复传入。")
    parser.add_argument("--limit-sequences", type=int, help="仅取按名称排序后的前 N 个序列。")
    parser.add_argument("--env-config", help="本机环境 YAML；默认按框架环境规则发现。")
    parser.add_argument("--output-dir", required=True, help="源 JSONL 与 images/ 的输出根目录。")
    parser.add_argument("--mode", choices=("pair", "mosaic", "both"), default="pair")
    parser.add_argument("--frame-stride", type=int, default=1, help="候选当前帧步长，默认逐帧。")
    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        help="每序列最多样本数；使用固定 seed 做稳定抽样。",
    )
    parser.add_argument("--seed", type=int, default=20260805, help="稳定抽样随机种子。")
    parser.add_argument("--history-size", type=int, default=4, help="mosaic 最多使用的过去正历史帧数。")
    parser.add_argument("--mosaic-panel-height", type=int, default=240)
    parser.add_argument(
        "--max-image-side",
        type=int,
        help="可选：等比例限制导出图片长边；norm1000 bbox 不受整体缩放影响。",
    )
    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help="只从 Sequence.keyframe_indices 采样；适用于带关键帧标注的数据集。",
    )
    parser.add_argument(
        "--balance-presence",
        action="store_true",
        help="在每个序列的采样上限内优先均衡 present/absent；不足的一类由另一类补足。",
    )
    parser.add_argument(
        "--present-only",
        action="store_true",
        help="只生成目标可见且 bbox 有效的样本；用于 Stage-1 跟踪定位预热。",
    )
    parser.add_argument(
        "--no-language-description",
        action="store_true",
        help="不把数据集语言描述写入 Prompt，避免描述泄漏并仅依赖初始化图确认身份。",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="图片资产 JPEG 质量 [1,100]。")
    parser.add_argument("--force", action="store_true", help="覆盖同名 JSONL 与图片资产。")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.limit_sequences is not None and args.limit_sequences <= 0:
            raise ValueError("--limit-sequences 必须为正整数")
        environment = load_environment(args.env_config)
        dataset_kwargs = {"split": args.split} if args.split else {}
        sequences = load_dataset(
            args.dataset,
            environment=environment,
            sequence_names=args.sequence,
            limit=args.limit_sequences if args.sequence is None else None,
            **dataset_kwargs,
        )
        if args.sequence is not None and args.limit_sequences is not None:
            sequences = SequenceList(sequences[: args.limit_sequences])

        config = TrackingSampleConfig(
            mode=args.mode,
            frame_stride=args.frame_stride,
            max_samples_per_sequence=args.max_samples_per_sequence,
            seed=args.seed,
            history_size=args.history_size,
            mosaic_panel_height=args.mosaic_panel_height,
            keyframes_only=args.keyframes_only,
            balance_presence=args.balance_presence,
            present_only=args.present_only,
            use_language_description=not args.no_language_description,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
        )
        report = build_tracking_samples(
            sequences,
            args.output_dir,
            config=config,
            overwrite=args.force,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    print("[CognitiveTrack] 跟踪训练源数据构建完成")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"[下一步] 使用 tracking/export_swift_dataset.py 转换 {Path(args.output_dir) / report.source_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
