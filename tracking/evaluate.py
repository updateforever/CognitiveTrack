#!/usr/bin/env python3
"""评测 CognitiveTrack 帧级 JSONL 结果。

示例：

    python tracking/evaluate.py \
        --input outputs/qwen_pair/cognitivebench \
        --output-dir outputs/qwen_pair/cognitivebench/evaluation

输入既可以是单个 ``*_frames.jsonl``，也可以是包含多个序列结果的目录。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.evaluation import (  # noqa: E402
    discover_jsonl_files,
    evaluate_jsonl_files,
    partition_debug_limited,
    write_evaluation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--result-root",
        dest="input",
        required=True,
        help="单个 JSONL 文件或结果目录；--result-root 是兼容别名。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="报告输出目录；默认写入输入目录下的 evaluation。",
    )
    parser.add_argument(
        "--pattern",
        default="*_frames.jsonl",
        help="目录递归匹配模式，默认：*_frames.jsonl。",
    )
    parser.add_argument(
        "--recovery-iou-threshold",
        type=float,
        default=0.5,
        help="重现后判定成功恢复所需的最低 IoU，默认 0.5。",
    )
    parser.add_argument(
        "--include-debug-runs",
        action="store_true",
        help=(
            "把 --debug-frames 截断跑的结果也计入指标。默认跳过："
            "截断序列只有前若干帧，却会在按序列宏平均时与完整序列等权。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    files = discover_jsonl_files(input_path, pattern=args.pattern)
    if not args.include_debug_runs:
        files, skipped = partition_debug_limited(files)
        if skipped:
            print(
                f"[CognitiveTrack] 跳过 {len(skipped)} 个 debug 截断结果"
                "（--debug-frames 跑出的部分序列，会扭曲按序列宏平均）："
            )
            for path in skipped:
                print(f"    {path}")
            print("  如需纳入统计请加 --include-debug-runs。")
        if not files:
            raise SystemExit(
                "所有结果都是 debug 截断跑，没有可评测的完整序列。"
                "请跑完整序列，或加 --include-debug-runs 强制评测。"
            )
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif input_path.is_file():
        output_dir = input_path.parent / f"{input_path.stem}_evaluation"
    else:
        output_dir = input_path / "evaluation"

    summary = evaluate_jsonl_files(
        files,
        recovery_iou_threshold=args.recovery_iou_threshold,
    )
    paths = write_evaluation_outputs(summary, output_dir)

    main_metrics = summary["pytracking"]
    sparse_metrics = summary.get("pytracking_sparse", {})
    diagnostics = summary["cognitive_diagnostics"]
    presence = diagnostics["presence"]
    recovery = diagnostics["reappearance"]
    observation_rate = main_metrics.get("sparsity", {}).get("observation_rate")
    is_sparse = observation_rate is not None and observation_rate < 1.0

    def _fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}"

    heading = (
        "[CognitiveTrack] 评测完成（稀疏实验并列报告 hold-last 与 observation-only）："
        if is_sparse
        else "[CognitiveTrack] 评测完成（主指标为 pytracking 口径）："
    )
    print(
        heading
        + " "
        f"sequences={summary['num_sequences']} "
        f"valid={main_metrics['num_valid_sequences']} "
        f"frames={summary['num_frames']}"
    )
    print(
        f"  [{'dense_zero/兼容' if is_sparse else 'dense'}] "
        f"AUC={_fmt(main_metrics['success_auc'])} "
        f"OP50={_fmt(main_metrics['success_op50'])} "
        f"OP75={_fmt(main_metrics['success_op75'])} "
        f"P={_fmt(main_metrics['precision_p20'])} "
        f"Pnorm={_fmt(main_metrics['norm_precision_np20'])}"
    )
    for convention in ("hold_last", "observation_only"):
        values = sparse_metrics.get(convention)
        if not values:
            continue
        print(
            f"  [{convention}] AUC={_fmt(values['success_auc'])} "
            f"OP50={_fmt(values['success_op50'])} "
            f"OP75={_fmt(values['success_op75'])} "
            f"P={_fmt(values['precision_p20'])} "
            f"Pnorm={_fmt(values['norm_precision_np20'])}"
        )
    print(
        f"  [presence] F1={_fmt(presence['f1'])} "
        f"absent_FPR={_fmt(presence['false_positive_rate'])} "
        f"present_miss={_fmt(presence['miss_rate'])} "
        f"decision_coverage={_fmt(presence['decision_coverage'])} "
        f"recovery_rate={_fmt(recovery['recovery_rate'])}"
    )
    for name, path in paths.items():
        print(f"[CognitiveTrack] {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
