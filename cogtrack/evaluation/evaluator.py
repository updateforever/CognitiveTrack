"""统一 JSONL 评测入口与报告落盘。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .io import CanonicalFrame, load_frame_records
from .metrics import (
    aggregate_benchmark_standard,
    evaluate_benchmark_sequence,
    evaluate_cognitive_visible_only,
    evaluate_execution,
    evaluate_identity,
    evaluate_presence,
    evaluate_reappearance,
    safe_div,
)
from .pytracking_metrics import extract_results_from_canonical_frames
from .report import build_markdown_report


def _group_sequences(frames: Iterable[CanonicalFrame]) -> dict[str, list[CanonicalFrame]]:
    grouped: dict[str, list[CanonicalFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.sequence].append(frame)
    for sequence, sequence_frames in grouped.items():
        sequence_frames.sort(key=lambda item: item.frame_id)
        for previous, current in zip(sequence_frames, sequence_frames[1:], strict=False):
            if previous.frame_id == current.frame_id:
                raise ValueError(
                    f"发现重复帧：sequence={sequence!r}, frame_id={current.frame_id}。"
                    "请确认输入目录中没有混入多个 tracker/run 的结果。"
                )
    return dict(sorted(grouped.items()))


def summarize_pytracking_curves(raw_results: dict[str, Any]) -> dict[str, Any]:
    """从 pytracking 原始曲线提取常用标量摘要（AUC、OP50、P20 等）。"""
    import torch

    overlap = torch.tensor(raw_results["ave_success_rate_plot_overlap"])  # (N, 1, 21)
    center = torch.tensor(raw_results["ave_success_rate_plot_center"])  # (N, 1, 51)
    center_norm = torch.tensor(raw_results["ave_success_rate_plot_center_norm"])  # (N, 1, 51)
    valid = torch.tensor(raw_results["valid_sequence"], dtype=torch.bool)  # (N,)

    if valid.sum() == 0:
        return {
            "num_valid_sequences": 0,
            "success_auc": None,
            "success_op50": None,
            "success_op75": None,
            "precision_p20": None,
            "norm_precision_np20": None,
        }

    # 只对有效序列做宏平均
    overlap_valid = overlap[valid].mean(dim=0).squeeze()  # (21,)
    center_valid = center[valid].mean(dim=0).squeeze()  # (51,)
    center_norm_valid = center_norm[valid].mean(dim=0).squeeze()  # (51,)

    success_auc = overlap_valid.mean().item()
    success_op50 = overlap_valid[10].item()  # threshold 0.5 在第 10 个位置
    success_op75 = overlap_valid[15].item()  # threshold 0.75 在第 15 个位置
    precision_p20 = center_valid[20].item()  # 20 像素在第 20 个位置
    norm_precision_np20 = center_norm_valid[20].item()  # 0.20 在第 20 个位置

    return {
        "num_valid_sequences": int(valid.sum().item()),
        "success_auc": float(success_auc),
        "success_op50": float(success_op50),
        "success_op75": float(success_op75),
        "precision_p20": float(precision_p20),
        "norm_precision_np20": float(norm_precision_np20),
        "curves": {
            "success": overlap_valid.tolist(),
            "precision": center_valid.tolist(),
            "norm_precision": center_norm_valid.tolist(),
        },
    }


def _evaluate_one_sequence(
    frames: Sequence[CanonicalFrame],
    *,
    recovery_iou_threshold: float,
) -> dict[str, Any]:
    benchmark_standard = evaluate_benchmark_sequence(frames)
    cognitive_visible_only = evaluate_cognitive_visible_only(frames)
    return {
        "num_frames": len(frames),
        "benchmark_standard": benchmark_standard,
        "cognitive_visible_only": cognitive_visible_only,
        # 保留旧键，避免既有分析脚本立即失效；其语义始终是可见帧微平均。
        "tracking": cognitive_visible_only,
        "presence": evaluate_presence(frames),
        "identity": evaluate_identity(frames),
        "execution": evaluate_execution(frames),
        "reappearance": evaluate_reappearance(
            frames,
            recovery_iou_threshold=recovery_iou_threshold,
        ),
    }


def _aggregate_reappearance(
    sequence_metrics: Mapping[str, Mapping[str, Any]],
    *,
    recovery_iou_threshold: float,
) -> dict[str, Any]:
    events = 0
    recovered = 0
    unrecovered = 0
    delays: list[int] = []
    for metrics in sequence_metrics.values():
        recovery = metrics["reappearance"]
        events += int(recovery["events"])
        recovered += int(recovery["recovered_events"])
        unrecovered += int(recovery["unrecovered_events"])
        delays.extend(int(value) for value in recovery.get("recovery_delays", []))
    return {
        "iou_threshold": recovery_iou_threshold,
        "events": events,
        "recovered_events": recovered,
        "unrecovered_events": unrecovered,
        "recovery_rate": safe_div(recovered, events),
        "mean_recovery_delay": safe_div(sum(delays), len(delays)),
        "median_recovery_delay": float(median(delays)) if delays else None,
        "max_recovery_delay": max(delays) if delays else None,
        "recovery_delays": delays,
    }


def evaluate_frames(
    frames: Sequence[CanonicalFrame],
    *,
    source_files: Sequence[str] = (),
    recovery_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """评测已经归一化的帧列表，返回完整聚合摘要。

    ``benchmark_standard`` 先在每条序列上计算三条定位曲线，再做序列等权
    宏平均；``cognitive_visible_only`` 则把全部 GT 可见帧拼接后做微平均，
    仅用于认知定位诊断。两者不能混作同一实验主指标。
    """

    if not 0.0 <= recovery_iou_threshold <= 1.0:
        raise ValueError("recovery_iou_threshold 必须位于 [0, 1]")
    if not any(frame.gt_presence in {"present", "absent"} for frame in frames):
        raise ValueError(
            "输入 JSONL 不含 ground_truth/gt_target_presence，无法执行有监督评测。"
            "请在 runner 写结果时逐帧保存 GT bbox 与 present/absent 标签。"
        )
    grouped = _group_sequences(frames)
    sequence_metrics = {
        name: _evaluate_one_sequence(
            sequence_frames,
            recovery_iou_threshold=recovery_iou_threshold,
        )
        for name, sequence_frames in grouped.items()
    }

    # ``frames`` 已是完整帧列表；避免百万帧 benchmark 再复制一份引用数组。
    all_frames = frames

    # 主指标：移植的 pytracking extract_results，可直接与 SUTrack MODEL_ZOO 比较。
    pytracking_results = extract_results_from_canonical_frames(all_frames)
    pytracking_summary = summarize_pytracking_curves(pytracking_results)

    benchmark_standard = aggregate_benchmark_standard(
        metrics["benchmark_standard"] for metrics in sequence_metrics.values()
    )
    cognitive_visible_only = evaluate_cognitive_visible_only(all_frames)
    return {
        "schema_version": "cogtrack.eval.v3",
        "num_sequences": len(grouped),
        "num_frames": len(all_frames),
        "source": {
            "num_files": len(source_files),
            "files": list(source_files),
        },
        # === 主指标：pytracking 口径 ===
        "pytracking": pytracking_summary,
        "pytracking_raw": pytracking_results,
        # === 以下为认知能力诊断，不是 benchmark 主指标，不参与对外比较 ===
        "cognitive_diagnostics": {
            "_note": (
                "以下指标为 CognitiveTrack 自定义诊断口径，用于分析 absent 判断、"
                "身份判别与重捕获能力。不可与 SUTrack/pytracking 发表数字直接比较。"
            ),
            "benchmark_standard": benchmark_standard,
            "cognitive_visible_only": cognitive_visible_only,
            "presence": evaluate_presence(all_frames),
            "identity": evaluate_identity(all_frames),
            "execution": evaluate_execution(all_frames),
            "reappearance": _aggregate_reappearance(
                sequence_metrics,
                recovery_iou_threshold=recovery_iou_threshold,
            ),
        },
        "sequences": sequence_metrics,
    }


def evaluate_jsonl_files(
    paths: Iterable[str | Path],
    *,
    recovery_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """从一个或多个 JSONL 文件读取并执行完整评测。"""

    normalized_paths = [str(Path(path).resolve()) for path in paths]
    if not normalized_paths:
        raise ValueError("至少需要一个 JSONL 输入文件")
    frames = load_frame_records(normalized_paths)
    if not frames:
        raise ValueError("JSONL 输入中没有有效帧记录")
    return evaluate_frames(
        frames,
        source_files=normalized_paths,
        recovery_iou_threshold=recovery_iou_threshold,
    )


def _flatten_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套指标字典展开为 CSV 列；列表等非标量留在 JSON 中。"""

    output: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            output.update(_flatten_scalars(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = item
    return output


def _write_sequence_csv(path: Path, sequences: Mapping[str, Mapping[str, Any]]) -> None:
    rows = []
    field_names = {"sequence", "num_frames"}
    for sequence, metrics in sequences.items():
        row = {"sequence": sequence}
        row.update(_flatten_scalars(metrics))
        rows.append(row)
        field_names.update(row)
    ordered_fields = ["sequence", "num_frames"] + sorted(
        field for field in field_names if field not in {"sequence", "num_frames"}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def write_evaluation_outputs(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """写出 ``summary.json``、逐序列 CSV 和中文 Markdown 报告。"""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    csv_path = target / "sequence_metrics.csv"
    report_path = target / "report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    _write_sequence_csv(csv_path, summary.get("sequences", {}))
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    return {
        "summary": summary_path,
        "sequence_metrics": csv_path,
        "report": report_path,
    }
