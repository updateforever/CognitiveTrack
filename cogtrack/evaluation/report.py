"""将评测摘要渲染为便于实验归档的 Markdown 报告。"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _number(value: Optional[float], digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Optional[float], digits: int = 2) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.{digits}f}%"


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    """生成中文 Markdown 报告。

    报告只解释可稳定比较的聚合指标，完整机器可读数据保存在
    ``summary.json``，逐序列指标保存在 ``sequence_metrics.csv``。
    """

    # 主指标：pytracking 口径
    pytracking = summary.get("pytracking", {})
    sparse_metrics = summary.get("pytracking_sparse", {})

    # 认知诊断（v3 起收进 cognitive_diagnostics；兼容 v2 扁平结构）
    diagnostics = summary.get("cognitive_diagnostics", summary)
    legacy_tracking = diagnostics.get("tracking", {})
    benchmark = diagnostics.get("benchmark_standard", legacy_tracking)
    visible_only = diagnostics.get("cognitive_visible_only", legacy_tracking)
    presence = diagnostics.get("presence", {})
    identity = diagnostics.get("identity", {})
    execution = diagnostics.get("execution", {})
    recovery = diagnostics.get("reappearance", {})
    source = summary.get("source", {})
    observation_rate = pytracking.get("sparsity", {}).get("observation_rate")
    is_sparse = observation_rate is not None and float(observation_rate) < 1.0

    lines = [
        "# CognitiveTrack 评测报告",
        "",
        "## 实验范围",
        "",
        f"- 序列数：{summary.get('num_sequences', 0)}",
        f"- 帧数：{summary.get('num_frames', 0)}",
        f"- 输入 JSONL：{source.get('num_files', 0)} 个",
        f"- 恢复判定 IoU 阈值：{recovery.get('iou_threshold', 'N/A')}",
        "",
        (
            "## 兼容指标：pytracking dense-zero 口径"
            if is_sparse
            else "## 主指标：pytracking 口径"
        ),
        "",
        (
            "稀疏实验中该数值受 observation rate 限制，只保留用于结果兼容；正式比较见"
            "下一节的 hold-last 与 observation-only。"
            if is_sparse
            else "与 SUTrack / pytracking 官方评测工具链逐值对齐，可直接与 MODEL_ZOO 发表数字比较。"
        ),
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 有效序列 | {pytracking.get('num_valid_sequences', 0)} |",
        f"| Success AUC | {_number(pytracking.get('success_auc'))} |",
        f"| Success OP50 | {_number(pytracking.get('success_op50'))} |",
        f"| Success OP75 | {_number(pytracking.get('success_op75'))} |",
        f"| Precision @ 20 px | {_number(pytracking.get('precision_p20'))} |",
        f"| Normalized Precision @ 0.2 | {_number(pytracking.get('norm_precision_np20'))} |",
        "",
        (
            "口径：IoU 使用离散像素几何；第 0 帧强制替换为 GT；"
            "`target_visible=False` 的帧标为未命中；按序列宏平均。"
        ),
        "",
        "## 稀疏关键帧口径",
        "",
        (
            "纯 VLM 稀疏执行必须同时报告下列两种口径：`hold_last` 衡量任意时刻的"
            "最近状态，`observation_only` 衡量模型实际看图时的能力。不能单独使用"
            "受关键帧率限制的 `dense_zero` 比较不同观察策略。"
        ),
        "",
        "| 口径 | AUC | OP50 | OP75 | P@20 | Pnorm@0.2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            (
                f"| `{name}` | {_number(values.get('success_auc'))} | "
                f"{_number(values.get('success_op50'))} | {_number(values.get('success_op75'))} | "
                f"{_number(values.get('precision_p20'))} | "
                f"{_number(values.get('norm_precision_np20'))} |"
            )
            for name, values in sparse_metrics.items()
        ],
        "",
        "---",
        "",
        "# 认知能力诊断",
        "",
        (
            "以下均为 CognitiveTrack 自定义口径，用于分析 absent 判断、身份判别与重捕获能力。"
            "**不可与 SUTrack/pytracking 发表数字直接比较。**"
        ),
        "",
        "## 认知定位（自定义口径）",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 参与宏平均的序列 | {benchmark.get('evaluated_sequences', summary.get('num_sequences', 0))} |",
        f"| GT 标注帧 | {benchmark.get('evaluated_frames', 0)} |",
        f"| GT present / absent 帧 | {benchmark.get('present_frames', 0)} / {benchmark.get('absent_frames', 0)} |",
        f"| Success AUC | {_number(benchmark.get('success_auc'))} |",
        f"| Success OP50 | {_number(benchmark.get('success_op50'))} |",
        f"| Success OP75 | {_number(benchmark.get('success_op75'))} |",
        f"| Precision @ 20 px | {_number(benchmark.get('precision_at_20'))} |",
        f"| Normalized Precision @ 0.2 | {_number(benchmark.get('normalized_precision_at_0_2'))} |",
        "",
        (
            "与主指标的差异：absent 帧计为定位失败并占分母（主指标中不参与分母），"
            "IoU 使用连续几何（主指标使用离散像素几何）。"
        ),
        "",
        "## 仅可见帧认知定位诊断",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Present GT 有效帧 | {visible_only.get('evaluated_frames', 0)} |",
        f"| Mean IoU | {_number(visible_only.get('mean_iou'))} |",
        f"| Success AUC | {_number(visible_only.get('success_auc'))} |",
        f"| Precision @ 20 px | {_number(visible_only.get('precision_at_20'))} |",
        f"| Normalized Precision @ 0.2 | {_number(visible_only.get('normalized_precision_at_0_2'))} |",
        "",
        "该组指标仅在 GT present 且框有效的帧上做帧级微平均，用于分析可见时的定位能力。",
        "",
        "## 目标存在性",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        (
            f"| TP / FP / TN / FN | {presence.get('tp', 0)} / {presence.get('fp', 0)} / "
            f"{presence.get('tn', 0)} / {presence.get('fn', 0)} |"
        ),
        f"| Precision | {_number(presence.get('precision'))} |",
        f"| Recall | {_number(presence.get('recall'))} |",
        f"| F1 | {_number(presence.get('f1'))} |",
        f"| Absent false-positive rate | {_percent(presence.get('false_positive_rate'))} |",
        f"| Present miss rate | {_percent(presence.get('miss_rate'))} |",
        f"| Uncertain coverage | {_percent(presence.get('uncertain_coverage'))} |",
        f"| Decision coverage | {_percent(presence.get('decision_coverage'))} |",
        f"| Selective accuracy | {_percent(presence.get('selective_accuracy'))} |",
        f"| Unavailable rate | {_percent(presence.get('unavailable_rate'))} |",
        "",
        "`uncertain` 是模型拒绝判断，不作为第三类 GT；错误或缺少输出的帧计入 unavailable。",
        "",
        "## 身份判别",
        "",
    ]
    if identity.get("evaluated_frames", 0):
        lines.extend(
            [
                "| 指标 | 数值 |",
                "| --- | ---: |",
                f"| 有身份 GT 的帧 | {identity.get('evaluated_frames', 0)} |",
                f"| Precision | {_number(identity.get('precision'))} |",
                f"| Recall | {_number(identity.get('recall'))} |",
                f"| F1 | {_number(identity.get('f1'))} |",
                f"| Decision coverage | {_percent(identity.get('decision_coverage'))} |",
                f"| Selective accuracy | {_percent(identity.get('selective_accuracy'))} |",
            ]
        )
    else:
        lines.append("当前输入没有 same/different 身份 GT，因此不报告伪造的身份分类分数。")

    lines.extend(
        [
            "",
            "## 长时重现恢复",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| Reappearance 事件 | {recovery.get('events', 0)} |",
            f"| 成功恢复 / 未恢复 | {recovery.get('recovered_events', 0)} / {recovery.get('unrecovered_events', 0)} |",
            f"| 恢复率 | {_percent(recovery.get('recovery_rate'))} |",
            f"| 平均恢复延迟（帧） | {_number(recovery.get('mean_recovery_delay'), 2)} |",
            f"| 中位恢复延迟（帧） | {_number(recovery.get('median_recovery_delay'), 2)} |",
            "",
            "## 执行状态",
            "",
            f"- 错误帧：{execution.get('error_frames', 0)}（{_percent(execution.get('error_rate'))}）",
            f"- 有 observation 标注的帧：{execution.get('observation_labeled_frames', 0)}",
            f"- Observation rate：{_percent(execution.get('observation_rate'))}",
            "- 各状态计数：",
            "",
        ]
    )
    status_counts = execution.get("status_counts", {})
    if status_counts:
        lines.extend(f"  - `{name}`: {count}" for name, count in sorted(status_counts.items()))
    else:
        lines.append("  - 无记录")
    lines.append("")
    return "\n".join(lines)
