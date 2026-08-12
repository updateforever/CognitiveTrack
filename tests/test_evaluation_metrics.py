import json
from pathlib import Path

import pytest

from cogtrack.evaluation import evaluate_frames, evaluate_jsonl_files
from cogtrack.evaluation.io import CanonicalFrame


def _frame(
    sequence: str,
    frame_id: int,
    *,
    dataset: str = "default",
    gt_presence: str = "present",
    pred_presence: str = "present",
    gt_bbox: tuple[float, float, float, float] | None = (0.0, 0.0, 10.0, 10.0),
    pred_bbox: tuple[float, float, float, float] | None = (0.0, 0.0, 10.0, 10.0),
    is_observation_frame: bool = True,
) -> CanonicalFrame:
    return CanonicalFrame(
        sequence=sequence,
        dataset=dataset,
        frame_id=frame_id,
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        gt_presence=gt_presence,
        pred_presence=pred_presence,
        gt_identity=None,
        pred_identity=None,
        execution_status="ok",
        is_observation_frame=is_observation_frame,
    )


def test_sparse_metrics_report_hold_last_and_observation_only_separately():
    frames = [
        _frame("sparse", 0),
        _frame(
            "sparse",
            1,
            pred_bbox=None,
            pred_presence=None,
            is_observation_frame=False,
        ),
        _frame("sparse", 2, pred_bbox=(100.0, 100.0, 10.0, 10.0)),
    ]

    summary = evaluate_frames(frames)

    # dense-zero 把未观测帧放在全序列分母中记零；hold-last 用第 0 帧合法框
    # 填充第 1 帧；observation-only 则只计算第 0、2 两个实际观测帧。
    assert summary["pytracking"]["precision_p20"] == pytest.approx(1.0 / 3.0)
    assert summary["pytracking"]["norm_precision_np20"] == pytest.approx(1.0 / 3.0)
    sparse = summary["pytracking_sparse"]
    assert sparse["hold_last"]["precision_p20"] == pytest.approx(2.0 / 3.0)
    assert sparse["hold_last"]["norm_precision_np20"] == pytest.approx(2.0 / 3.0)
    assert sparse["observation_only"]["precision_p20"] == pytest.approx(1.0 / 2.0)
    assert sparse["observation_only"]["norm_precision_np20"] == pytest.approx(1.0 / 2.0)
    for values in sparse.values():
        assert values["sparsity"]["total_frames"] == 3
        assert values["sparsity"]["observation_frames"] == 2
        assert values["sparsity"]["observation_rate"] == pytest.approx(2.0 / 3.0)


def test_observation_only_excludes_non_vlm_hybrid_predictions():
    frames = [
        _frame("hybrid", 0),
        _frame(
            "hybrid",
            1,
            pred_bbox=(0.0, 0.0, 10.0, 10.0),
            is_observation_frame=False,
        ),
        _frame("hybrid", 2, pred_bbox=(100.0, 100.0, 10.0, 10.0)),
    ]

    summary = evaluate_frames(frames)

    # 非观察帧虽然有 SUTrack/Hybrid 发布框，也不能进入 VLM observation-only 分母。
    observation_only = summary["pytracking_sparse"]["observation_only"]
    assert observation_only["precision_p20"] == pytest.approx(1.0 / 2.0)
    assert observation_only["sparsity"]["observation_frames"] == 2


def test_benchmark_uses_sequence_macro_average_not_frame_micro_average():
    # 短序列全部正确，长序列全部错误。按帧微平均是 1/10，按序列宏平均
    # 是 (1 + 0) / 2；该构造可防止后续误把全部帧直接拼接评测。
    frames = [_frame("short", 0)]
    frames.extend(
        _frame("long", index, pred_bbox=(100.0, 100.0, 10.0, 10.0))
        for index in range(9)
    )

    summary = evaluate_frames(frames)
    # v3 起认知诊断统一下沉到 cognitive_diagnostics，主指标是 pytracking 口径。
    diagnostics = summary["cognitive_diagnostics"]
    benchmark = diagnostics["benchmark_standard"]
    visible_only = diagnostics["cognitive_visible_only"]

    assert benchmark["aggregation"] == "sequence_macro_average"
    assert benchmark["precision_at_20"] == pytest.approx(0.5)
    assert benchmark["normalized_precision_at_0_2"] == pytest.approx(0.5)
    assert benchmark["success_auc"] == pytest.approx(10.0 / 21.0)
    assert visible_only["aggregation"] == "frame_micro_average"
    assert visible_only["precision_at_20"] == pytest.approx(0.1)
    assert visible_only["normalized_precision_at_0_2"] == pytest.approx(0.1)
    assert visible_only["success_auc"] == pytest.approx(2.0 / 21.0)

    # 顶层曲线确实是两个单序列曲线逐点平均，而非从总帧数重算。
    short_curve = summary["sequences"]["short"]["benchmark_standard"]["precision_curve"]
    long_curve = summary["sequences"]["long"]["benchmark_standard"]["precision_curve"]
    expected_curve = [(short + long) / 2.0 for short, long in zip(short_curve, long_curve, strict=True)]
    assert benchmark["precision_curve"] == expected_curve

    # 主指标（pytracking 口径）同样是按序列宏平均，不能退化成按帧微平均。
    # 注意 pytracking 会把每条序列第 0 帧的预测强制replace成 GT
    # （extract_results.py 的 pred_bb[0, :] = anno_bb[0, :]），所以 long 序列
    # 9 帧里有 1 帧必然命中，宏平均是 (1 + 1/9) / 2 = 5/9，而不是 0.5。
    # 这里锁住该约定，避免后人"修掉"首帧替换而偏离已发表数字。
    main = summary["pytracking"]
    assert main["num_valid_sequences"] == 2
    assert main["precision_p20"] == pytest.approx(5.0 / 9.0)


def test_absent_frames_fail_localization_but_are_scored_by_presence():
    frames = [
        _frame("longterm", 0),
        _frame(
            "longterm",
            1,
            gt_presence="absent",
            pred_presence="absent",
            gt_bbox=None,
            pred_bbox=None,
        ),
    ]

    summary = evaluate_frames(frames)
    diagnostics = summary["cognitive_diagnostics"]
    benchmark = diagnostics["benchmark_standard"]
    visible_only = diagnostics["cognitive_visible_only"]

    assert benchmark["absent_policy"] == "count_as_localization_failure"
    assert benchmark["evaluated_frames"] == 2
    assert benchmark["absent_frames"] == 1
    assert benchmark["precision_at_20"] == pytest.approx(0.5)
    assert benchmark["normalized_precision_at_0_2"] == pytest.approx(0.5)
    assert benchmark["success_auc"] == pytest.approx(10.0 / 21.0)

    # 诊断口径只看可见帧，因此不受 absent 帧影响；presence 则正确奖励拒绝。
    assert visible_only["evaluated_frames"] == 1
    assert visible_only["precision_at_20"] == pytest.approx(1.0)
    assert diagnostics["presence"]["selective_accuracy"] == pytest.approx(1.0)
    assert diagnostics["presence"]["tn"] == 1
    # CognitiveBench absent 帧必须在默认全序列分母中贡献零分，不能让 pytracking
    # 的 normalized-error=-1 哨兵被 ``<= threshold`` 错当成命中。
    assert summary["pytracking"]["precision_p20"] == pytest.approx(0.5)
    assert summary["pytracking"]["norm_precision_np20"] == pytest.approx(0.5)


def test_benchmark_uses_published_bbox_while_cognitive_metric_requires_commit():
    # hybrid 显式回退到 SUTrack 时可发布传统框，但身份状态仍是
    # uncertain。benchmark 必须与 TXT 一致；认知诊断则不应把未确认框当真。
    frames = [_frame("fallback", 0, pred_presence="uncertain")]

    summary = evaluate_frames(frames)
    diagnostics = summary["cognitive_diagnostics"]

    assert diagnostics["benchmark_standard"]["precision_at_20"] == pytest.approx(1.0)
    assert diagnostics["cognitive_visible_only"]["precision_at_20"] == pytest.approx(0.0)
    # 主指标只看已发布的框，因此和 benchmark 一致，不受 uncertain 影响。
    assert summary["pytracking"]["precision_p20"] == pytest.approx(1.0)


def test_legacy_flat_jsonl_still_loads_and_exposes_compatibility_alias(tmp_path: Path):
    # 历史文件没有显式 presence，评测器应继续从有效框推断，而不要求迁移。
    records = [
        {
            "sequence_name": "legacy",
            "frame_num": 0,
            "ground_truth_rect": [2, 3, 8, 8],
            "target_bbox": [2, 3, 8, 8],
        },
        {
            "sequence_name": "legacy",
            "frame_num": 1,
            "ground_truth_rect": [2, 3, 8, 8],
            "target_bbox": [2, 3, 8, 8],
        },
    ]
    input_path = tmp_path / "legacy_frames.jsonl"
    input_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = evaluate_jsonl_files([input_path])

    assert summary["schema_version"] == "cogtrack.eval.v3"
    diagnostics = summary["cognitive_diagnostics"]
    assert diagnostics["benchmark_standard"]["evaluated_frames"] == 2
    assert diagnostics["benchmark_standard"]["precision_at_20"] == pytest.approx(1.0)
    assert diagnostics["cognitive_visible_only"]["evaluated_frames"] == 2
    # 旧键 tracking 只在单序列层保留别名，语义仍是可见帧微平均。
    assert summary["sequences"]["legacy"]["tracking"] == diagnostics["cognitive_visible_only"]
    # 主指标可用。注意 pytracking 官方口径用严格大于比较 21 个阈值（含 1.0），
    # 所以完美预测的 AUC 是 20/21 而不是 1.0；这里刻意锁住这个约定，
    # 防止后人"修正"成 >= 而偏离 SUTrack 已发表数字。
    assert summary["pytracking"]["success_auc"] == pytest.approx(20.0 / 21.0)
    assert summary["pytracking"]["precision_p20"] == pytest.approx(1.0)
