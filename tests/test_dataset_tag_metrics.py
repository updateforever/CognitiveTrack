"""``dataset`` 标签必须原样传给 pytracking 误差计算，不能从序列名猜。

原版 ``calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible)`` 拿
``seq.dataset`` 选分支，其中 lasot 那条：

    err_center_normalized[~target_visible] = Inf
    err_center[~target_visible] = Inf

决定了不可见帧算不算命中。通用分支只做 ``err_center_normalized[~valid] = -1.0``，
而 ``-1.0 <= threshold`` 对 0..0.5 的全部阈值都成立，于是不可见帧会被算成
Pnorm 命中。lasot 序列名形如 ``airplane-1``，一旦靠序列名去猜 dataset 就会
落到通用分支，Pnorm 被系统性抬高。这里锁住正确行为。
"""

from __future__ import annotations

import pytest

from cogtrack.evaluation import evaluate_frames
from cogtrack.evaluation.io import CanonicalFrame, canonicalize_record


def _frame(
    sequence: str,
    frame_id: int,
    *,
    dataset: str,
    gt_presence: str,
    gt_bbox: tuple[float, float, float, float] | None,
) -> CanonicalFrame:
    # 预测恒定放在原点附近，远离真实目标，保证不可见帧不会"恰好"命中。
    return CanonicalFrame(
        sequence=sequence,
        dataset=dataset,
        frame_id=frame_id,
        gt_bbox=gt_bbox,
        pred_bbox=(0.0, 0.0, 10.0, 10.0),
        gt_presence=gt_presence,
        pred_presence="present",
        gt_identity=None,
        pred_identity=None,
        execution_status="ok",
        is_observation_frame=True,
    )


def _sequence(dataset: str) -> list[CanonicalFrame]:
    """一条 4 帧序列：首帧可见，随后 3 帧目标不可见。

    不可见帧的 GT bbox 给全零，对应 pytracking 里 absent 帧的常见标注方式，
    这样 ``valid`` 同时被 bbox 面积和 target_visible 判掉。
    """

    frames = [
        _frame(
            "airplane-1",
            0,
            dataset=dataset,
            gt_presence="present",
            gt_bbox=(100.0, 100.0, 20.0, 20.0),
        )
    ]
    frames += [
        _frame(
            "airplane-1",
            index,
            dataset=dataset,
            gt_presence="absent",
            gt_bbox=(0.0, 0.0, 0.0, 0.0),
        )
        for index in (1, 2, 3)
    ]
    return frames


def test_lasot_marks_invisible_frames_as_norm_precision_miss() -> None:
    summary = evaluate_frames(_sequence("lasot"))
    # 首帧预测被强制替换成 GT，必然命中；其余 3 帧不可见，lasot 分支置 Inf，
    # 一律不命中。所以 Pnorm = 1/4。
    assert summary["pytracking"]["norm_precision_np20"] == pytest.approx(0.25)


def test_generic_dataset_keeps_original_minus_one_behaviour() -> None:
    summary = evaluate_frames(_sequence("default"))
    # 通用分支把不可见帧的 err_center_norm 置 -1.0，而 -1.0 <= 阈值恒成立，
    # 于是 4 帧全部被算成命中。这是 pytracking 原版行为，移植时必须保留，
    # 不能"顺手修好"，否则非 lasot 数据集的数字会偏离已发表结果。
    assert summary["pytracking"]["norm_precision_np20"] == pytest.approx(1.0)


def test_lasot_and_generic_actually_differ() -> None:
    """两条分支必须给出不同数字，否则说明 dataset 标签根本没生效。"""

    lasot = evaluate_frames(_sequence("lasot"))["pytracking"]["norm_precision_np20"]
    generic = evaluate_frames(_sequence("default"))["pytracking"]["norm_precision_np20"]
    assert lasot != pytest.approx(generic)


def test_dataset_tag_is_parsed_from_jsonl_record() -> None:
    """runner 写的 ``dataset`` 字段要被 parser 原样带进 CanonicalFrame。"""

    frame = canonicalize_record(
        {
            "sequence": "airplane-1",
            "dataset": "lasot",
            "frame_id": 7,
            "target_bbox": [1.0, 2.0, 3.0, 4.0],
            "execution": {"status": "ok"},
            "ground_truth": {"target_presence": "present", "bbox_xywh": [1.0, 2.0, 3.0, 4.0]},
        },
        source_line=1,
        default_sequence="fallback",
    )
    assert frame.dataset == "lasot"


def test_missing_dataset_falls_back_to_generic_branch() -> None:
    """老结果文件没有 ``dataset`` 字段时退回 "default"，而不是崩掉。"""

    frame = canonicalize_record(
        {
            "sequence": "airplane-1",
            "frame_id": 7,
            "target_bbox": [1.0, 2.0, 3.0, 4.0],
            "execution": {"status": "ok"},
            "ground_truth": {"target_presence": "present", "bbox_xywh": [1.0, 2.0, 3.0, 4.0]},
        },
        source_line=1,
        default_sequence="fallback",
    )
    assert frame.dataset == "default"
