from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogtrack.training.temporal_sampling import (
    REFERENCE_POLICY_FIXED_ANCHOR,
    TemporalCaseSamplingPlan,
    plan_temporal_presence_cases,
)
from pytracking.evaluation.data import Sequence


def _sequence(
    root: Path, name: str, visible: list[bool], *, dataset: str = "synthetic"
) -> Sequence:
    sequence_root = root / name
    sequence_root.mkdir(parents=True)
    frames = []
    boxes = []
    for frame_id, is_visible in enumerate(visible):
        frame_path = sequence_root / f"{frame_id:04d}.jpg"
        assert cv2.imwrite(str(frame_path), np.full((32, 48, 3), frame_id, dtype=np.uint8))
        frames.append(str(frame_path))
        boxes.append([5, 6, 12, 10] if is_visible else [0, 0, 0, 0])
    return Sequence(
        name=name,
        frames=frames,
        dataset=dataset,
        ground_truth_rect=boxes,
        target_visible=visible,
        metadata={"split": "train"},
    )


def test_global_plan_keeps_seven_to_three_using_same_sequence_absent_frames(tmp_path: Path) -> None:
    sequences = [
        _sequence(tmp_path, "mixed-1", [True, True, True, False, False, True, True, True]),
        _sequence(tmp_path, "positive", [True] * 8),
        _sequence(tmp_path, "mixed-2", [True, False, False, False, True, True, True, True]),
        _sequence(tmp_path, "mixed-3", [True, True, False, False, False, True, True, True]),
    ]

    first = plan_temporal_presence_cases(
        sequences,
        max_cases_per_sequence=5,
        absent_ratio=0.3,
        seed=11,
    )
    second = plan_temporal_presence_cases(
        sequences,
        max_cases_per_sequence=5,
        absent_ratio=0.3,
        seed=11,
    )

    assert first == second
    assert first.case_count == 20
    assert first.present_count == 14
    assert first.absent_count == 6
    assert first.actual_absent_ratio == pytest.approx(0.3)

    by_name = {sequence.name: sequence for sequence in sequences}
    for item in first.sequences:
        sequence = by_name[item.sequence]
        selected_absent = [frame_id for frame_id in item.frame_ids if not sequence.target_visible[frame_id]]
        assert len(selected_absent) == item.absent_count
        assert all(frame_id > 0 for frame_id in item.frame_ids)
        assert len(item.reference_frame_ids) == len(item.frame_ids)
        assert all(
            reference < current
            for reference, current in zip(item.reference_frame_ids, item.frame_ids, strict=True)
        )
        assert all(sequence.target_visible[reference] for reference in item.reference_frame_ids)


def test_plan_rejects_unreachable_absent_ratio(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path, "positive", [True] * 8)
    with pytest.raises(ValueError, match="无法达到 absent_ratio"):
        plan_temporal_presence_cases(
            [sequence],
            max_cases_per_sequence=5,
            absent_ratio=0.3,
        )


def test_plan_reuses_real_absent_current_with_distinct_earlier_references(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path, "one-absent", [True, True, True, False])
    plan = plan_temporal_presence_cases(
        [sequence],
        max_cases_per_sequence=3,
        absent_ratio=2 / 3,
        seed=23,
    )

    item = plan.sequences[0]
    pairs = list(zip(item.reference_frame_ids, item.frame_ids, strict=True))
    assert item.frame_ids.count(3) == 2
    assert len(set(pairs)) == len(pairs)
    assert all(reference < current for reference, current in pairs)


def test_fixed_anchor_plan_never_reuses_current_or_changes_identity_anchor(tmp_path: Path) -> None:
    sequences = [
        _sequence(tmp_path, "mixed-1", [True, True, True, False, False, True, True, True]),
        _sequence(tmp_path, "positive", [True] * 8),
        _sequence(tmp_path, "mixed-2", [True, False, False, False, True, True, True, True]),
        _sequence(tmp_path, "mixed-3", [True, True, False, False, False, True, True, True]),
    ]

    plan = plan_temporal_presence_cases(
        sequences,
        max_cases_per_sequence=5,
        absent_ratio=0.3,
        seed=11,
        reference_policy=REFERENCE_POLICY_FIXED_ANCHOR,
    )

    assert plan.reference_policy == REFERENCE_POLICY_FIXED_ANCHOR
    assert plan.actual_absent_ratio == pytest.approx(0.3)
    for item in plan.sequences:
        assert len(set(item.frame_ids)) == len(item.frame_ids)
        assert set(item.reference_frame_ids) == {item.anchor_frame_id}


def test_plan_uses_first_present_as_anchor_and_ignores_leading_absence(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path, "late-init", [False, False, True, True, False])

    plan = plan_temporal_presence_cases(
        [sequence],
        max_cases_per_sequence=2,
        absent_ratio=0.5,
    )

    item = plan.sequences[0]
    assert item.anchor_frame_id == 2
    assert item.frame_ids == (3, 4)
    assert item.present_count == 1
    assert item.absent_count == 1


def test_sampling_plan_json_round_trip_is_strict(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path, "mixed", [True, True, False, True, False, True])
    original = plan_temporal_presence_cases(
        [sequence],
        max_cases_per_sequence=4,
        absent_ratio=0.5,
        seed=17,
    )

    restored = TemporalCaseSamplingPlan.from_dict(original.to_dict())

    assert restored == original

    broken = original.to_dict()
    broken["case_count"] += 1
    with pytest.raises(ValueError, match="case_count"):
        TemporalCaseSamplingPlan.from_dict(broken)


def test_per_dataset_cap_only_lifts_the_named_source(tmp_path: Path) -> None:
    """长序列源抬上限不能顺带把短序列源也抬上去。

    MGIT 是 ``verified_update`` 的唯一来源且单条上万帧，必须单独抬；lasot 只有几百帧，
    跟着抬只会把同一条短视频重复采样。
    """

    visible = [True, True, True, False, False, True, True, True, True, True]
    sequences = [
        _sequence(tmp_path, "long-1", visible * 4, dataset="mgit"),
        _sequence(tmp_path, "short-1", visible, dataset="lasot"),
    ]

    plan = plan_temporal_presence_cases(
        sequences,
        max_cases_per_sequence=4,
        max_cases_by_dataset={"mgit": 20},
        absent_ratio=0.5,
        seed=23,
    )

    by_dataset = {item.dataset: item for item in plan.sequences}
    assert len(by_dataset["mgit"].frame_ids) == 20
    assert len(by_dataset["lasot"].frame_ids) == 4
    assert plan.resolved_max_cases_by_dataset == {"mgit": 20}
    # 分数据源上限必须能跨 JSON 往返，否则重放会静默产出另一份数据。
    assert TemporalCaseSamplingPlan.from_dict(plan.to_dict()) == plan


def test_plan_omits_empty_per_dataset_caps_from_json(tmp_path: Path) -> None:
    """空 caps 不写入 JSON，旧 plan 的 checksum 保持字节兼容。"""

    sequence = _sequence(tmp_path, "mixed", [True, True, False, True, False, True])
    plan = plan_temporal_presence_cases(
        [sequence], max_cases_per_sequence=4, absent_ratio=0.5, seed=17
    )

    assert "max_cases_by_dataset" not in plan.to_dict()


@pytest.mark.parametrize(
    "caps, message",
    [
        ({"mgit": 0}, "必须是正整数"),
        ({"mgit": -3}, "必须是正整数"),
        ({"  ": 5}, "不能为空"),
    ],
)
def test_plan_rejects_invalid_per_dataset_caps(
    tmp_path: Path, caps: dict[str, int], message: str
) -> None:
    sequence = _sequence(tmp_path, "mixed", [True, True, False, True, False, True])
    with pytest.raises(ValueError, match=message):
        plan_temporal_presence_cases(
            [sequence],
            max_cases_per_sequence=4,
            max_cases_by_dataset=caps,
            absent_ratio=0.5,
            seed=17,
        )


def test_plan_rejects_unsorted_per_dataset_caps(tmp_path: Path) -> None:
    """字段顺序决定 JSON 字节，未排序就拒绝，避免同一计划算出两个 checksum。"""

    sequence = _sequence(tmp_path, "mixed", [True, True, False, True, False, True])
    plan = plan_temporal_presence_cases(
        [sequence], max_cases_per_sequence=4, absent_ratio=0.5, seed=17
    )
    with pytest.raises(ValueError, match="升序"):
        replace(plan, max_cases_by_dataset=(("tnl2k", 5), ("mgit", 9)))


def test_sampled_prior_policy_spreads_reference_frames(tmp_path: Path) -> None:
    """默认策略下模板帧不能永远是第 0 帧。

    固定锚点让全部 case 的 Image 1 都是序列首帧，模型只见过"从视频开头初始化"这一种
    情形；随机模板把 (reference, current) 间隔摊成分布，且始终严格早于 current。
    """

    sequence = _sequence(tmp_path, "long", [True] * 40)

    sampled = plan_temporal_presence_cases(
        [sequence], max_cases_per_sequence=12, absent_ratio=0.0, seed=29
    )
    fixed = plan_temporal_presence_cases(
        [sequence],
        max_cases_per_sequence=12,
        absent_ratio=0.0,
        seed=29,
        reference_policy=REFERENCE_POLICY_FIXED_ANCHOR,
    )

    assert set(fixed.sequences[0].reference_frame_ids) == {0}
    sampled_refs = sampled.sequences[0].reference_frame_ids
    assert len(set(sampled_refs)) > 1
    assert all(
        reference < current
        for reference, current in zip(sampled_refs, sampled.sequences[0].frame_ids, strict=True)
    )
