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


def _sequence(root: Path, name: str, visible: list[bool]) -> Sequence:
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
        dataset="synthetic",
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
