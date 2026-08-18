from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogtrack.training.temporal_sampling import (
    REFERENCE_POLICY_FIXED_ANCHOR,
    SequenceCasePlan,
    TemporalCaseSamplingPlan,
)
from tracking.merge_state_update_sft_data import _merge, _validate_source_labels


def _plan(dataset: str, sequence: str, frames: tuple[int, ...]) -> TemporalCaseSamplingPlan:
    item = SequenceCasePlan(
        dataset=dataset,
        sequence=sequence,
        anchor_frame_id=0,
        frame_ids=frames,
        reference_frame_ids=(0,) * len(frames),
        present_count=len(frames),
        absent_count=0,
        absent_run_count=0,
    )
    return TemporalCaseSamplingPlan(
        seed=1,
        requested_absent_ratio=0.0,
        actual_absent_ratio=0.0,
        max_cases_per_sequence=20,
        sequence_count=1,
        case_count=len(frames),
        present_count=len(frames),
        absent_count=0,
        absent_run_count=0,
        sequences=(item,),
        reference_policy=REFERENCE_POLICY_FIXED_ANCHOR,
    )


def _row(dataset: str, sequence: str, frame_id: int, *, update: str | None) -> dict:
    return {
        "dataset": dataset,
        "sequence": sequence,
        "frame_id": frame_id,
        "target_status": "present",
        "memory_update": update,
        "verified_null": update is None,
        "source": "test",
        "reviewed": dataset != "mgit",
        "input_state": "the initial target identity",
    }


def test_merge_recomputes_plan_from_labelled_frames() -> None:
    mgit_plan = _plan("mgit", "m", (1, 2, 3))
    teacher_plan = _plan("lasot", "l", (4, 5, 6))
    mgit_rows = _validate_source_labels(
        mgit_plan,
        [_row("mgit", "m", 1, update="a new state"), _row("mgit", "m", 3, update=None)],
        source_name="MGIT",
    )
    teacher_rows = _validate_source_labels(
        teacher_plan,
        [_row("lasot", "l", 5, update="another state")],
        source_name="teacher",
    )
    merged, rows, report = _merge(
        mgit_plan,
        mgit_rows,
        teacher_plan,
        teacher_rows,
        seed=9,
        max_cases_per_sequence=10000,
    )
    assert merged.seed == 9
    assert merged.case_count == 3
    assert [(item.dataset, item.frame_ids) for item in merged.sequences] == [
        ("lasot", (5,)),
        ("mgit", (1, 3)),
    ]
    assert [(row["dataset"], row["frame_id"]) for row in rows] == [
        ("lasot", 5),
        ("mgit", 1),
        ("mgit", 3),
    ]
    assert report["merged"]["updates"] == 2
    assert report["merged"]["hard_nulls"] == 1


def test_merge_rejects_unlabelled_null_and_out_of_plan() -> None:
    plan = _plan("mgit", "m", (1,))
    with pytest.raises(ValueError, match="verified_null"):
        _validate_source_labels(
            plan,
            [_row("mgit", "m", 1, update=None) | {"verified_null": False}],
            source_name="MGIT",
        )
    with pytest.raises(ValueError, match="不在输入 plan"):
        _validate_source_labels(
            plan,
            [_row("mgit", "m", 2, update="state")],
            source_name="MGIT",
        )
    with pytest.raises(ValueError, match="reviewed=true"):
        _validate_source_labels(
            _plan("lasot", "l", (1,)),
            [_row("lasot", "l", 1, update="state") | {"reviewed": False}],
            source_name="teacher",
        )


def test_teacher_report_requires_independent_non_dry_run(tmp_path: Path) -> None:
    from tracking.merge_state_update_sft_data import _teacher_report_verified

    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "independently_verified": True,
                "dry_run": False,
                "minimum_output_reached": True,
                "min_output_labels": 1200,
                "written_labels": 1400,
            }
        )
    )
    assert _teacher_report_verified(path)
    path.write_text(
        json.dumps(
            {
                "independently_verified": True,
                "dry_run": True,
                "minimum_output_reached": True,
                "min_output_labels": 1200,
                "written_labels": 1400,
            }
        )
    )
    assert not _teacher_report_verified(path)


def test_teacher_report_accepts_single_pass_frontier_api_quality_gate(tmp_path: Path) -> None:
    from tracking.merge_state_update_sft_data import _teacher_report_verified

    path = tmp_path / "api_report.json"
    path.write_text(
        json.dumps(
            {
                "annotation_policy": "single_pass_frontier_api_v1",
                "single_pass_frontier_teacher": True,
                "quality_gate_applied": True,
                "teacher_model": "qwen3.6-closed",
                "prompt_version": "2.0.0",
                "independently_verified": False,
                "dry_run": False,
                "minimum_output_reached": True,
                "min_output_labels": 1200,
                "written_labels": 1500,
            }
        )
    )
    assert _teacher_report_verified(path)
