#!/usr/bin/env python3
"""从 MGIT 全部可靠 action 分段生成 state_update_sft 计划与标签。

每个真实文本变化段尽量保留一个可见更新帧；每段再抽取少量远离边界的稳定可见帧作为
hard-null。真实 absent 只提供 hard-null，绝不会推进输入侧状态快照。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.loss_mask import (  # noqa: E402
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_HARD_NULL,
    MEMORY_STATE_VERIFIED_UPDATE,
)
from cogtrack.training.mgit_state_labels import (  # noqa: E402
    MGIT_HARD_NULL_SOURCE,
    MGIT_LABEL_SOURCE,
    SegmentParseReport,
    build_frame_memory_labels,
    load_action_segments,
    state_text_at,
)
from cogtrack.training.temporal_sampling import (  # noqa: E402
    REFERENCE_POLICY_FIXED_ANCHOR,
    SequenceCasePlan,
    TemporalCaseSamplingPlan,
)
from pytracking.datasets import iter_dataset  # noqa: E402
from pytracking.datasets.mgit import load_split_definition  # noqa: E402
from pytracking.evaluation.data import Sequence  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402


def _valid_bbox(sequence: Sequence, frame_id: int) -> bool:
    values = np.asarray(sequence.ground_truth_rect[frame_id], dtype=np.float64).reshape(-1)
    return bool(
        values.size == 4
        and np.all(np.isfinite(values))
        and values[2] > 0
        and values[3] > 0
    )


def _present(sequence: Sequence, frame_id: int) -> bool:
    visible = (
        bool(sequence.target_visible[frame_id])
        if sequence.target_visible is not None
        else True
    )
    return visible and _valid_bbox(sequence, frame_id)


def _absent_frames(sequence: Sequence) -> frozenset[int]:
    if sequence.target_visible is None:
        raise ValueError(f"MGIT {sequence.name} 缺少 absent 标注")
    return frozenset(
        frame_id
        for frame_id in range(len(sequence))
        if not bool(sequence.target_visible[frame_id])
    )


def _contiguous_runs(frame_ids: Iterable[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for frame_id in sorted(frame_ids):
        if not runs or frame_id != runs[-1][-1] + 1:
            runs.append([frame_id])
        else:
            runs[-1].append(frame_id)
    return runs


def _uniform(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, num=count)
    return [values[int(round(index))] for index in indices]


def _distance_to_boundaries(frame_id: int, boundaries: set[int]) -> int:
    if not boundaries:
        return 1 << 30
    return min(abs(frame_id - boundary) for boundary in boundaries)


def _mine_sequence(
    sequence: Sequence,
    *,
    boundary_margin: int,
    hard_null_per_segment: int,
    absent_per_run: int,
) -> tuple[SequenceCasePlan | None, list[dict[str, Any]], dict[str, Any]]:
    description_path = Path(str(sequence.metadata.get("description_path") or ""))
    parse_report = SegmentParseReport(sequence=sequence.name)
    segments = load_action_segments(description_path, report=parse_report)
    anchor_frame_id = next(
        (frame_id for frame_id in range(len(sequence)) if _present(sequence, frame_id)),
        None,
    )
    if anchor_frame_id is None or not segments:
        return None, [], {**parse_report.as_dict(), "reason": "no_anchor_or_segments"}

    absent = _absent_frames(sequence)
    boundaries = {
        value
        for segment in segments
        for value in (segment.start_frame, segment.end_frame + 1)
        if value > anchor_frame_id
    }
    candidates: set[int] = set()
    update_probe_count = 0
    stable_probe_count = 0
    for segment in segments:
        start = max(anchor_frame_id + 1, segment.start_frame)
        end = min(len(sequence) - 1, segment.end_frame)
        if start > end:
            continue
        present_ids = []
        for frame_id in range(start, end + 1):
            if not _present(sequence, frame_id):
                continue
            text, ambiguous = state_text_at(segments, frame_id)
            if not ambiguous and text == segment.description:
                present_ids.append(frame_id)
        if not present_ids:
            continue
        # 该探针使每个可靠新分段至少有一次机会产出 verified_update。
        candidates.add(present_ids[0])
        update_probe_count += 1
        stable_ids = [
            frame_id
            for frame_id in present_ids
            if _distance_to_boundaries(frame_id, boundaries) >= boundary_margin
        ]
        chosen_stable = _uniform(stable_ids, hard_null_per_segment)
        candidates.update(chosen_stable)
        stable_probe_count += len(chosen_stable)

    absent_candidates: set[int] = set()
    for run in _contiguous_runs(
        frame_id for frame_id in absent if frame_id > anchor_frame_id
    ):
        chosen = _uniform(run, absent_per_run)
        absent_candidates.update(chosen)
    candidates.update(absent_candidates)
    frame_plan = sorted(candidates)
    labels = build_frame_memory_labels(
        segments,
        frame_plan,
        absent_frames=absent,
        initial_state=segments[0].description,
        boundary_margin=boundary_margin,
        within_segment_hard_null=True,
        report=parse_report,
    )
    selected_ids = [
        frame_id
        for frame_id in frame_plan
        if labels[frame_id].state
        in {MEMORY_STATE_VERIFIED_UPDATE, MEMORY_STATE_VERIFIED_HARD_NULL}
    ]
    if not selected_ids:
        return None, [], {**parse_report.as_dict(), "reason": "no_verified_labels"}

    rows: list[dict[str, Any]] = []
    for frame_id in selected_ids:
        label = labels[frame_id]
        is_update = label.state == MEMORY_STATE_VERIFIED_UPDATE
        rows.append(
            {
                "dataset": "mgit",
                "sequence": sequence.name,
                "frame_id": frame_id,
                "target_status": "present" if _present(sequence, frame_id) else "absent",
                "memory_update": label.memory_update if is_update else None,
                "verified_null": not is_update,
                "source": MGIT_LABEL_SOURCE if is_update else MGIT_HARD_NULL_SOURCE,
                "reviewed": False,
                "input_state": label.input_state,
                "reason": label.reason,
                "annotation_origin": "official_mgit_action_segment",
            }
        )

    present_count = sum(_present(sequence, frame_id) for frame_id in selected_ids)
    absent_count = len(selected_ids) - present_count
    selected_absent_runs = _contiguous_runs(
        frame_id for frame_id in selected_ids if frame_id in absent
    )
    plan = SequenceCasePlan(
        dataset="mgit",
        sequence=sequence.name,
        anchor_frame_id=anchor_frame_id,
        frame_ids=tuple(selected_ids),
        reference_frame_ids=(anchor_frame_id,) * len(selected_ids),
        present_count=present_count,
        absent_count=absent_count,
        absent_run_count=len(selected_absent_runs),
    )
    sequence_report = {
        **parse_report.as_dict(),
        "update_probes": update_probe_count,
        "stable_probes": stable_probe_count,
        "absent_probes": len(absent_candidates),
        "selected_cases": len(selected_ids),
        "verified_updates": sum(row["memory_update"] is not None for row in rows),
        "verified_hard_nulls": sum(row["verified_null"] for row in rows),
        "masked_probes_dropped": sum(
            labels[frame_id].state == MEMORY_STATE_MASKED_UNKNOWN
            for frame_id in frame_plan
        ),
    }
    return plan, rows, sequence_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--env-config")
    parser.add_argument("--mgit-version", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--boundary-margin", type=int, default=30)
    parser.add_argument("--hard-null-per-segment", type=int, default=1)
    parser.add_argument(
        "--absent-per-run",
        type=int,
        default=0,
        help=(
            "每个真实缺失区间额外取多少 hard-null；默认 0，因为 tracking_sft 已经"
            "覆盖 absent，本计划优先保留 MGIT 文本更新与 present hard-null。"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--max-cases-per-sequence",
        type=int,
        default=10000,
        help="写入 plan 的重放上限；state_update 计划本身按可靠分段数量决定实际 case 数。",
    )
    parser.add_argument("--limit-sequences", type=int)
    args = parser.parse_args()
    if args.boundary_margin < 0:
        raise SystemExit("--boundary-margin 不能为负数")
    if args.hard_null_per_segment < 0 or args.absent_per_run < 0:
        raise SystemExit("hard-null/absent 每段数量不能为负数")
    if args.max_cases_per_sequence <= 0:
        raise SystemExit("--max-cases-per-sequence 必须为正数")

    environment = load_environment(args.env_config)
    mgit_root = environment.dataset_root("mgit")
    frames_root = mgit_root / "data" / "train"
    official_names = load_split_definition(args.mgit_version, "train")
    usable_names = [
        name
        for name in official_names
        if (frames_root / name / f"frame_{name}").is_dir()
        and any((frames_root / name / f"frame_{name}").iterdir())
    ]
    if args.limit_sequences is not None:
        usable_names = usable_names[: args.limit_sequences]

    plans: list[SequenceCasePlan] = []
    label_rows: list[dict[str, Any]] = []
    sequence_reports: list[dict[str, Any]] = []
    sequences = iter_dataset(
        "mgit",
        environment=environment,
        split="train",
        version=args.mgit_version,
        sequence_names=usable_names,
    )
    for sequence in sequences:
        plan, rows, report = _mine_sequence(
            sequence,
            boundary_margin=args.boundary_margin,
            hard_null_per_segment=args.hard_null_per_segment,
            absent_per_run=args.absent_per_run,
        )
        sequence_reports.append(report)
        if plan is not None:
            plans.append(plan)
            label_rows.extend(rows)
    if not plans or not label_rows:
        raise SystemExit("没有挖到可用的 MGIT state_update_sft 标签")

    present_count = sum(plan.present_count for plan in plans)
    absent_count = sum(plan.absent_count for plan in plans)
    case_count = present_count + absent_count
    sampling_plan = TemporalCaseSamplingPlan(
        seed=args.seed,
        requested_absent_ratio=absent_count / case_count,
        actual_absent_ratio=absent_count / case_count,
        max_cases_per_sequence=args.max_cases_per_sequence,
        sequence_count=len(plans),
        case_count=case_count,
        present_count=present_count,
        absent_count=absent_count,
        absent_run_count=sum(plan.absent_run_count for plan in plans),
        sequences=tuple(plans),
        reference_policy=REFERENCE_POLICY_FIXED_ANCHOR,
    )

    plan_path = Path(args.output_plan).expanduser().resolve()
    labels_path = Path(args.output_labels).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    for path in (plan_path, labels_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(sampling_plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in label_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    states = Counter(
        "verified_update" if row["memory_update"] is not None else "verified_hard_null"
        for row in label_rows
    )
    summary = {
        "schema_version": "cogtrack.mgit_state_update_plan.v1",
        "annotation_source": "official_mgit_action_segments",
        "official_tiny_train_names": len(official_names),
        "usable_sequences": len(usable_names),
        "planned_sequences": len(plans),
        "planned_cases": case_count,
        "state_distribution": dict(sorted(states.items())),
        "boundary_margin": args.boundary_margin,
        "hard_null_per_segment": args.hard_null_per_segment,
        "absent_per_run": args.absent_per_run,
        "sampling_plan": str(plan_path),
        "state_update_labels": str(labels_path),
    }
    report_path.write_text(
        json.dumps(
            {"summary": summary, "sequences": sequence_reports},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
