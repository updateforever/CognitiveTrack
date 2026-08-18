#!/usr/bin/env python3
"""从大候选 plan 选择可搬运 API 状态标注的 present 决策点。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.temporal_sampling import (  # noqa: E402
    REFERENCE_POLICY_FIXED_ANCHOR,
    SequenceCasePlan,
    TemporalCaseSamplingPlan,
)
from cogtrack.training.tracking_samples import _presence  # noqa: E402
from pytracking.datasets import iter_dataset  # noqa: E402
from pytracking.evaluation.environment import load_environment  # noqa: E402


def _load_plan(path: Path) -> TemporalCaseSamplingPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = TemporalCaseSamplingPlan.from_dict(payload)
    if plan.reference_policy != REFERENCE_POLICY_FIXED_ANCHOR:
        raise ValueError("API 状态标注必须使用 fixed_identity_anchor plan")
    return plan


def _stable_sequence_order(dataset: str, sequence: str, seed: int) -> bytes:
    return hashlib.sha256(f"api-select\0{seed}\0{dataset}\0{sequence}".encode()).digest()


def select_api_plan(
    plan: TemporalCaseSamplingPlan,
    sequences_by_key: dict[tuple[str, str], Any],
    *,
    datasets: tuple[str, ...],
    sequences_per_dataset: int,
    stride: int,
    max_steps_per_sequence: int,
    min_planned_frames: int,
    min_anchor_gap: int,
    seed: int,
) -> TemporalCaseSamplingPlan:
    """确定性选择序列，并只保留具有 GT bbox 的 present 决策帧。"""

    selected: list[SequenceCasePlan] = []
    entries_by_dataset: dict[str, list[SequenceCasePlan]] = {name: [] for name in datasets}
    for entry in plan.sequences:
        if entry.dataset in entries_by_dataset and len(entry.frame_ids) >= min_planned_frames:
            entries_by_dataset[entry.dataset].append(entry)
    for dataset in datasets:
        entries = sorted(
            entries_by_dataset[dataset],
            key=lambda item: _stable_sequence_order(dataset, item.sequence, seed),
        )[:sequences_per_dataset]
        for entry in entries:
            sequence = sequences_by_key.get((dataset, entry.sequence))
            if sequence is None:
                continue
            eligible = [
                frame_id
                for frame_id in entry.frame_ids
                if frame_id >= entry.anchor_frame_id + min_anchor_gap
                and _presence(sequence, frame_id) in {"present", "absent"}
            ]
            # 优先保留计划内的 present↔absent 边界，使动态指代表达同时覆盖消失和重现；
            # 再用固定步长帧补齐一般状态变化。这里只使用 GT presence 选训练 case，信息
            # 不进入学生输入。
            priority: set[int] = set()
            previous_status = "present"
            for frame_id in eligible:
                status = _presence(sequence, frame_id)
                if status != previous_status:
                    priority.add(frame_id)
                previous_status = status
            ordered = sorted(priority)
            for frame_id in eligible[::stride]:
                if frame_id not in priority:
                    ordered.append(frame_id)
            points = tuple(sorted(ordered[:max_steps_per_sequence]))
            if not points:
                continue
            present_count = sum(_presence(sequence, frame_id) == "present" for frame_id in points)
            absent_count = len(points) - present_count
            selected.append(
                SequenceCasePlan(
                    dataset=dataset,
                    sequence=entry.sequence,
                    anchor_frame_id=entry.anchor_frame_id,
                    frame_ids=points,
                    reference_frame_ids=(entry.anchor_frame_id,) * len(points),
                    present_count=present_count,
                    absent_count=absent_count,
                    absent_run_count=sum(
                        _presence(sequence, frame_id) == "absent"
                        and (index == 0 or _presence(sequence, points[index - 1]) != "absent")
                        for index, frame_id in enumerate(points)
                    ),
                )
            )
    selected.sort(key=lambda item: (item.dataset, item.sequence))
    case_count = sum(len(item.frame_ids) for item in selected)
    present_count = sum(item.present_count for item in selected)
    absent_count = sum(item.absent_count for item in selected)
    if not selected:
        raise ValueError("筛选后没有 API 标注决策点")
    return TemporalCaseSamplingPlan(
        seed=seed,
        requested_absent_ratio=absent_count / case_count,
        actual_absent_ratio=absent_count / case_count,
        max_cases_per_sequence=max_steps_per_sequence,
        sequence_count=len(selected),
        case_count=case_count,
        present_count=present_count,
        absent_count=absent_count,
        absent_run_count=sum(item.absent_run_count for item in selected),
        sequences=tuple(selected),
        reference_policy=REFERENCE_POLICY_FIXED_ANCHOR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-plan", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--env-config")
    parser.add_argument("--datasets", default="lasot,tnl2k")
    parser.add_argument("--sequences-per-dataset", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-steps-per-sequence", type=int, default=30)
    parser.add_argument("--min-planned-frames", type=int, default=8)
    parser.add_argument("--min-anchor-gap", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    positive = {
        "sequences_per_dataset": args.sequences_per_dataset,
        "stride": args.stride,
        "max_steps_per_sequence": args.max_steps_per_sequence,
        "min_planned_frames": args.min_planned_frames,
    }
    if any(value <= 0 for value in positive.values()) or args.min_anchor_gap < 0:
        raise SystemExit("序列数、stride、steps、planned frames 必须为正，anchor gap 必须非负")
    datasets = tuple(value.strip().lower() for value in args.datasets.split(",") if value.strip())
    if not datasets:
        raise SystemExit("--datasets 不能为空")
    plan = _load_plan(Path(args.input_plan).expanduser().resolve())
    environment = load_environment(args.env_config)
    sequences_by_key: dict[tuple[str, str], Any] = {}
    for dataset in datasets:
        for sequence in iter_dataset(dataset, environment=environment, split="train"):
            if str(sequence.metadata.get("split", "")).lower() != "train":
                raise SystemExit(f"拒绝非 train 序列：{dataset}/{sequence.name}")
            sequences_by_key[(dataset, sequence.name)] = sequence
    selected = select_api_plan(
        plan,
        sequences_by_key,
        datasets=datasets,
        sequences_per_dataset=args.sequences_per_dataset,
        stride=args.stride,
        max_steps_per_sequence=args.max_steps_per_sequence,
        min_planned_frames=args.min_planned_frames,
        min_anchor_gap=args.min_anchor_gap,
        seed=args.seed,
    )
    output = Path(args.output_plan).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected.to_dict(), ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sequences": selected.sequence_count,
                "decision_points": selected.case_count,
                "by_dataset": {
                    dataset: sum(item.dataset == dataset for item in selected.sequences)
                    for dataset in datasets
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
