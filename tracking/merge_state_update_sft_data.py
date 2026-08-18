#!/usr/bin/env python3
"""合并 MGIT 官方分段标签与额外 teacher/verifier 标签。

两个来源必须先各自生成固定锚点 sampling plan。合并器只保留确实有标签的
``(dataset, sequence, frame_id)``，再重算一个可重放的统一计划；这样显式
``state_update_sft`` 构建不会遇到“计划里有帧、标签里没有帧”的隐式
``masked_unknown``。

该工具不读取图像、不重新抽帧，也不改变标签文本。它负责的质量边界是：

* 两个输入计划都必须是 ``fixed_identity_anchor``；
* 每个标签必须来自对应计划且 current 严格晚于 reference；
* teacher 标签必须带旧版独立 verifier 报告，或新版强 API 单次标注质量门报告；
* 同一序列同一 current 不能有两条不同标签；
* 输出计划的计数、状态和参考帧全部由保留下来的标签重算。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.temporal_sampling import (  # noqa: E402
    REFERENCE_POLICY_FIXED_ANCHOR,
    SequenceCasePlan,
    TemporalCaseSamplingPlan,
)


def _read_json(path: str | Path) -> Any:
    source = Path(path).expanduser().resolve()
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON：{source}") from exc


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取 JSONL：{source}") from exc
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败：{source}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 每行必须是对象：{source}:{line_no}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL 为空：{source}")
    return rows


def _load_plan(path: str | Path) -> TemporalCaseSamplingPlan:
    source = Path(path).expanduser().resolve()
    try:
        return TemporalCaseSamplingPlan.from_dict(_read_json(source))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"sampling plan 无效：{source}: {exc}") from exc


def _plan_entries(plan: TemporalCaseSamplingPlan) -> dict[tuple[str, str], SequenceCasePlan]:
    if plan.reference_policy != REFERENCE_POLICY_FIXED_ANCHOR:
        raise ValueError(
            "state_update_sft 只接受 fixed_identity_anchor；"
            f"实际为 {plan.reference_policy!r}"
        )
    return {(item.dataset, item.sequence): item for item in plan.sequences}


def _teacher_report_verified(path: str | Path) -> bool:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return False
    # 这是正式 release 的硬门槛。dry-run 报告通常会把 dry_run=true 写入，不能混入。
    try:
        minimum = int(payload.get("min_output_labels", 0))
        written = int(payload.get("written_labels", 0))
    except (TypeError, ValueError):
        return False
    common = (
        not bool(payload.get("dry_run"))
        and bool(payload.get("minimum_output_reached"))
        and minimum >= 1200
        and written >= minimum
    )
    if not common:
        return False
    if bool(payload.get("independently_verified")):
        return True
    return (
        payload.get("annotation_policy") == "single_pass_frontier_api_v1"
        and bool(payload.get("single_pass_frontier_teacher"))
        and bool(payload.get("quality_gate_applied"))
        and bool(str(payload.get("teacher_model") or "").strip())
        and bool(str(payload.get("prompt_version") or "").strip())
    )


def _label_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    dataset = str(row.get("dataset", row.get("source_dataset", ""))).strip().lower()
    sequence = str(row.get("sequence", row.get("source_sequence", ""))).strip()
    if not dataset or not sequence or "frame_id" not in row:
        raise ValueError("标签缺少 dataset/sequence/frame_id")
    try:
        frame_id = int(row["frame_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"标签 frame_id 非整数：{row.get('frame_id')!r}") from exc
    if frame_id < 0:
        raise ValueError("标签 frame_id 不能为负数")
    return dataset, sequence, frame_id


def _normalise_label(row: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
    dataset, sequence, frame_id = _label_key(row)
    if "memory_update" not in row:
        raise ValueError(f"{source_name} 标签缺少 memory_update：{dataset}/{sequence}/{frame_id}")
    status = str(row.get("target_status", "")).strip().lower()
    if status not in {"present", "absent"}:
        raise ValueError(
            f"{source_name} 标签必须显式带 target_status=present/absent："
            f"{dataset}/{sequence}/{frame_id}"
        )
    value = row.get("memory_update")
    if value is not None and not isinstance(value, str):
        raise ValueError(f"非空 memory_update 必须是字符串：{dataset}/{sequence}/{frame_id}")
    if value is None and not bool(row.get("verified_null", False)):
        raise ValueError(
            f"state_update_sft 的 null 必须是 verified_null=true：{dataset}/{sequence}/{frame_id}"
        )
    if source_name == "teacher":
        independently_reviewed = bool(row.get("reviewed", False))
        api_gate = (
            bool(row.get("quality_gate_passed", False))
            and row.get("review_method") == "single_pass_frontier_api_v1"
        )
        if not independently_reviewed and not api_gate:
            raise ValueError(
                "teacher 标签必须逐行带 reviewed=true，或带强 API 单次标注质量门："
                f"{dataset}/{sequence}/{frame_id}"
            )
    normalised = dict(row)
    normalised.update(
        {
            "dataset": dataset,
            "sequence": sequence,
            "frame_id": frame_id,
            "target_status": status,
        }
    )
    return normalised


def _validate_source_labels(
    plan: TemporalCaseSamplingPlan,
    rows: Iterable[Mapping[str, Any]],
    *,
    source_name: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    entries = _plan_entries(plan)
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in rows:
        row = _normalise_label(raw, source_name=source_name)
        key = _label_key(row)
        dataset, sequence, frame_id = key
        entry = entries.get((dataset, sequence))
        if entry is None:
            raise ValueError(f"{source_name} 标签不在输入 plan：{dataset}/{sequence}")
        if frame_id not in entry.frame_ids:
            raise ValueError(
                f"{source_name} 标签 frame 不在输入 plan：{dataset}/{sequence}/{frame_id}"
            )
        reference = entry.reference_frame_ids[entry.frame_ids.index(frame_id)]
        if reference >= frame_id:
            raise ValueError(
                f"{source_name} reference 必须严格早于 current："
                f"{dataset}/{sequence}/{reference}->{frame_id}"
            )
        previous = output.get(key)
        if previous is not None:
            # 同一来源的 duplicate 也必须完全一致，避免拼接时静默覆盖。
            if previous.get("memory_update") != row.get("memory_update") or previous.get(
                "target_status"
            ) != row.get("target_status"):
                raise ValueError(f"{source_name} 存在冲突的重复标签：{key}")
            raise ValueError(f"{source_name} 存在重复标签：{key}")
        output[key] = row
    if not output:
        raise ValueError(f"{source_name} 没有可合并标签")
    if source_name == "teacher":
        by_sequence: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for (dataset, sequence, _frame_id), row in output.items():
            by_sequence[(dataset, sequence)].append(row)
        for key, sequence_rows in by_sequence.items():
            sequence_rows.sort(key=lambda item: int(item["frame_id"]))
            expected_state = str(sequence_rows[0].get("input_state") or "").strip()
            if not expected_state:
                raise ValueError(f"teacher 状态链缺少首个 input_state：{key}")
            for row in sequence_rows:
                actual_state = str(row.get("input_state") or "").strip()
                if actual_state != expected_state:
                    raise ValueError(
                        "teacher 状态链断裂："
                        f"{key} frame={row['frame_id']} expected={expected_state!r} "
                        f"actual={actual_state!r}"
                    )
                if row["memory_update"] is not None:
                    expected_state = str(row["memory_update"])
    return output


def _merge(
    mgit_plan: TemporalCaseSamplingPlan,
    mgit_rows: dict[tuple[str, str, int], dict[str, Any]],
    teacher_plan: TemporalCaseSamplingPlan,
    teacher_rows: dict[tuple[str, str, int], dict[str, Any]],
    *,
    seed: int,
    max_cases_per_sequence: int,
) -> tuple[TemporalCaseSamplingPlan, list[dict[str, Any]], dict[str, Any]]:
    mgit_entries = _plan_entries(mgit_plan)
    teacher_entries = _plan_entries(teacher_plan)
    all_entries = {**mgit_entries}
    overlap_sequences = set(all_entries).intersection(teacher_entries)
    if overlap_sequences:
        raise ValueError(
            "MGIT 与 teacher 计划出现同名序列，拒绝混合以避免标签来源覆盖："
            f"{sorted(overlap_sequences)[:3]}"
        )
    all_entries.update(teacher_entries)

    combined: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rows in (mgit_rows, teacher_rows):
        for key, row in rows.items():
            if key in combined:
                raise ValueError(f"合并标签出现重复 dataset/sequence/frame：{key}")
            combined[key] = dict(row)

    by_sequence: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, sequence, _), row in combined.items():
        by_sequence[(dataset, sequence)].append(row)

    plans: list[SequenceCasePlan] = []
    for key in sorted(by_sequence):
        entry = all_entries.get(key)
        if entry is None:
            raise ValueError(f"合并标签缺少对应序列计划：{key}")
        rows = sorted(by_sequence[key], key=lambda row: int(row["frame_id"]))
        frame_ids = tuple(int(row["frame_id"]) for row in rows)
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"合并标签有重复 current frame：{key}")
        reference_by_frame = dict(zip(entry.frame_ids, entry.reference_frame_ids, strict=True))
        references = tuple(reference_by_frame[frame_id] for frame_id in frame_ids)
        if any(reference != entry.anchor_frame_id for reference in references):
            raise ValueError(f"合并计划不是固定 identity anchor：{key}")
        plans.append(
            SequenceCasePlan(
                dataset=key[0],
                sequence=key[1],
                anchor_frame_id=entry.anchor_frame_id,
                frame_ids=frame_ids,
                reference_frame_ids=references,
                present_count=sum(row["target_status"] == "present" for row in rows),
                absent_count=sum(row["target_status"] == "absent" for row in rows),
                absent_run_count=0,
            )
        )

    present = sum(item.present_count for item in plans)
    absent = sum(item.absent_count for item in plans)
    cases = present + absent
    ratio = absent / cases if cases else 0.0
    merged_plan = TemporalCaseSamplingPlan(
        seed=seed,
        requested_absent_ratio=ratio,
        actual_absent_ratio=ratio,
        max_cases_per_sequence=max_cases_per_sequence,
        sequence_count=len(plans),
        case_count=cases,
        present_count=present,
        absent_count=absent,
        absent_run_count=sum(item.absent_run_count for item in plans),
        sequences=tuple(plans),
        reference_policy=REFERENCE_POLICY_FIXED_ANCHOR,
    )
    merged_rows = [combined[key] for key in sorted(combined)]
    report = {
        "schema_version": "cogtrack.state_update_sft_merge.v1",
        "sources": {
            "mgit_action_segments": {
                "sequences": len({(r["dataset"], r["sequence"]) for r in mgit_rows.values()}),
                "labels": len(mgit_rows),
                "updates": sum(r["memory_update"] is not None for r in mgit_rows.values()),
                "hard_nulls": sum(r["memory_update"] is None for r in mgit_rows.values()),
            },
            "additional_state_labels": {
                "sequences": len({(r["dataset"], r["sequence"]) for r in teacher_rows.values()}),
                "labels": len(teacher_rows),
                "updates": sum(r["memory_update"] is not None for r in teacher_rows.values()),
                "hard_nulls": sum(r["memory_update"] is None for r in teacher_rows.values()),
            },
        },
        "merged": {
            "sequences": len(plans),
            "labels": cases,
            "present": present,
            "absent": absent,
            "absent_ratio": ratio,
            "updates": sum(r["memory_update"] is not None for r in merged_rows),
            "hard_nulls": sum(r["memory_update"] is None for r in merged_rows),
            "by_dataset": dict(
                sorted(Counter(str(r["dataset"]) for r in merged_rows).items())
            ),
        },
        "reference_policy": REFERENCE_POLICY_FIXED_ANCHOR,
        "plan_seed": seed,
        "max_cases_per_sequence": max_cases_per_sequence,
    }
    return merged_plan, merged_rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgit-plan", required=True)
    parser.add_argument("--mgit-labels", required=True)
    parser.add_argument("--teacher-plan", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--teacher-report", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-cases-per-sequence", type=int, default=10000)
    args = parser.parse_args()
    if args.max_cases_per_sequence <= 0:
        raise SystemExit("--max-cases-per-sequence 必须为正数")
    if not _teacher_report_verified(args.teacher_report):
        raise SystemExit(
            "teacher report 未满足旧版独立 verifier 或新版 single_pass_frontier_api_v1"
            " 的正式质量门，且必须 dry_run=false、minimum_output_reached=true、"
            "min_output_labels>=1200；该标签包不能进入正式 state_update_sft"
        )

    mgit_plan = _load_plan(args.mgit_plan)
    teacher_plan = _load_plan(args.teacher_plan)
    mgit_rows = _validate_source_labels(
        mgit_plan, _read_jsonl(args.mgit_labels), source_name="MGIT"
    )
    teacher_rows = _validate_source_labels(
        teacher_plan, _read_jsonl(args.teacher_labels), source_name="teacher"
    )
    merged_plan, merged_rows, report = _merge(
        mgit_plan,
        mgit_rows,
        teacher_plan,
        teacher_rows,
        seed=args.seed,
        max_cases_per_sequence=args.max_cases_per_sequence,
    )

    output_plan = Path(args.output_plan).expanduser().resolve()
    output_labels = Path(args.output_labels).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    for path in (output_plan, output_labels, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_plan.write_text(
        json.dumps(merged_plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with output_labels.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    report["inputs"] = {
        "mgit_plan": str(Path(args.mgit_plan).expanduser().resolve()),
        "mgit_labels": str(Path(args.mgit_labels).expanduser().resolve()),
        "teacher_plan": str(Path(args.teacher_plan).expanduser().resolve()),
        "teacher_labels": str(Path(args.teacher_labels).expanduser().resolve()),
        "teacher_report": str(Path(args.teacher_report).expanduser().resolve()),
    }
    report["outputs"] = {
        "sampling_plan": str(output_plan),
        "labels": str(output_labels),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
