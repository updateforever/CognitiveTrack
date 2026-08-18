#!/usr/bin/env python3
"""审计 tracking SFT 的固定三图输入、场景分类与监督边界。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.loss_mask import (  # noqa: E402
    MEMORY_STATE_MASKED_UNKNOWN,
    SFT_SUPERVISION_TRACKING_SFT,
    validate_sft_supervision_profile,
)
from cogtrack.training.swift_dataset import read_jsonl  # noqa: E402
from cogtrack.training.tracking_samples import (  # noqa: E402
    ABSENT_PHASES,
    HISTORY_COMPLETENESS_H0,
    HISTORY_COMPLETENESS_H1,
    HISTORY_COMPLETENESS_H2,
    HISTORY_COMPLETENESS_H3,
    HISTORY_COMPLETENESS_LEVELS,
    HISTORY_QUALITIES,
    HISTORY_QUALITY_CLEAN,
    HISTORY_QUALITY_STALE,
    REFERENCE_SOURCES,
    TRACKING_EVENT_ABSENT,
    TRACKING_EVENTS,
    TRACKING_SCENARIOS,
    VALID_HISTORY_QUALITIES_BY_COMPLETENESS,
    VALID_VISUAL_COMBINATIONS,
)


def _required_mapping(value: Any, *, name: str, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: {name} 必须是对象")
    return value


def _history_completeness(
    history_frame_ids: list[int], *, reference_frame_id: int
) -> str:
    unique_ids = set(history_frame_ids)
    if unique_ids == {reference_frame_id}:
        return HISTORY_COMPLETENESS_H0
    count = len(unique_ids)
    if count == 1:
        return HISTORY_COMPLETENESS_H1
    if count == 2:
        return HISTORY_COMPLETENESS_H2
    if count == 3:
        return HISTORY_COMPLETENESS_H3
    raise ValueError(f"固定三格历史出现非法 distinct frame 数量：{count}")


def _empty_counter(keys: set[str] | frozenset[str]) -> Counter[str]:
    return Counter({key: 0 for key in keys})


def validate_tracking_sft_dataset(
    source_jsonl: str | Path,
    *,
    build_report: str | Path | None = None,
    require_complete_coverage: bool = False,
) -> dict[str, Any]:
    """校验一份 source JSONL，并返回可序列化的场景统计。"""

    counts = {
        "temporal_event_counts": _empty_counter(TRACKING_EVENTS),
        "absent_phase_counts": _empty_counter(ABSENT_PHASES),
        "history_quality_counts": _empty_counter(HISTORY_QUALITIES),
        "history_completeness_counts": _empty_counter(HISTORY_COMPLETENESS_LEVELS),
        "tracking_scenario_counts": _empty_counter(TRACKING_SCENARIOS),
        "visual_combination_counts": _empty_counter(VALID_VISUAL_COMBINATIONS),
        "reference_source_counts": _empty_counter(REFERENCE_SOURCES),
    }
    sample_count = 0
    for row in read_jsonl(source_jsonl):
        sample_count += 1
        source = str(row.get("_source", f"{source_jsonl}:?"))
        metadata = _required_mapping(row.get("metadata"), name="metadata", source=source)
        profile = validate_sft_supervision_profile(
            str(metadata.get("sft_supervision_profile", ""))
        )
        if profile != SFT_SUPERVISION_TRACKING_SFT:
            raise ValueError(f"{source}: 不是 tracking_sft 监督档位")
        if metadata.get("prompt_profile") != "vlt_v6":
            raise ValueError(f"{source}: tracking_sft 必须使用 vlt_v6 Prompt")

        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"{source}: tracking_sft 必须严格包含三张输入图")
        if not Path(str(images[0])).name.startswith("reference_boxed_"):
            raise ValueError(f"{source}: Image 1 必须是带框 reference 资产")
        if not Path(str(images[1])).name.startswith("history_"):
            raise ValueError(f"{source}: Image 2 必须是固定三格 history 资产")
        if not Path(str(images[2])).name.startswith("current_"):
            raise ValueError(f"{source}: Image 3 必须是未画框 current 资产")

        reference_frame_id = int(metadata["reference_frame_id"])
        current_frame_id = int(metadata["frame_id"])
        if reference_frame_id >= current_frame_id:
            raise ValueError(f"{source}: reference 必须严格早于 current")
        raw_history = metadata.get("history_frame_ids")
        if not isinstance(raw_history, list) or len(raw_history) != 3:
            raise ValueError(f"{source}: Image 2 必须对应三个历史 panel")
        history_frame_ids = [int(frame_id) for frame_id in raw_history]
        dynamic_ids = [
            frame_id for frame_id in history_frame_ids if frame_id != reference_frame_id
        ]
        if any(
            frame_id <= reference_frame_id or frame_id >= current_frame_id
            for frame_id in dynamic_ids
        ):
            raise ValueError(f"{source}: 动态历史必须严格位于 reference 与 current 之间")
        if history_frame_ids != sorted(history_frame_ids):
            raise ValueError(f"{source}: 三个历史 panel 必须按时间从左到右排列")

        completeness = _history_completeness(
            history_frame_ids, reference_frame_id=reference_frame_id
        )
        if metadata.get("history_completeness") != completeness:
            raise ValueError(f"{source}: history_completeness 与真实 panel 帧号不一致")
        quality = str(metadata.get("history_quality"))
        if quality not in VALID_HISTORY_QUALITIES_BY_COMPLETENESS[completeness]:
            raise ValueError(
                f"{source}: {completeness} 不允许 history_quality={quality}"
            )
        corruption = metadata.get("history_corruption")
        if quality == HISTORY_QUALITY_CLEAN:
            if corruption is not None:
                raise ValueError(f"{source}: clean 行不能声明 history_corruption")
        elif corruption != quality:
            raise ValueError(f"{source}: history_quality 与 history_corruption 不一致")
        if quality == HISTORY_QUALITY_STALE and len(set(dynamic_ids)) < 2:
            raise ValueError(f"{source}: stale_box 至少需要两个不同动态历史观测")

        event = str(metadata.get("temporal_event"))
        if event not in TRACKING_EVENTS:
            raise ValueError(f"{source}: 未知 temporal_event={event}")
        status = str(row.get("target_status"))
        if (event == TRACKING_EVENT_ABSENT) != (status == "absent"):
            raise ValueError(f"{source}: temporal_event 与模型目标 status 矛盾")
        absent_phase = metadata.get("absent_phase")
        if event == TRACKING_EVENT_ABSENT:
            if absent_phase not in ABSENT_PHASES:
                raise ValueError(f"{source}: absent 行缺少合法 absent_phase")
            counts["absent_phase_counts"][str(absent_phase)] += 1
        elif absent_phase is not None:
            raise ValueError(f"{source}: present/reappearance 行的 absent_phase 必须为 null")

        scenario = f"{event}__{quality}"
        combination = f"{scenario}__{completeness}"
        if metadata.get("tracking_scenario") != scenario:
            raise ValueError(f"{source}: tracking_scenario 与基础分类不一致")
        if metadata.get("visual_combination") != combination:
            raise ValueError(f"{source}: visual_combination 与基础分类不一致")
        if combination not in VALID_VISUAL_COMBINATIONS:
            raise ValueError(f"{source}: 非法 visual_combination={combination}")
        reference_source = str(metadata.get("reference_source"))
        if reference_source not in REFERENCE_SOURCES:
            raise ValueError(f"{source}: 未知 reference_source={reference_source}")

        answer = _required_mapping(row.get("assistant"), name="assistant", source=source)
        if list(answer) != ["bbox_2d", "status", "memory_update"]:
            raise ValueError(f"{source}: assistant 字段及顺序必须是 bbox_2d/status/memory_update")
        if answer["status"] != status or answer["memory_update"] is not None:
            raise ValueError(f"{source}: tracking_sft 的目标 JSON 与监督边界不一致")
        state = metadata.get("memory_supervision_state")
        expected_state = MEMORY_STATE_MASKED_UNKNOWN
        if state != expected_state:
            raise ValueError(f"{source}: tracking_sft 的状态更新监督应为 {expected_state}")

        counts["temporal_event_counts"][event] += 1
        counts["history_quality_counts"][quality] += 1
        counts["history_completeness_counts"][completeness] += 1
        counts["tracking_scenario_counts"][scenario] += 1
        counts["visual_combination_counts"][combination] += 1
        counts["reference_source_counts"][reference_source] += 1

    if not sample_count:
        raise ValueError(f"tracking_sft 数据为空：{source_jsonl}")

    serialized_counts = {
        name: {key: int(counter[key]) for key in sorted(counter)}
        for name, counter in counts.items()
    }
    if build_report is not None:
        report = json.loads(Path(build_report).read_text(encoding="utf-8"))
        if int(report.get("sample_count", -1)) != sample_count:
            raise ValueError("build_report.sample_count 与 source JSONL 不一致")
        for name, actual in serialized_counts.items():
            if report.get(name) != actual:
                raise ValueError(f"build_report.{name} 与逐行复算不一致")

    missing_scenarios = sorted(
        key for key, value in serialized_counts["tracking_scenario_counts"].items() if value == 0
    )
    missing_combinations = sorted(
        key for key, value in serialized_counts["visual_combination_counts"].items() if value == 0
    )
    if require_complete_coverage and (missing_scenarios or missing_combinations):
        raise ValueError(
            "正式 tracking_sft 未覆盖全部分类："
            f"missing_scenarios={missing_scenarios} "
            f"missing_visual_combinations={missing_combinations}"
        )
    return {
        "sample_count": sample_count,
        **serialized_counts,
        "missing_tracking_scenarios": missing_scenarios,
        "missing_visual_combinations": missing_combinations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="source_samples.jsonl")
    parser.add_argument("--build-report", help="可选 build_report.json，用于统计交叉核对")
    parser.add_argument("--require-complete-coverage", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_tracking_sft_dataset(
            args.dataset,
            build_report=args.build_report,
            require_complete_coverage=args.require_complete_coverage,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
