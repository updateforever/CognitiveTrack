"""MGIT action 分段 -> 逐帧记忆监督标签的聚焦测试。

重点覆盖三件事：三态判定是否与标注证据一致、脏标注是否一律降级为 masked_unknown、
以及输入侧是否绝不泄漏当前帧的目标文本。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cogtrack.training.loss_mask import (
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_HARD_NULL,
    MEMORY_STATE_VERIFIED_UPDATE,
    validate_memory_supervision_state,
)
from cogtrack.training.mgit_state_labels import (
    ActionSegment,
    FrameMemoryLabel,
    SegmentParseReport,
    build_frame_memory_labels,
    load_action_segments,
    normalize_state_text,
)

_A = "A man in a black suit walks in the washroom"
_B = "The man fights with a guard near the sink"
_C = "The man lifts an unconscious guard"


def _segments() -> list[ActionSegment]:
    return [
        ActionSegment(name="action_1", start_frame=0, end_frame=99, description=_A),
        ActionSegment(name="action_2", start_frame=100, end_frame=299, description=_B),
        ActionSegment(name="action_3", start_frame=300, end_frame=399, description=_C),
    ]


def _write_description(tmp_path, actions):
    path = tmp_path / "001.json"
    path.write_text(json.dumps({"action": actions}), encoding="utf-8")
    return path


def test_load_action_segments_normalizes_nbsp_frame_numbers_and_sorts(tmp_path):
    path = _write_description(
        tmp_path,
        {
            # 官方 JSON 里少量帧号是带 NBSP 的字符串，且键序不等于帧序。
            "action_2": {"start_frame": 100, "end_frame": 299, "description": _B},
            "action_1": {"start_frame": "0 ", "end_frame": "99 ", "description": _A},
        },
    )
    segments = load_action_segments(path)
    assert [seg.start_frame for seg in segments] == [0, 100]
    assert segments[0].description == _A


def test_load_action_segments_drops_dirty_segments_and_records_report(tmp_path):
    path = _write_description(
        tmp_path,
        {
            "action_1": {"start_frame": 0, "end_frame": 99, "description": _A},
            "action_2": {"start_frame": 4315, "end_frame": 1514, "description": _B},
            "action_3": {"start_frame": 200, "end_frame": 299, "description": "   "},
            "action_4": {"start_frame": "not-a-number", "end_frame": 500, "description": _C},
            "action_5": {"start_frame": 600, "description": _C},
            "action_6": "not-an-object",
        },
    )
    report = SegmentParseReport(sequence="001")
    segments = load_action_segments(path, report=report)
    assert [seg.name for seg in segments] == ["action_1"]
    assert report.total_segments == 6
    reasons = sorted(entry.split(":", 1)[1] for entry in report.dropped_segments)
    assert reasons == [
        "empty_description",
        "end_lt_start(4315>1514)",
        "missing_frame_bounds",
        "not_an_object",
        "unparsable_frame_number",
    ]


def test_normalize_state_text_collapses_whitespace_and_capitalizes():
    assert normalize_state_text("  the  garfield\teats\n a can ") == "The garfield eats a can"
    assert normalize_state_text("") == ""
    assert normalize_state_text("a") == "A"


def test_snapshot_advance_yields_verified_update_at_first_frame_past_boundary():
    # 采样计划 [90, 110]：snapshot 在 90 停在段 1，110 落进段 2 -> 文本变化 -> 更新。
    labels = build_frame_memory_labels(_segments(), [90, 110])
    label = labels[110]
    assert label.state == MEMORY_STATE_VERIFIED_UPDATE
    assert label.memory_update == _B
    assert label.input_state == _A
    assert label.reason == "action_segment_text_changed"


def test_first_frame_without_prior_snapshot_falls_back_to_initial_description():
    """首帧的输入侧就是初始身份描述，与 Prompt 的回落值一致。"""

    labels = build_frame_memory_labels(_segments(), [50])
    # 50 仍在段 1，snapshot 已被初始化为段 1 文本，因此不是更新。
    assert labels[50].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[50].input_state == _A


def test_update_target_never_appears_on_input_side():
    """防泄漏不变量：目标文本不得等于输入侧已存状态。"""

    plan = list(range(20, 400, 20))
    labels = build_frame_memory_labels(
        _segments(), plan, within_segment_hard_null=True, boundary_margin=30
    )
    assert any(label.state == MEMORY_STATE_VERIFIED_UPDATE for label in labels.values())
    for label in labels.values():
        validate_memory_supervision_state(label.state)
        if label.memory_update is not None:
            assert label.memory_update != label.input_state


def test_each_change_point_produces_exactly_one_update():
    """snapshot 前进后同一段内不再重复报更新，否则会把一次变化刷成几十条标签。"""

    plan = list(range(20, 400, 20))
    labels = build_frame_memory_labels(_segments(), plan)
    updates = [
        label for _, label in sorted(labels.items())
        if label.state == MEMORY_STATE_VERIFIED_UPDATE
    ]
    assert [label.memory_update for label in updates] == [_B, _C]


def test_absent_frame_is_hard_null_and_does_not_advance_snapshot():
    """absent 帧不吃掉变化点：下一个 present 帧仍要报这次更新。"""

    labels = build_frame_memory_labels(_segments(), [90, 110, 130], absent_frames={110})
    assert labels[110].state == MEMORY_STATE_VERIFIED_HARD_NULL
    assert labels[110].memory_update is None
    # 输入侧仍要给出已存状态，Prompt 才能展示当前记忆。
    assert labels[110].input_state == _A
    assert labels[110].reason == "absent_no_new_appearance_evidence"
    # snapshot 没前进，130 接着报同一次更新。
    assert labels[130].state == MEMORY_STATE_VERIFIED_UPDATE
    assert labels[130].memory_update == _B
    assert labels[130].input_state == _A


def test_stable_window_defaults_to_masked_unknown():
    labels = build_frame_memory_labels(_segments(), [110, 160])
    label = labels[160]
    assert label.state == MEMORY_STATE_MASKED_UNKNOWN
    assert label.reason == "within_segment_hard_null_disabled"
    assert label.input_state == _B


def test_stable_window_becomes_hard_null_only_deep_inside_segment():
    labels = build_frame_memory_labels(
        _segments(), [110, 120, 160, 290], within_segment_hard_null=True, boundary_margin=30
    )
    # 距最近变化点 60 帧，段内稳定。
    assert labels[160].state == MEMORY_STATE_VERIFIED_HARD_NULL
    assert labels[160].reason == "within_segment_state_text_unchanged"
    # 距变化点 100 只有 20 帧，标注边界本身有不确定性，不敢声明"确定不更新"。
    assert labels[120].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[120].reason == "within_boundary_margin(20)"
    # 距下一个变化点 300 只有 10 帧。
    assert labels[290].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[290].reason == "within_boundary_margin(10)"


def test_identical_consecutive_descriptions_are_not_treated_as_update():
    segments = [
        ActionSegment(name="action_1", start_frame=0, end_frame=99, description=_A),
        ActionSegment(name="action_2", start_frame=100, end_frame=199, description=_A),
    ]
    labels = build_frame_memory_labels(
        segments, [90, 110], within_segment_hard_null=True, boundary_margin=5
    )
    label = labels[110]
    assert label.state == MEMORY_STATE_VERIFIED_HARD_NULL
    assert label.memory_update is None


def test_uncovered_and_conflicting_frames_degrade_to_masked_unknown():
    segments = [
        ActionSegment(name="action_1", start_frame=0, end_frame=99, description=_A),
        ActionSegment(name="action_2", start_frame=200, end_frame=299, description=_B),
        ActionSegment(name="action_3", start_frame=250, end_frame=349, description=_C),
    ]
    report = SegmentParseReport(sequence="001")
    labels = build_frame_memory_labels(
        segments, [90, 150, 260], within_segment_hard_null=True, report=report
    )
    assert labels[150].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[150].reason == "frame_not_covered_by_any_segment"
    assert labels[260].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[260].reason == "overlapping_segments_conflicting_text"
    assert report.uncovered_frames == 1
    assert report.overlapping_frames == 1


def test_oversized_update_text_is_masked_not_truncated():
    long_text = " ".join(f"word{index}" for index in range(40))
    segments = [
        ActionSegment(name="action_1", start_frame=0, end_frame=99, description=_A),
        ActionSegment(name="action_2", start_frame=100, end_frame=199, description=long_text),
    ]
    report = SegmentParseReport(sequence="001")
    labels = build_frame_memory_labels(segments, [90, 110], report=report)
    label = labels[110]
    assert label.state == MEMORY_STATE_MASKED_UNKNOWN
    assert label.reason == "update_text_exceeds_label_limit"
    assert label.memory_update is None
    assert report.oversized_updates == ["frame=110"]


def test_frames_before_any_segment_have_no_snapshot():
    """整条序列都没有可用初始文本时，输入侧必须显式为 None，不能编一个。"""

    labels = build_frame_memory_labels([], [50, 150])
    assert labels[50].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[50].input_state is None
    assert labels[50].reason == "frame_not_covered_by_any_segment"


def test_explicit_initial_state_overrides_first_segment():
    """允许调用方传入 Prompt 实际展示的初始身份描述，保证两侧完全一致。"""

    labels = build_frame_memory_labels(_segments(), [50], initial_state="A custom identity")
    assert labels[50].input_state == "A custom identity"
    # 50 落在段 1，文本与传入的初始状态不同 -> 视为一次更新。
    assert labels[50].state == MEMORY_STATE_VERIFIED_UPDATE
    assert labels[50].memory_update == _A


def test_frame_plan_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="严格升序"):
        build_frame_memory_labels(_segments(), [110, 110])
    with pytest.raises(ValueError, match="严格升序"):
        build_frame_memory_labels(_segments(), [200, 110])
    with pytest.raises(ValueError, match="boundary_margin"):
        build_frame_memory_labels(_segments(), [110], boundary_margin=-1)


def test_frame_memory_label_rejects_self_contradicting_values():
    with pytest.raises(ValueError, match="必须带非空 memory_update"):
        FrameMemoryLabel(
            frame_id=1, state=MEMORY_STATE_VERIFIED_UPDATE, memory_update=None,
            input_state=_A, reason="x",
        )
    with pytest.raises(ValueError, match="不能等于输入侧已存状态"):
        FrameMemoryLabel(
            frame_id=1, state=MEMORY_STATE_VERIFIED_UPDATE, memory_update=_A,
            input_state=_A, reason="x",
        )
    with pytest.raises(ValueError, match="必须为 None"):
        FrameMemoryLabel(
            frame_id=1, state=MEMORY_STATE_VERIFIED_HARD_NULL, memory_update=_B,
            input_state=_A, reason="x",
        )


def test_dropped_update_leaves_snapshot_stale_instead_of_faking_hard_null():
    """回归：被丢弃的更新不能让后续帧变成"确定不更新"。

    若 snapshot 改按"上一帧所在段的文本"推导，110 处超限的更新被丢弃后，130 会被判成
    ``verified_hard_null``——而 Prompt 侧回放的记忆仍是段 1 的旧文本，等于一边显示过期
    记忆一边监督模型不要更新。这里锁住"snapshot 只在真正产出标签时前进"。
    """

    long_text = " ".join(f"word{index}" for index in range(40))
    segments = [
        ActionSegment(name="action_1", start_frame=0, end_frame=99, description=_A),
        ActionSegment(name="action_2", start_frame=100, end_frame=299, description=long_text),
        ActionSegment(name="action_3", start_frame=300, end_frame=399, description=_C),
    ]
    labels = build_frame_memory_labels(
        segments, [90, 110, 130, 310], within_segment_hard_null=True, boundary_margin=5
    )
    assert labels[110].reason == "update_text_exceeds_label_limit"
    assert labels[110].input_state == _A
    # 130 深在段 2 内部，但 snapshot 仍是段 1 文本，因此绝不能是 hard_null。
    assert labels[130].state == MEMORY_STATE_MASKED_UNKNOWN
    assert labels[130].reason == "update_text_exceeds_label_limit"
    assert labels[130].input_state == _A
    # 段 3 文本可用，snapshot 从未前进过，这里正常产出更新。
    assert labels[310].state == MEMORY_STATE_VERIFIED_UPDATE
    assert labels[310].memory_update == _C
    assert labels[310].input_state == _A


def test_hard_null_labels_are_written_with_verified_null_flag(tmp_path: Path) -> None:
    """回归：hard-null 必须落盘并带 verified_null=true。

    只写带文本的更新会让 --within-segment-hard-null 静默失效：下游看不到标签就当
    缺标签，于是这一帧退回 masked_unknown，开关等于没开。
    """
    plan = {
        "sequences": [
            {
                "dataset": "mgit",
                "sequence": "unit",
                "anchor_frame_id": 90,
                "frame_ids": [90, 150, 310],
                "reference_frame_ids": [0, 0, 0],
                "present_count": 3,
                "absent_count": 0,
                "absent_run_count": 0,
            }
        ]
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    desc_dir = tmp_path / "attribute" / "description"
    desc_dir.mkdir(parents=True)
    (desc_dir / "unit.json").write_text(
        json.dumps(
            {
                "action": {
                    "1": {"start_frame": 0, "end_frame": 200, "description": _A},
                    "2": {"start_frame": 201, "end_frame": 400, "description": _B},
                }
            }
        ),
        encoding="utf-8",
    )
    absent_dir = tmp_path / "attribute" / "absent"
    absent_dir.mkdir(parents=True)
    (absent_dir / "unit.txt").write_text("\n".join(["0"] * 401), encoding="utf-8")

    out = tmp_path / "labels.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "tools/archive/state_teacher_v1/build_mgit_state_update_labels.py"
            ),
            "--sampling-plan", str(plan_path),
            "--mgit-root", str(tmp_path),
            "--output", str(out),
            "--within-segment-hard-null",
            "--boundary-margin", "5",
        ],
        check=True,
        capture_output=True,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_frame = {row["frame_id"]: row for row in rows}

    # 段内稳定帧：null 是真负标签。
    assert by_frame[150]["memory_update"] is None
    assert by_frame[150]["verified_null"] is True
    assert by_frame[150]["source"] == "mgit_action_segment_stable_v1"

    # 变化点：带文本，且绝不能被当成 hard-null。
    assert by_frame[310]["memory_update"] == _B
    assert by_frame[310]["verified_null"] is False
    assert by_frame[310]["source"] == "mgit_action_segment_v1"
