"""从 MGIT action 分段推导逐帧 ``memory_update`` 监督标签。

MGIT 的 ``attribute/description/<seq>.json`` 在 ``action`` 层给出多段带
``start_frame`` / ``end_frame`` / ``description`` 的细粒度文本。第一段的 description
已被 :mod:`pytracking.datasets.mgit` 用作初始目标描述（``language_scope =
first_action_description``），所以后续段的 description 天然就是"目标状态发生变化"的
人工标注，正好是三态监督里 ``verified_update`` 的来源。

设计要点：

* **不虚构文本。** ``memory_update`` 只能是标注原文（仅空白归一 + 首字母大写），
  不摘要、不改写、不截断。
* **snapshot 与 Prompt 严格一致。** 帧 ``f`` 的输入侧已存记忆取"采样计划中严格早于
  ``f`` 的最近一条非空标签"，与 :func:`cogtrack.training.tracking_samples.
  _latest_prior_semantic_memory` 的回放规则逐字对应；为空时回落到初始身份描述。因此
  必须按采样计划顺序推进 snapshot，不能改用"上一帧所在段的文本"（见
  :func:`build_frame_memory_labels` 的说明）。
* **变化点由文本差异定义，而非段编号。** 只有 ``target_state(f) != snapshot`` 才算一次
  更新，因此相邻段文本相同（实测 7 处）不会被误判成更新。
* **标注脏数据一律降级为 ``masked_unknown``，不猜。** 实测 150 个文件中存在
  ``end_frame < start_frame``（3 处）、段间不连续（21 处）、段重叠，以及 21 段超过
  ``MemoryUpdateLabel`` 的 30 词上限。这些位置只能说"不知道"，不能说"确定不更新"。
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cogtrack.training.loss_mask import (
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_HARD_NULL,
    MEMORY_STATE_VERIFIED_UPDATE,
)

# 复用已审计的帧号归一化：官方 JSON 里少量帧号是带 NBSP 的字符串，必须按数值而非
# 字符串排序。刻意不复制实现，避免两处对脏帧号的处理逐渐分叉。
from pytracking.datasets.mgit import _normalize_action_start_frame as _normalize_frame_number

__all__ = [
    "ActionSegment",
    "FrameMemoryLabel",
    "SegmentParseReport",
    "MAX_MEMORY_UPDATE_CHARS",
    "MAX_MEMORY_UPDATE_WORDS",
    "MGIT_HARD_NULL_SOURCE",
    "MGIT_LABEL_SOURCE",
    "build_frame_memory_labels",
    "load_action_segments",
    "normalize_state_text",
    "state_text_at",
]

# 与 cogtrack.training.tracking_samples.MemoryUpdateLabel 的上限保持一致。超限的段
# 不截断，直接放弃该更新点（降级为 masked_unknown），避免把标注截成半句话。
MAX_MEMORY_UPDATE_CHARS = 256
MAX_MEMORY_UPDATE_WORDS = 30

MGIT_LABEL_SOURCE = "mgit_action_segment_v1"
# hard-null 用独立 source 串：审计时要能一眼分清"有文本的更新"和"已证明不更新"，
# 两者的证据强度和出错后果完全不同。
MGIT_HARD_NULL_SOURCE = "mgit_action_segment_stable_v1"


def normalize_state_text(text: str) -> str:
    """空白归一 + 首字母大写，与 MGIT loader 的初始描述处理保持一致。"""

    normalized = " ".join(str(text).split())
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized.upper()
    return normalized[0].upper() + normalized[1:]


def _is_labelable_update(text: str) -> bool:
    """该文本能否作为 ``MemoryUpdateLabel`` 的取值。"""

    return bool(text) and len(text) <= MAX_MEMORY_UPDATE_CHARS and len(text.split()) <= MAX_MEMORY_UPDATE_WORDS


@dataclass(frozen=True)
class ActionSegment:
    """一段 MGIT action 标注（闭区间帧号，0-based）。"""

    name: str
    start_frame: int
    end_frame: int
    description: str

    def contains(self, frame_id: int) -> bool:
        return self.start_frame <= frame_id <= self.end_frame


@dataclass(frozen=True)
class FrameMemoryLabel:
    """一帧的记忆监督标签及其可审计依据。

    ``input_state`` 是该帧 Prompt 里应当呈现的已存记忆（覆盖 ``frame_id - 1`` 的段
    文本）；``memory_update`` 是该帧 response 第三字段的目标值。二者分离是防泄漏的
    关键：``memory_update`` 非空时它一定不等于 ``input_state``，且绝不能出现在输入侧。
    """

    frame_id: int
    state: str
    memory_update: str | None
    input_state: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.state == MEMORY_STATE_VERIFIED_UPDATE:
            if not self.memory_update:
                raise ValueError("verified_update 必须带非空 memory_update")
            if self.memory_update == self.input_state:
                raise ValueError("verified_update 的目标文本不能等于输入侧已存状态")
        elif self.memory_update is not None:
            raise ValueError(f"{self.state} 的 memory_update 必须为 None")


@dataclass
class SegmentParseReport:
    """解析 / 建标过程中被丢弃或降级的部分，用于可审计统计。"""

    sequence: str
    total_segments: int = 0
    dropped_segments: list[str] = field(default_factory=list)
    overlapping_frames: int = 0
    uncovered_frames: int = 0
    oversized_updates: list[str] = field(default_factory=list)
    duplicate_consecutive: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "total_segments": self.total_segments,
            "dropped_segments": list(self.dropped_segments),
            "overlapping_frames": self.overlapping_frames,
            "uncovered_frames": self.uncovered_frames,
            "oversized_updates": list(self.oversized_updates),
            "duplicate_consecutive": list(self.duplicate_consecutive),
        }


def load_action_segments(
    path: str | Path,
    *,
    report: SegmentParseReport | None = None,
) -> list[ActionSegment]:
    """读取并清洗一个 MGIT 描述文件的 action 分段。

    脏段（非对象、帧号不可解析、``end_frame < start_frame``、description 为空）被丢弃
    并记入 ``report``，而不是抛异常：单个序列的标注瑕疵不应让整个数据集构建失败，
    但必须留下审计痕迹。返回结果按 ``start_frame`` 升序。
    """

    description_path = Path(path).expanduser()
    try:
        with description_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"MGIT 描述文件损坏: {description_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"MGIT 描述文件顶层必须是对象: {description_path}")
    actions = payload.get("action") or {}
    if not isinstance(actions, Mapping):
        raise ValueError(f"MGIT 描述文件 {description_path} 的 action 层必须是对象")

    segments: list[ActionSegment] = []
    for name, raw in actions.items():
        if report is not None:
            report.total_segments += 1
        if not isinstance(raw, Mapping):
            _drop(report, name, "not_an_object")
            continue
        if "start_frame" not in raw or "end_frame" not in raw:
            _drop(report, name, "missing_frame_bounds")
            continue
        try:
            start = _normalize_frame_number(
                raw["start_frame"], path=description_path, action_name=name
            )
            end = _normalize_frame_number(
                raw["end_frame"], path=description_path, action_name=name
            )
        except ValueError:
            _drop(report, name, "unparsable_frame_number")
            continue
        if end < start:
            # 实测 3 处（037/051/361）。区间反了就无法判断覆盖范围，只能丢弃。
            _drop(report, name, f"end_lt_start({start}>{end})")
            continue
        description = normalize_state_text(raw.get("description") or "")
        if not description:
            _drop(report, name, "empty_description")
            continue
        segments.append(
            ActionSegment(name=name, start_frame=start, end_frame=end, description=description)
        )

    segments.sort(key=lambda seg: (seg.start_frame, seg.end_frame, seg.name))
    return segments


def _drop(report: SegmentParseReport | None, name: str, reason: str) -> None:
    if report is not None:
        report.dropped_segments.append(f"{name}:{reason}")


def _state_at(segments: "list[ActionSegment]", frame_id: int) -> tuple[str | None, bool]:
    """返回 ``frame_id`` 处的段文本，以及是否因段重叠而歧义。

    重叠但文本相同不算歧义（实测存在相邻段文本完全一致的情况）；重叠且文本冲突时
    无法判断真值，返回歧义标记，交由调用方降级为 ``masked_unknown``。
    """

    texts = {seg.description for seg in segments if seg.contains(frame_id)}
    if not texts:
        return None, False
    if len(texts) > 1:
        return None, True
    return next(iter(texts)), False


def state_text_at(
    segments: "list[ActionSegment]", frame_id: int
) -> tuple[str | None, bool]:
    """``_state_at`` 的公开别名，供采样层按模板帧重锚身份文本。

    返回 ``(text, ambiguous)``；``text is None`` 表示无段覆盖该帧，``ambiguous`` 表示
    多段重叠且文本冲突。两种情况调用方都必须降级，不能猜。
    """

    return _state_at(segments, frame_id)


def _boundary_frames(segments: "list[ActionSegment]") -> list[int]:
    """文本真正发生变化的帧号（升序、去重）。

    只在段边缘取候选：文本仅可能在 ``start_frame`` 或 ``end_frame + 1`` 处变化，
    因此无需逐帧扫描整条上万帧的序列。
    """

    candidates: set[int] = set()
    for seg in segments:
        candidates.add(seg.start_frame)
        candidates.add(seg.end_frame + 1)
    boundaries: list[int] = []
    for frame_id in sorted(candidates):
        if frame_id <= 0:
            continue
        current, current_ambiguous = _state_at(segments, frame_id)
        previous, previous_ambiguous = _state_at(segments, frame_id - 1)
        if current_ambiguous or previous_ambiguous or current != previous:
            boundaries.append(frame_id)
    return boundaries


def _distance_to_boundary(boundaries: "list[int]", frame_id: int) -> int:
    """到最近文本变化点的帧距；没有任何变化点时视为无穷远。"""

    if not boundaries:
        return 1 << 30
    index = bisect_left(boundaries, frame_id)
    best = 1 << 30
    if index < len(boundaries):
        best = min(best, abs(boundaries[index] - frame_id))
    if index > 0:
        best = min(best, abs(frame_id - boundaries[index - 1]))
    return best

def build_frame_memory_labels(
    segments: "list[ActionSegment]",
    frame_plan: "list[int]",
    *,
    absent_frames: "frozenset[int] | set[int] | None" = None,
    initial_state: str | None = None,
    boundary_margin: int = 30,
    within_segment_hard_null: bool = False,
    report: SegmentParseReport | None = None,
) -> dict[int, FrameMemoryLabel]:
    """按采样计划顺序生成逐帧记忆标签。

    ``frame_plan`` 必须是该序列**实际被采样的 current frame 升序列表**，因为输入侧的
    已存记忆完全由采样计划决定：:func:`cogtrack.training.tracking_samples.
    _latest_prior_semantic_memory` 取"严格早于当前帧的最近一条非空标签"，为空时 Prompt
    回落到初始身份描述。本函数顺序推进同一个 snapshot，保证标签与 Prompt 逐帧一致。

    这一点不能简化成 ``state_at(frame_plan[i - 1])``：当某个变化点因文本超限或落在
    absent 帧而没能产出标签时，snapshot 会继续停留在旧文本，此时下一帧的正确答案仍是
    "更新"。若按上一帧的真实段文本判断，就会错判成 ``verified_hard_null``，让 Prompt
    显示过期记忆却又监督模型不要更新。

    ``absent_frames`` 里的帧强制 ``verified_hard_null``：目标不可见就观测不到新外观
    证据，确定不该替换已存记忆（与 ``GatedMemoryUpdatePolicy`` 只接受 present 观测一致），
    且 snapshot 不前进，变化点会在下一个 present 帧重试。
    """

    if boundary_margin < 0:
        raise ValueError("boundary_margin 不能为负数")
    plan = [int(frame_id) for frame_id in frame_plan]
    if any(later <= earlier for earlier, later in zip(plan, plan[1:], strict=False)):
        raise ValueError("frame_plan 必须严格升序且不含重复帧")
    absent = frozenset(absent_frames or ())
    boundaries = _boundary_frames(segments)
    if initial_state is None and segments:
        # Prompt 的回落值是初始身份描述，即第一段 description（与 MGIT loader 的
        # language_scope=first_action_description 一致）。
        initial_state = segments[0].description
    snapshot = initial_state
    labels: dict[int, FrameMemoryLabel] = {}

    for frame_id in plan:
        target_state, target_ambiguous = _state_at(segments, frame_id)

        if frame_id in absent:
            label = FrameMemoryLabel(
                frame_id=frame_id,
                state=MEMORY_STATE_VERIFIED_HARD_NULL,
                memory_update=None,
                input_state=snapshot,
                reason="absent_no_new_appearance_evidence",
            )
        elif target_ambiguous:
            if report is not None:
                report.overlapping_frames += 1
            label = _masked(frame_id, snapshot, "overlapping_segments_conflicting_text")
        elif target_state is None:
            if report is not None:
                report.uncovered_frames += 1
            label = _masked(frame_id, snapshot, "frame_not_covered_by_any_segment")
        elif snapshot is None:
            label = _masked(frame_id, None, "no_prior_state_snapshot")
        elif target_state != snapshot:
            if _is_labelable_update(target_state):
                label = FrameMemoryLabel(
                    frame_id=frame_id,
                    state=MEMORY_STATE_VERIFIED_UPDATE,
                    memory_update=target_state,
                    input_state=snapshot,
                    reason="action_segment_text_changed",
                )
                # 只有真正产出标签时 snapshot 才前进，与 Prompt 侧的记忆回放一致。
                snapshot = target_state
            else:
                if report is not None:
                    report.oversized_updates.append(f"frame={frame_id}")
                label = _masked(frame_id, snapshot, "update_text_exceeds_label_limit")
        else:
            label = _stable_label(
                current_frame=frame_id,
                input_state=snapshot,
                boundaries=boundaries,
                boundary_margin=boundary_margin,
                within_segment_hard_null=within_segment_hard_null,
            )
        labels[frame_id] = label
    return labels




def _masked(frame_id: int, input_state: str | None, reason: str) -> FrameMemoryLabel:
    return FrameMemoryLabel(
        frame_id=frame_id,
        state=MEMORY_STATE_MASKED_UNKNOWN,
        memory_update=None,
        input_state=input_state,
        reason=reason,
    )


def _stable_label(
    *,
    current_frame: int,
    input_state: str,
    boundaries: "list[int]",
    boundary_margin: int,
    within_segment_hard_null: bool,
) -> FrameMemoryLabel:
    """区间内文本未变化时的标签。

    这里是"present 且确定不更新"的唯一来源，也是三态里最容易过度声明的一格：action
    标签不变并不严格等价于外观不变。因此默认 ``within_segment_hard_null=False``，
    只有显式开启时才把"深入段内部（距最近变化点 >= boundary_margin）"的帧记为
    ``verified_hard_null``；靠近变化点的帧一律 ``masked_unknown``，因为 MGIT 的边界帧号
    是人工标注，真实外观变化可能早于或晚于标注帧。
    """

    if not within_segment_hard_null:
        return _masked(current_frame, input_state, "within_segment_hard_null_disabled")
    distance = _distance_to_boundary(boundaries, current_frame)
    if distance < boundary_margin:
        return _masked(current_frame, input_state, f"within_boundary_margin({distance})")
    return FrameMemoryLabel(
        frame_id=current_frame,
        state=MEMORY_STATE_VERIFIED_HARD_NULL,
        memory_update=None,
        input_state=input_state,
        reason="within_segment_state_text_unchanged",
    )
