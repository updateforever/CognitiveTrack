"""按同一视频内真实可见/消失区间规划 presence 训练 case。

该模块只读取 ``Sequence`` 标注，不读取图像。它先为每条序列确定总 case 数，再在
全数据集层面分配 absent 配额，因此即使很多序列从未发生目标消失，最终比例也不会
被静默稀释。负样本仅来自该序列自己的真实 absent 帧。
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping
from typing import Sequence as TypingSequence

import numpy as np

from pytracking.evaluation.data import Sequence

REFERENCE_POLICY_SAMPLED_PRIOR = "sampled_prior_present"
REFERENCE_POLICY_FIXED_ANCHOR = "fixed_identity_anchor"
REFERENCE_POLICIES = frozenset(
    {REFERENCE_POLICY_SAMPLED_PRIOR, REFERENCE_POLICY_FIXED_ANCHOR}
)


def sequence_sampling_key(sequence: Sequence) -> str:
    """返回跨数据集不冲突的稳定序列键。"""

    return f"{sequence.dataset}::{sequence.name}"


@dataclass(frozen=True)
class SequenceCasePlan:
    """单序列最终选中的有序帧及状态统计。"""

    dataset: str
    sequence: str
    anchor_frame_id: int
    frame_ids: tuple[int, ...]
    reference_frame_ids: tuple[int, ...]
    present_count: int
    absent_count: int
    absent_run_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_ids"] = list(self.frame_ids)
        payload["reference_frame_ids"] = list(self.reference_frame_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SequenceCasePlan":
        """从发布包 JSON 恢复单序列计划，并校验自洽性。"""

        frame_ids = tuple(int(value) for value in payload["frame_ids"])
        if not frame_ids or any(value < 0 for value in frame_ids):
            raise ValueError("sampling plan 的 frame_ids 必须是非空非负整数列表")
        if tuple(sorted(frame_ids)) != frame_ids:
            raise ValueError("sampling plan 的 frame_ids 必须按时间非递减排列")
        raw_references = payload.get("reference_frame_ids")
        if raw_references is None:
            # 兼容旧版每序列单 anchor 计划；新计划始终写逐 case reference。
            reference_frame_ids = (int(payload["anchor_frame_id"]),) * len(frame_ids)
        else:
            reference_frame_ids = tuple(int(value) for value in raw_references)
        if len(reference_frame_ids) != len(frame_ids):
            raise ValueError("sampling plan 的 reference_frame_ids 必须与 frame_ids 等长")
        if any(
            reference < 0 or reference >= current
            for reference, current in zip(reference_frame_ids, frame_ids, strict=True)
        ):
            raise ValueError("sampling plan 的每个 reference_frame_id 必须严格早于 current frame")
        if len(set(zip(reference_frame_ids, frame_ids, strict=True))) != len(frame_ids):
            raise ValueError("sampling plan 不能包含重复的 reference/current pair")
        present_count = int(payload["present_count"])
        absent_count = int(payload["absent_count"])
        if present_count < 0 or absent_count < 0 or present_count + absent_count != len(frame_ids):
            raise ValueError("sampling plan 的 present/absent 计数与 frame_ids 不一致")
        anchor_frame_id = int(payload["anchor_frame_id"])
        if anchor_frame_id < 0:
            raise ValueError("sampling plan 的 anchor_frame_id 不能为负数")
        dataset = str(payload["dataset"]).strip()
        sequence = str(payload["sequence"]).strip()
        if not dataset or not sequence:
            raise ValueError("sampling plan 的 dataset/sequence 不能为空")
        return cls(
            dataset=dataset,
            sequence=sequence,
            anchor_frame_id=anchor_frame_id,
            frame_ids=frame_ids,
            reference_frame_ids=reference_frame_ids,
            present_count=present_count,
            absent_count=absent_count,
            absent_run_count=int(payload["absent_run_count"]),
        )


@dataclass(frozen=True)
class TemporalCaseSamplingPlan:
    """可直接写入 JSON 的全局采样计划。"""

    seed: int
    requested_absent_ratio: float
    actual_absent_ratio: float
    max_cases_per_sequence: int
    sequence_count: int
    case_count: int
    present_count: int
    absent_count: int
    absent_run_count: int
    sequences: tuple[SequenceCasePlan, ...]
    reference_policy: str = REFERENCE_POLICY_SAMPLED_PRIOR

    def __post_init__(self) -> None:
        if self.reference_policy not in REFERENCE_POLICIES:
            raise ValueError(
                f"reference_policy 必须是 {sorted(REFERENCE_POLICIES)} 之一"
            )
        if self.reference_policy == REFERENCE_POLICY_FIXED_ANCHOR:
            for item in self.sequences:
                if len(set(item.frame_ids)) != len(item.frame_ids):
                    raise ValueError("fixed_identity_anchor plan 不能复用同一个 current frame")
                if any(
                    reference != item.anchor_frame_id
                    for reference in item.reference_frame_ids
                ):
                    raise ValueError(
                        "fixed_identity_anchor plan 的所有 reference 必须等于永久 anchor"
                    )

    @property
    def frame_ids_by_sequence(self) -> Mapping[str, tuple[int, ...]]:
        return {
            f"{item.dataset}::{item.sequence}": item.frame_ids
            for item in self.sequences
        }

    @property
    def anchor_frame_ids_by_sequence(self) -> Mapping[str, int]:
        return {
            f"{item.dataset}::{item.sequence}": item.anchor_frame_id
            for item in self.sequences
        }

    @property
    def reference_frame_ids_by_sequence(self) -> Mapping[str, tuple[int, ...]]:
        return {
            f"{item.dataset}::{item.sequence}": item.reference_frame_ids
            for item in self.sequences
        }

    def to_dict(self, *, include_frame_ids: bool = True) -> dict[str, Any]:
        payload = {
            "seed": self.seed,
            "requested_absent_ratio": self.requested_absent_ratio,
            "actual_absent_ratio": self.actual_absent_ratio,
            "max_cases_per_sequence": self.max_cases_per_sequence,
            "sequence_count": self.sequence_count,
            "case_count": self.case_count,
            "present_count": self.present_count,
            "absent_count": self.absent_count,
            "absent_run_count": self.absent_run_count,
        }
        # 默认策略对应已发布的旧 Stage-1 plan。省略该字段可保持旧 JSON 与 checksum
        # 字节级兼容；visual-v5 的固定锚点策略必须显式写入并在重放时校验。
        if self.reference_policy != REFERENCE_POLICY_SAMPLED_PRIOR:
            payload["reference_policy"] = self.reference_policy
        if include_frame_ids:
            payload["sequences"] = [item.to_dict() for item in self.sequences]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalCaseSamplingPlan":
        """从 ``sampling_plan.json`` 恢复可执行计划。

        发布包里的计划是跨机器重建正式数据的科学输入，因此这里拒绝缺少逐序列
        frame IDs 的摘要版计划，也拒绝重复序列或汇总计数不一致。
        """

        raw_sequences = payload.get("sequences")
        if not isinstance(raw_sequences, list) or not raw_sequences:
            raise ValueError("sampling plan 缺少可重放的 sequences 明细")
        sequences = tuple(SequenceCasePlan.from_dict(item) for item in raw_sequences)
        keys = [(item.dataset, item.sequence) for item in sequences]
        if len(set(keys)) != len(keys):
            raise ValueError("sampling plan 含重复 dataset::sequence")

        plan = cls(
            seed=int(payload["seed"]),
            requested_absent_ratio=float(payload["requested_absent_ratio"]),
            actual_absent_ratio=float(payload["actual_absent_ratio"]),
            max_cases_per_sequence=int(payload["max_cases_per_sequence"]),
            sequence_count=int(payload["sequence_count"]),
            case_count=int(payload["case_count"]),
            present_count=int(payload["present_count"]),
            absent_count=int(payload["absent_count"]),
            absent_run_count=int(payload["absent_run_count"]),
            sequences=sequences,
            reference_policy=str(
                payload.get("reference_policy", REFERENCE_POLICY_SAMPLED_PRIOR)
            ),
        )
        if plan.sequence_count != len(sequences):
            raise ValueError("sampling plan 的 sequence_count 与明细数量不一致")
        if plan.case_count != sum(len(item.frame_ids) for item in sequences):
            raise ValueError("sampling plan 的 case_count 与明细数量不一致")
        if plan.present_count != sum(item.present_count for item in sequences):
            raise ValueError("sampling plan 的 present_count 与明细不一致")
        if plan.absent_count != sum(item.absent_count for item in sequences):
            raise ValueError("sampling plan 的 absent_count 与明细不一致")
        if plan.case_count != plan.present_count + plan.absent_count:
            raise ValueError("sampling plan 的正负样本总数不一致")
        actual_ratio = plan.absent_count / plan.case_count
        if abs(actual_ratio - plan.actual_absent_ratio) > 1.0 / plan.case_count:
            raise ValueError("sampling plan 的 actual_absent_ratio 与计数不一致")
        return plan


@dataclass(frozen=True)
class _StatePool:
    sequence: Sequence
    anchor_frame_id: int
    reference_ids: tuple[int, ...]
    present_ids: tuple[int, ...]
    absent_runs: tuple[tuple[int, ...], ...]
    target_count: int
    min_absent_count: int
    max_absent_count: int

    @property
    def absent_ids(self) -> tuple[int, ...]:
        return tuple(frame_id for run in self.absent_runs for frame_id in run)


def _is_valid_bbox(sequence: Sequence, frame_id: int) -> bool:
    if sequence.ground_truth_rect is None:
        return False
    values = np.asarray(sequence.ground_truth_rect[frame_id], dtype=np.float64).reshape(-1)
    return bool(
        values.size == 4
        and np.all(np.isfinite(values))
        and values[2] > 0
        and values[3] > 0
    )


def _presence(sequence: Sequence, frame_id: int) -> str | None:
    if sequence.target_visible is not None:
        if bool(sequence.target_visible[frame_id]):
            return "present" if _is_valid_bbox(sequence, frame_id) else None
        return "absent"
    return "present" if _is_valid_bbox(sequence, frame_id) else None


def _contiguous_runs(frame_ids: TypingSequence[int]) -> tuple[tuple[int, ...], ...]:
    runs: list[list[int]] = []
    for frame_id in frame_ids:
        if not runs or frame_id != runs[-1][-1] + 1:
            runs.append([frame_id])
        else:
            runs[-1].append(frame_id)
    return tuple(tuple(run) for run in runs)


def _stable_rank(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def _temporal_uniform(frame_ids: TypingSequence[int], count: int) -> list[int]:
    """在整个时间跨度上均匀覆盖，避免长视频的随机样本聚集。"""

    values = list(frame_ids)
    if count <= 0:
        return []
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, num=count)
    return [values[int(round(position))] for position in positions]


def _select_absent(absent_runs: TypingSequence[TypingSequence[int]], count: int) -> list[int]:
    """优先覆盖每个消失区间的首尾，再均匀补充区间内部帧。"""

    if count <= 0:
        return []
    runs = [tuple(run) for run in absent_runs if run]
    if not runs:
        return []

    selected: list[int] = []
    selected_set: set[int] = set()

    def add(values: TypingSequence[int], capacity: int) -> None:
        for frame_id in _temporal_uniform(values, capacity):
            if frame_id not in selected_set:
                selected.append(frame_id)
                selected_set.add(frame_id)

    # 当区间数很多时先在整段视频上均匀选区间，保证 case 多样性。
    chosen_runs = runs
    if len(runs) > count:
        chosen_runs = [runs[index] for index in _temporal_uniform(range(len(runs)), count)]
    add([run[0] for run in chosen_runs], count)
    if len(selected) < count:
        add([run[-1] for run in chosen_runs], count - len(selected))
    if len(selected) < count:
        all_absent = [frame_id for run in runs for frame_id in run if frame_id not in selected_set]
        add(all_absent, count - len(selected))
    return sorted(selected[:count])


def _select_absent_pair_currents(
    sequence_name: str,
    absent_runs: TypingSequence[TypingSequence[int]],
    count: int,
    *,
    reference_frame_ids: TypingSequence[int],
) -> list[int]:
    """选择 absent current；不足时以不同历史 reference 复用真实 absent 帧。"""

    absent_ids = [frame_id for run in absent_runs for frame_id in run]
    selected = _select_absent(absent_runs, min(count, len(absent_ids)))
    used = {frame_id: selected.count(frame_id) for frame_id in selected}
    remaining = count - len(selected)
    while remaining > 0:
        progressed = False
        for frame_id in absent_ids:
            reference_capacity = bisect_left(reference_frame_ids, frame_id)
            if used.get(frame_id, 0) >= reference_capacity:
                continue
            selected.append(frame_id)
            used[frame_id] = used.get(frame_id, 0) + 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError(f"序列 {sequence_name} 的 absent pair 容量不足")
    return sorted(selected)


def _select_present(
    present_ids: TypingSequence[int],
    absent_runs: TypingSequence[TypingSequence[int]],
    count: int,
) -> list[int]:
    """优先保留消失前/重现后的正帧，再用均匀采样覆盖普通跟踪过程。"""

    if count <= 0:
        return []
    present_set = set(present_ids)
    boundary_ids: list[int] = []
    for run in absent_runs:
        for frame_id in (run[0] - 1, run[-1] + 1):
            if frame_id in present_set:
                boundary_ids.append(frame_id)
    boundary_ids = sorted(set(boundary_ids))
    selected = _temporal_uniform(boundary_ids, min(count, len(boundary_ids)))
    selected_set = set(selected)
    if len(selected) < count:
        remainder = [frame_id for frame_id in present_ids if frame_id not in selected_set]
        selected.extend(_temporal_uniform(remainder, count - len(selected)))
    return sorted(selected)


def _select_present_pair_currents(
    sequence_name: str,
    present_ids: TypingSequence[int],
    absent_runs: TypingSequence[TypingSequence[int]],
    count: int,
    *,
    reference_frame_ids: TypingSequence[int],
) -> list[int]:
    """选择 present current；不足时用不同过去 reference 组成不重复 pair。"""

    selected = _select_present(present_ids, absent_runs, min(count, len(present_ids)))
    used = {frame_id: selected.count(frame_id) for frame_id in selected}
    remaining = count - len(selected)
    while remaining > 0:
        progressed = False
        for frame_id in present_ids:
            reference_capacity = bisect_left(reference_frame_ids, frame_id)
            if used.get(frame_id, 0) >= reference_capacity:
                continue
            selected.append(frame_id)
            used[frame_id] = used.get(frame_id, 0) + 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError(f"序列 {sequence_name} 的 present pair 容量不足")
    return sorted(selected)


def _build_pool(
    sequence: Sequence,
    *,
    max_cases_per_sequence: int,
    frame_stride: int,
    reference_policy: str,
) -> _StatePool | None:
    if sequence.ground_truth_rect is None:
        raise ValueError(f"序列 {sequence.name} 缺少 ground_truth_rect")
    anchor_frame_id = next(
        (frame_id for frame_id in range(len(sequence)) if _presence(sequence, frame_id) == "present"),
        None,
    )
    if anchor_frame_id is None:
        raise ValueError(f"序列 {sequence.name} 不包含任何有效 present 初始化帧")

    full_absent_ids: list[int] = []
    reference_ids: list[int] = []
    present_ids: list[int] = []
    # 锚点出现前模型尚未获得目标身份，不能把这些帧当作可监督的 absent case。
    for frame_id in range(anchor_frame_id + 1, len(sequence)):
        state = _presence(sequence, frame_id)
        if state == "absent":
            full_absent_ids.append(frame_id)
        elif state == "present":
            reference_ids.append(frame_id)
            if (frame_id - anchor_frame_id - 1) % frame_stride == 0:
                present_ids.append(frame_id)

    absent_runs = _contiguous_runs(full_absent_ids)
    sampled_runs = tuple(
        tuple(
            frame_id
            for frame_id in run
            if (frame_id - anchor_frame_id - 1) % frame_stride == 0
        )
        for run in absent_runs
    )
    sampled_runs = tuple(run for run in sampled_runs if run)
    absent_count = sum(len(run) for run in sampled_runs)
    eligible_count = len(present_ids) + absent_count
    if eligible_count == 0:
        return None
    target_count = min(max_cases_per_sequence, eligible_count)
    if reference_policy == REFERENCE_POLICY_FIXED_ANCHOR:
        # 永久锚点协议下每个 current 只能出现一次；不能再靠更换 reference 复制 case。
        min_absent = max(0, target_count - len(present_ids))
        max_absent = min(target_count, absent_count)
    else:
        present_seen = 1
        present_pair_capacity = 0
        absent_pair_capacity = 0
        sampled_absent_set = {frame_id for run in sampled_runs for frame_id in run}
        for frame_id in range(anchor_frame_id + 1, len(sequence)):
            state = _presence(sequence, frame_id)
            if state == "present":
                if frame_id in present_ids:
                    present_pair_capacity += present_seen
                present_seen += 1
            elif frame_id in sampled_absent_set:
                absent_pair_capacity += present_seen
        min_absent = max(0, target_count - present_pair_capacity)
        max_absent = min(target_count, absent_pair_capacity)
    return _StatePool(
        sequence=sequence,
        anchor_frame_id=anchor_frame_id,
        reference_ids=tuple(reference_ids),
        present_ids=tuple(present_ids),
        absent_runs=sampled_runs,
        target_count=target_count,
        min_absent_count=min_absent,
        max_absent_count=max_absent,
    )


def _reference_frames(
    sequence: Sequence,
    frame_ids: TypingSequence[int],
    *,
    prior_present_ids: TypingSequence[int],
    seed: int,
) -> tuple[int, ...]:
    """为每个 current frame 选择同序列更早的真实 present reference。

    参考帧在所有更早的有效 present 帧中做确定性选择，因而既覆盖短间隔，也覆盖
    长间隔外观变化；绝不读取 current 或未来帧的 GT。
    """

    references: list[int] = []
    sequence_key = sequence_sampling_key(sequence)
    prior_present_ids = tuple(prior_present_ids)
    current_occurrences: dict[int, int] = {}
    for current_frame_id in frame_ids:
        candidate_count = bisect_left(prior_present_ids, current_frame_id)
        if candidate_count == 0:
            raise ValueError(f"序列 {sequence.name} 的 current {current_frame_id} 之前没有 present reference")
        occurrence = current_occurrences.get(current_frame_id, 0)
        current_occurrences[current_frame_id] = occurrence + 1
        start = _stable_rank(seed, f"{sequence_key}::{current_frame_id}::reference") % candidate_count
        index = (start + occurrence) % candidate_count
        reference = prior_present_ids[index]
        if reference >= current_frame_id:
            raise ValueError(
                f"序列 {sequence.name} 的 reference frame {reference} 不早于 current {current_frame_id}"
            )
        references.append(reference)
    return tuple(references)


def plan_temporal_presence_cases(
    sequences: Iterable[Sequence],
    *,
    max_cases_per_sequence: int = 20,
    absent_ratio: float = 0.3,
    frame_stride: int = 1,
    seed: int = 20260809,
    reference_policy: str = REFERENCE_POLICY_SAMPLED_PRIOR,
) -> TemporalCaseSamplingPlan:
    """生成全局比例受控、按连续状态区间覆盖的确定性帧计划。"""

    if isinstance(max_cases_per_sequence, bool) or max_cases_per_sequence <= 0:
        raise ValueError("max_cases_per_sequence 必须是正整数")
    if not 0 <= absent_ratio < 1:
        raise ValueError("absent_ratio 必须位于 [0,1)")
    if isinstance(frame_stride, bool) or frame_stride <= 0:
        raise ValueError("frame_stride 必须是正整数")
    if reference_policy not in REFERENCE_POLICIES:
        raise ValueError(f"reference_policy 必须是 {sorted(REFERENCE_POLICIES)} 之一")

    pools = [
        pool
        for sequence in sequences
        if (pool := _build_pool(
            sequence,
            max_cases_per_sequence=max_cases_per_sequence,
            frame_stride=frame_stride,
            reference_policy=reference_policy,
        ))
        is not None
    ]
    if not pools:
        raise ValueError("没有可规划的 present/absent 训练 case")

    total_cases = sum(pool.target_count for pool in pools)
    min_absent = sum(pool.min_absent_count for pool in pools)
    max_absent = sum(pool.max_absent_count for pool in pools)
    requested_absent = int(round(total_cases * absent_ratio))
    target_absent = min(max(requested_absent, min_absent), max_absent)
    if target_absent != requested_absent:
        raise ValueError(
            f"无法达到 absent_ratio={absent_ratio:.3f}：请求 {requested_absent}/{total_cases}，"
            f"可行范围为 [{min_absent},{max_absent}]"
        )

    absent_counts = [pool.min_absent_count for pool in pools]
    source_indices: dict[str, list[int]] = {}
    for index, pool in enumerate(pools):
        source_indices.setdefault(pool.sequence.dataset, []).append(index)
    for source, indices in sorted(source_indices.items()):
        source_cases = sum(pools[index].target_count for index in indices)
        source_min = sum(pools[index].min_absent_count for index in indices)
        source_max = sum(pools[index].max_absent_count for index in indices)
        source_target = int(round(source_cases * absent_ratio))
        if not source_min <= source_target <= source_max:
            raise ValueError(
                f"数据源 {source} 无法达到 absent_ratio={absent_ratio:.3f}："
                f"请求 {source_target}/{source_cases}，可行范围为 [{source_min},{source_max}]"
            )
        remaining = source_target - source_min
        order = sorted(
            indices,
            key=lambda index: _stable_rank(seed, sequence_sampling_key(pools[index].sequence)),
        )
        while remaining > 0:
            progressed = False
            for index in order:
                if absent_counts[index] >= pools[index].max_absent_count:
                    continue
                absent_counts[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                raise RuntimeError(f"数据源 {source} 的 absent 配额分配未收敛")

    sequence_plans: list[SequenceCasePlan] = []
    for pool, absent_count in zip(pools, absent_counts, strict=True):
        present_count = pool.target_count - absent_count
        if reference_policy == REFERENCE_POLICY_FIXED_ANCHOR:
            selected_absent = _select_absent(pool.absent_runs, absent_count)
            selected_present = _select_present(
                pool.present_ids,
                pool.absent_runs,
                present_count,
            )
        else:
            selected_absent = _select_absent_pair_currents(
                pool.sequence.name,
                pool.absent_runs,
                absent_count,
                reference_frame_ids=(pool.anchor_frame_id, *pool.reference_ids),
            )
            selected_present = _select_present_pair_currents(
                pool.sequence.name,
                pool.present_ids,
                pool.absent_runs,
                present_count,
                reference_frame_ids=(pool.anchor_frame_id, *pool.reference_ids),
            )
        frame_ids = tuple(sorted((*selected_present, *selected_absent)))
        if len(frame_ids) != pool.target_count:
            raise RuntimeError(f"序列 {pool.sequence.name} 的采样数量与计划不一致")
        sequence_plans.append(
            SequenceCasePlan(
                dataset=pool.sequence.dataset,
                sequence=pool.sequence.name,
                anchor_frame_id=pool.anchor_frame_id,
                frame_ids=frame_ids,
                reference_frame_ids=(
                    (pool.anchor_frame_id,) * len(frame_ids)
                    if reference_policy == REFERENCE_POLICY_FIXED_ANCHOR
                    else _reference_frames(
                        pool.sequence,
                        frame_ids,
                        prior_present_ids=(pool.anchor_frame_id, *pool.reference_ids),
                        seed=seed,
                    )
                ),
                present_count=present_count,
                absent_count=absent_count,
                absent_run_count=len(pool.absent_runs),
            )
        )

    actual_absent = sum(item.absent_count for item in sequence_plans)
    actual_present = sum(item.present_count for item in sequence_plans)
    actual_total = actual_present + actual_absent
    return TemporalCaseSamplingPlan(
        seed=seed,
        requested_absent_ratio=absent_ratio,
        actual_absent_ratio=actual_absent / actual_total,
        max_cases_per_sequence=max_cases_per_sequence,
        sequence_count=len(sequence_plans),
        case_count=actual_total,
        present_count=actual_present,
        absent_count=actual_absent,
        absent_run_count=sum(item.absent_run_count for item in sequence_plans),
        sequences=tuple(sequence_plans),
        reference_policy=reference_policy,
    )


__all__ = [
    "REFERENCE_POLICIES",
    "REFERENCE_POLICY_FIXED_ANCHOR",
    "REFERENCE_POLICY_SAMPLED_PRIOR",
    "SequenceCasePlan",
    "TemporalCaseSamplingPlan",
    "plan_temporal_presence_cases",
    "sequence_sampling_key",
]
