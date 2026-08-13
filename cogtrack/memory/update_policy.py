"""防止错误预测自我强化的记忆更新门控。"""

import re
import threading
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Mapping, Optional

from ..protocol.bbox import BBoxXYWH, bbox_iou_xywh, validate_xywh
from ..protocol.enums import ExecutionStatus, IdentityMatch, TargetPresence
from .bank import MemoryBank
from .records import MemoryKind, MemoryRecord, MemorySource

_NOOP_MEMORY_PATTERNS = (
    re.compile(r"\bno\s+(?:material(?:ly)?\s+|significant\s+|meaningful\s+)?change\b", re.IGNORECASE),
    re.compile(r"\b(?:nothing\s+changed|unchanged|no\s+update|none)\b", re.IGNORECASE),
    re.compile(r"(?:没有|无|未见)(?:明显|显著|重大|实质)?(?:外观|视角|构型|配置|状态)?变化"),
    re.compile(r"(?:无需|不用|不需要)更新"),
)


def _is_noop_semantic_text(text: str) -> bool:
    """识别模型误用字符串表达的“不更新”，防止无信息文本污染记忆。"""

    normalized = " ".join(text.strip().split())
    return any(pattern.search(normalized) for pattern in _NOOP_MEMORY_PATTERNS)


@dataclass(frozen=True)
class MemoryCandidate:
    """一次观测产生的候选记忆，尚未被信任。"""

    kind: MemoryKind
    frame_id: int
    source: MemorySource
    execution_status: ExecutionStatus
    target_presence: TargetPresence
    identity_match: IdentityMatch
    bbox_xywh: Optional[BBoxXYWH] = None
    text: str = ""
    image_ref: str = ""
    image: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "source", MemorySource(self.source))
        object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
        object.__setattr__(self, "target_presence", TargetPresence(self.target_presence))
        object.__setattr__(self, "identity_match", IdentityMatch(self.identity_match))
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id 必须是非负整数")
        if self.bbox_xywh is not None:
            object.__setattr__(self, "bbox_xywh", validate_xywh(self.bbox_xywh))
        if not isinstance(self.text, str) or not isinstance(self.image_ref, str):
            raise TypeError("text 和 image_ref 必须是字符串")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MemoryUpdatePolicyConfig:
    """不依赖 VLM 自报置信度的本地记忆门控配置。"""

    consecutive_positive_confirmations: int = 2
    max_confirmation_gap: int = 30
    min_bbox_iou_consistency: float = 0.0
    min_positive_frame_gap: int = 5
    min_semantic_frame_gap: int = 30
    # 缺省 1 保持旧 tracker 的即时语义写入行为；visual-v5 配置显式设为 2。
    semantic_confirmations: int = 1
    max_semantic_confirmation_gap: int = 300
    min_semantic_text_similarity: float = 0.35

    def __post_init__(self) -> None:
        bbox_iou = float(self.min_bbox_iou_consistency)
        if not 0.0 <= bbox_iou <= 1.0:
            raise ValueError("min_bbox_iou_consistency 必须位于 [0, 1]")
        text_similarity = float(self.min_semantic_text_similarity)
        if not 0.0 <= text_similarity <= 1.0:
            raise ValueError("min_semantic_text_similarity 必须位于 [0, 1]")
        for field_name in (
            "consecutive_positive_confirmations",
            "max_confirmation_gap",
            "min_positive_frame_gap",
            "min_semantic_frame_gap",
            "semantic_confirmations",
            "max_semantic_confirmation_gap",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")
        if self.consecutive_positive_confirmations < 1:
            raise ValueError("consecutive_positive_confirmations 至少为 1")
        if self.semantic_confirmations < 1:
            raise ValueError("semantic_confirmations 至少为 1")


@dataclass(frozen=True)
class MemoryUpdateDecision:
    """门控结果及可审计原因。"""

    accepted: bool
    reason: str
    confirmations: int = 0
    record: Optional[MemoryRecord] = None
    evicted_record_id: Optional[str] = None


class GatedMemoryUpdatePolicy:
    """仅将多帧一致、身份可信的预测写入长期记忆。

    初始化真值应直接构造成 ``IdentityAnchor``，不要经由本类覆盖。正记忆默认
    需要连续两次 ``present + same``；负记忆必须是 ``different``；
    semantic 记忆还要求非空文本。任何执行错误都会中断正记忆确认序列。
    """

    def __init__(self, config: Optional[MemoryUpdatePolicyConfig] = None) -> None:
        self.config = config or MemoryUpdatePolicyConfig()
        self._positive_streak = 0
        self._last_candidate_frame: Optional[int] = None
        self._last_candidate_bbox: Optional[BBoxXYWH] = None
        self._last_positive_update_frame: Optional[int] = None
        self._last_semantic_update_frame: Optional[int] = None
        self._pending_semantic_frame: Optional[int] = None
        self._pending_semantic_text = ""
        self._semantic_streak = 0
        self._record_counter = 0
        self._lock = threading.RLock()

    def reset_pending(self) -> None:
        """在 absent、uncertain 或执行错误后中止所有尚未完成的跨帧确认。"""

        with self._lock:
            self._positive_streak = 0
            self._last_candidate_frame = None
            self._last_candidate_bbox = None
            self._reset_semantic_pending_unlocked()

    def _reset_semantic_pending_unlocked(self) -> None:
        self._pending_semantic_frame = None
        self._pending_semantic_text = ""
        self._semantic_streak = 0

    def reset_semantic_pending(self) -> None:
        """当模型本帧返回 null/非法提议时打断语义变化连续确认。"""

        with self._lock:
            self._reset_semantic_pending_unlocked()

    @staticmethod
    def _semantic_similarity(first: str, second: str) -> float:
        """计算轻量、多语言可用的提议相似度，不引入额外模型推理。"""

        normalized_first = " ".join(first.casefold().split())
        normalized_second = " ".join(second.casefold().split())
        if not normalized_first or not normalized_second:
            return 0.0
        return SequenceMatcher(None, normalized_first, normalized_second).ratio()

    def _reject(self, reason: str, confirmations: int | None = None) -> MemoryUpdateDecision:
        count = self._positive_streak if confirmations is None else confirmations
        return MemoryUpdateDecision(False, reason, count)

    def _new_record(self, candidate: MemoryCandidate) -> MemoryRecord:
        self._record_counter += 1
        record_id = f"{candidate.kind.value}-{candidate.frame_id:08d}-{self._record_counter:06d}"
        metadata = dict(candidate.metadata)
        return MemoryRecord(
            record_id=record_id,
            kind=candidate.kind,
            frame_id=candidate.frame_id,
            source=candidate.source,
            bbox_xywh=candidate.bbox_xywh,
            text=candidate.text.strip(),
            image_ref=candidate.image_ref,
            image=candidate.image,
            metadata=metadata,
        )

    def evaluate(self, candidate: MemoryCandidate) -> MemoryUpdateDecision:
        """只评估候选，不修改 ``MemoryBank``。"""

        with self._lock:
            if candidate.source is MemorySource.INITIAL_GROUND_TRUTH:
                return self._reject("初始化真值必须写入永久 IdentityAnchor，而非动态记忆")
            if candidate.execution_status is not ExecutionStatus.OK:
                self.reset_pending()
                return self._reject("本帧执行未成功，禁止写入记忆")

            if candidate.kind is MemoryKind.NEGATIVE:
                if candidate.identity_match is not IdentityMatch.DIFFERENT:
                    return self._reject("负记忆只接受 identity_match=different 的候选")
                return MemoryUpdateDecision(True, "模型判定为 different 的干扰物证据", 1, self._new_record(candidate))

            if candidate.target_presence is not TargetPresence.PRESENT:
                self.reset_pending()
                return self._reject("只有 present 观测可更新正/语义记忆")
            if candidate.identity_match is not IdentityMatch.SAME:
                self.reset_pending()
                return self._reject("只有 identity_match=same 的观测可更新正/语义记忆")
            if candidate.kind is MemoryKind.SEMANTIC:
                if not candidate.text.strip():
                    self._reset_semantic_pending_unlocked()
                    return self._reject("语义记忆文本为空")
                if _is_noop_semantic_text(candidate.text):
                    self._reset_semantic_pending_unlocked()
                    return self._reject("模型用文本表达无变化，按 memory_update=null 处理")
                text = " ".join(candidate.text.split())
                within_gap = (
                    self._pending_semantic_frame is not None
                    and candidate.frame_id > self._pending_semantic_frame
                    and candidate.frame_id - self._pending_semantic_frame
                    <= self.config.max_semantic_confirmation_gap
                )
                similarity = (
                    self._semantic_similarity(self._pending_semantic_text, text)
                    if within_gap
                    else 0.0
                )
                if within_gap and similarity >= self.config.min_semantic_text_similarity:
                    self._semantic_streak += 1
                else:
                    self._semantic_streak = 1
                self._pending_semantic_frame = candidate.frame_id
                self._pending_semantic_text = text
                if self._semantic_streak < self.config.semantic_confirmations:
                    return self._reject(
                        "等待跨帧相近 memory_update 提议确认"
                        f"（{self._semantic_streak}/{self.config.semantic_confirmations}）",
                        self._semantic_streak,
                    )
                confirmations = self._semantic_streak
                self._reset_semantic_pending_unlocked()
                return MemoryUpdateDecision(
                    True,
                    f"跨帧相近语义提议已确认（similarity={similarity:.3f}）",
                    confirmations,
                    self._new_record(candidate),
                )

            # 正视觉记忆应包含可定位目标，便于后续正确裁剪和 mosaic。
            if candidate.bbox_xywh is None:
                self.reset_pending()
                return self._reject("正视觉记忆必须包含 bbox_xywh")

            is_contiguous = (
                self._last_candidate_frame is not None
                and candidate.frame_id > self._last_candidate_frame
                and candidate.frame_id - self._last_candidate_frame <= self.config.max_confirmation_gap
            )
            if not is_contiguous:
                self._positive_streak = 1
            else:
                if (
                    self.config.min_bbox_iou_consistency > 0.0
                    and self._last_candidate_bbox is not None
                    and bbox_iou_xywh(candidate.bbox_xywh, self._last_candidate_bbox)
                    < self.config.min_bbox_iou_consistency
                ):
                    self._positive_streak = 1
                else:
                    self._positive_streak += 1
            self._last_candidate_frame = candidate.frame_id
            self._last_candidate_bbox = candidate.bbox_xywh

            if self._positive_streak < self.config.consecutive_positive_confirmations:
                return self._reject("等待连续 present + same 身份确认")
            if (
                self._last_positive_update_frame is not None
                and candidate.frame_id - self._last_positive_update_frame < self.config.min_positive_frame_gap
            ):
                return self._reject("距离上次正记忆更新过近")

            self._last_positive_update_frame = candidate.frame_id
            return MemoryUpdateDecision(
                True,
                "连续 present + same 观测",
                self._positive_streak,
                self._new_record(candidate),
            )

    def process(self, bank: MemoryBank, candidate: MemoryCandidate) -> MemoryUpdateDecision:
        """评估并在通过时原子式写入指定记忆库。"""

        if not isinstance(bank, MemoryBank):
            raise TypeError("bank 必须是 MemoryBank")
        with self._lock:
            if candidate.kind is MemoryKind.SEMANTIC:
                normalized = " ".join(candidate.text.casefold().split())
                if any(
                    " ".join(record.text.casefold().split()) == normalized
                    for record in bank.records(MemoryKind.SEMANTIC)
                ):
                    return self._reject("语义记忆与已有记录重复")
                if (
                    self._last_semantic_update_frame is not None
                    and candidate.frame_id - self._last_semantic_update_frame
                    < self.config.min_semantic_frame_gap
                ):
                    return self._reject("距离上次语义记忆更新过近")
            decision = self.evaluate(candidate)
            if not decision.accepted or decision.record is None:
                return decision
            evicted = bank.add(decision.record)
            if candidate.kind is MemoryKind.SEMANTIC:
                self._last_semantic_update_frame = candidate.frame_id
            return MemoryUpdateDecision(
                accepted=True,
                reason=decision.reason,
                confirmations=decision.confirmations,
                record=decision.record,
                evicted_record_id=evicted.record_id if evicted is not None else None,
            )
