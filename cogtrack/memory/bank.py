"""有界、线程安全的认知记忆库。"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .records import IdentityAnchor, MemoryKind, MemoryRecord


@dataclass(frozen=True)
class MemoryBankConfig:
    """各类动态记忆容量；永久身份锚点不计入容量。"""

    positive_capacity: int = 8
    negative_capacity: int = 8
    semantic_capacity: int = 4

    def __post_init__(self) -> None:
        for field_name in ("positive_capacity", "negative_capacity", "semantic_capacity"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")

    def capacity_for(self, kind: MemoryKind) -> int:
        return {
            MemoryKind.POSITIVE: self.positive_capacity,
            MemoryKind.NEGATIVE: self.negative_capacity,
            MemoryKind.SEMANTIC: self.semantic_capacity,
        }[MemoryKind(kind)]


class MemoryBank:
    """保存永久身份锚点和三类有界动态记忆。

    容量满时淘汰最早记录；不使用 VLM 自报置信度排序。小容量线性选择
    不会成为 VLM 推理链路的性能瓶颈。
    """

    def __init__(self, anchor: IdentityAnchor, config: Optional[MemoryBankConfig] = None) -> None:
        if not isinstance(anchor, IdentityAnchor):
            raise TypeError("anchor 必须是 IdentityAnchor")
        self._anchor = anchor
        self.config = config or MemoryBankConfig()
        self._records: Dict[MemoryKind, List[MemoryRecord]] = {kind: [] for kind in MemoryKind}
        self._record_ids = set()
        self._lock = threading.RLock()

    @property
    def anchor(self) -> IdentityAnchor:
        """返回不可替换的初始化身份锚点。"""

        return self._anchor

    def add(self, record: MemoryRecord) -> Optional[MemoryRecord]:
        """写入一条动态记忆，返回被淘汰的记录（若有）。"""

        if not isinstance(record, MemoryRecord):
            raise TypeError("record 必须是 MemoryRecord")
        with self._lock:
            if record.record_id in self._record_ids:
                raise ValueError(f"重复的 memory record_id: {record.record_id}")
            capacity = self.config.capacity_for(record.kind)
            if capacity == 0:
                return record

            collection = self._records[record.kind]
            collection.append(record)
            self._record_ids.add(record.record_id)
            evicted: Optional[MemoryRecord] = None
            if len(collection) > capacity:
                evict_index = min(
                    range(len(collection)),
                    key=lambda index: collection[index].frame_id,
                )
                evicted = collection.pop(evict_index)
                self._record_ids.remove(evicted.record_id)
            return evicted

    def records(self, kind: Optional[MemoryKind] = None) -> Tuple[MemoryRecord, ...]:
        """返回不可变快照；不暴露内部 list 以防绕过门控修改。"""

        with self._lock:
            if kind is not None:
                return tuple(self._records[MemoryKind(kind)])
            all_records: List[MemoryRecord] = []
            for memory_kind in MemoryKind:
                all_records.extend(self._records[memory_kind])
            return tuple(all_records)

    def select_positive(self, limit: int) -> Tuple[MemoryRecord, ...]:
        """选择最近的正记忆，并按时间顺序供 mosaic 使用。"""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit 必须是非负整数")
        with self._lock:
            ranked = sorted(
                self._records[MemoryKind.POSITIVE],
                key=lambda record: record.frame_id,
                reverse=True,
            )[:limit]
            return tuple(sorted(ranked, key=lambda record: record.frame_id))

    def clear_dynamic(self) -> None:
        """清空预测产生的动态记忆；永久 anchor 不受影响。"""

        with self._lock:
            for records in self._records.values():
                records.clear()
            self._record_ids.clear()

    def snapshot(self) -> Dict[str, object]:
        """生成用于调试与复现实验的轻量 JSON 快照。"""

        with self._lock:
            return {
                "anchor": {
                    "frame_id": self._anchor.frame_id,
                    "bbox_xywh": list(self._anchor.bbox_xywh),
                    "target_text": self._anchor.target_text,
                    "image_ref": self._anchor.image_ref,
                    "source": self._anchor.source.value,
                },
                "records": {kind.value: [record.to_dict() for record in self._records[kind]] for kind in MemoryKind},
            }

    def __len__(self) -> int:
        with self._lock:
            return sum(len(records) for records in self._records.values())
