"""认知跟踪记忆的不可变记录类型。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from ..protocol.bbox import BBoxXYWH, validate_xywh


class MemoryKind(str, Enum):
    """动态记忆库的三种用途。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    SEMANTIC = "semantic"


class MemorySource(str, Enum):
    """记忆证据来源，用于防止预测结果冒充真值。"""

    INITIAL_GROUND_TRUTH = "initial_ground_truth"
    VLM_PREDICTION = "vlm_prediction"
    SUTRACK_PREDICTION = "sutrack_prediction"
    HYBRID_VERIFIED = "hybrid_verified"
    MANUAL = "manual"


def _validate_frame_id(frame_id: int) -> int:
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id 必须是非负整数")
    return frame_id


@dataclass(frozen=True)
class IdentityAnchor:
    """由初始化真值创建、整个序列中永久不可修改的身份锚点。"""

    frame_id: int
    bbox_xywh: BBoxXYWH
    target_text: str = ""
    image_ref: str = ""
    # 在线 tracker 可暂存 PIL/numpy 图像；序列化时只记录 image_ref。
    image: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _validate_frame_id(self.frame_id))
        object.__setattr__(self, "bbox_xywh", validate_xywh(self.bbox_xywh))
        if not isinstance(self.target_text, str) or not isinstance(self.image_ref, str):
            raise TypeError("target_text 和 image_ref 必须是字符串")

    @property
    def source(self) -> MemorySource:
        """身份锚点的来源固定为初始化真值。"""

        return MemorySource.INITIAL_GROUND_TRUTH


@dataclass(frozen=True)
class MemoryRecord:
    """一条带来源和帧号的动态记忆。

    记录不保存 VLM 自报置信度。动态记忆的可信性来自本地连续确认、状态
    一致性和来源审计，容量淘汰按时间执行。
    """

    record_id: str
    kind: MemoryKind
    frame_id: int
    source: MemorySource
    bbox_xywh: Optional[BBoxXYWH] = None
    text: str = ""
    image_ref: str = ""
    image: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id 必须是非空字符串")
        object.__setattr__(self, "kind", MemoryKind(self.kind))
        object.__setattr__(self, "source", MemorySource(self.source))
        object.__setattr__(self, "frame_id", _validate_frame_id(self.frame_id))
        if self.bbox_xywh is not None:
            object.__setattr__(self, "bbox_xywh", validate_xywh(self.bbox_xywh))
        if not isinstance(self.text, str) or not isinstance(self.image_ref, str):
            raise TypeError("text 和 image_ref 必须是字符串")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """返回可写入 JSON 的记录；不序列化可能很大的内存图像。"""

        return {
            "record_id": self.record_id,
            "kind": self.kind.value,
            "frame_id": self.frame_id,
            "source": self.source.value,
            "bbox_xywh": list(self.bbox_xywh) if self.bbox_xywh is not None else None,
            "text": self.text,
            "image_ref": self.image_ref,
            "metadata": dict(self.metadata),
        }
