"""CognitiveTrack v4 的帧级数据协议。

本协议将“工程执行是否成功”和“目标是否存在”放在两个独立命名空间中：
模型输出无法解析时，``execution.status`` 必须是 ``parse_error`` 且
``prediction`` 为 ``None``，绝不能写成一个虚假的 ``absent`` 预测。
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .bbox import BBoxXYWH, validate_xywh
from .enums import (
    ExecutionStatus,
    GroundTruthPresence,
    IdentityMatch,
    Localizability,
    TargetPresence,
)
from .exceptions import ProtocolValidationError

SCHEMA_VERSION = "cogtrack.v4"


@dataclass(frozen=True)
class GroundTruth:
    """帧级真值；只允许可验证的 ``present/absent`` 二分类。"""

    target_presence: GroundTruthPresence
    bbox_xywh: Optional[BBoxXYWH]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_presence", GroundTruthPresence(self.target_presence))
        if self.bbox_xywh is not None:
            object.__setattr__(self, "bbox_xywh", validate_xywh(self.bbox_xywh))
        if self.target_presence is GroundTruthPresence.PRESENT and self.bbox_xywh is None:
            raise ProtocolValidationError("present 真值必须包含 bbox_xywh")
        if self.target_presence is GroundTruthPresence.ABSENT and self.bbox_xywh is not None:
            raise ProtocolValidationError("absent 真值的 bbox_xywh 必须为 None")


@dataclass(frozen=True)
class ExecutionInfo:
    """一次帧级执行的工程状态和耗时。"""

    status: ExecutionStatus
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        if self.latency_ms is not None:
            latency = float(self.latency_ms)
            if not math.isfinite(latency) or latency < 0.0:
                raise ProtocolValidationError("latency_ms 必须是非负有限数值")
            object.__setattr__(self, "latency_ms", latency)
        if self.status is ExecutionStatus.OK and (self.error_type or self.error_message):
            raise ProtocolValidationError("execution.status=ok 时不能携带错误信息")

    @classmethod
    def success(cls, latency_ms: Optional[float] = None) -> "ExecutionInfo":
        """构造成功状态。"""

        return cls(status=ExecutionStatus.OK, latency_ms=latency_ms)

    @classmethod
    def failure(
        cls,
        status: ExecutionStatus,
        error: BaseException,
        latency_ms: Optional[float] = None,
    ) -> "ExecutionInfo":
        """由异常构造失败状态；不生成任何目标存在性预测。"""

        status = ExecutionStatus(status)
        if status is ExecutionStatus.OK:
            raise ProtocolValidationError("failure() 不能使用 ok 状态")
        return cls(
            status=status,
            latency_ms=latency_ms,
            error_type=type(error).__name__,
            error_message=str(error),
        )


@dataclass(frozen=True)
class Prediction:
    """模型/跟踪器的结构化认知预测，框始终为像素 ``xywh``。

    v4 的 VLM 主决策仍是 ``target_status + bbox``，第三字段
    ``memory_update`` 只是可空的语义记忆提议。``identity_match`` 和
    ``localizability`` 是由二分类状态确定性派生的内部兼容字段，不是模型输出，
    也不参与 SFT/GRPO 监督。
    """

    target_presence: TargetPresence
    identity_match: IdentityMatch
    localizability: Localizability
    bbox_xywh: Optional[BBoxXYWH]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_presence", TargetPresence(self.target_presence))
        object.__setattr__(self, "identity_match", IdentityMatch(self.identity_match))
        object.__setattr__(self, "localizability", Localizability(self.localizability))
        if self.bbox_xywh is not None:
            object.__setattr__(self, "bbox_xywh", validate_xywh(self.bbox_xywh))

        if self.localizability is Localizability.LOCALIZABLE and self.bbox_xywh is None:
            raise ProtocolValidationError("localizable 预测必须包含 bbox_xywh")
        if self.localizability is not Localizability.LOCALIZABLE and self.bbox_xywh is not None:
            raise ProtocolValidationError("unlocalizable/not_applicable 预测不能包含 bbox_xywh")
        if self.target_presence is TargetPresence.ABSENT:
            if self.localizability is not Localizability.NOT_APPLICABLE:
                raise ProtocolValidationError("absent 预测的 localizability 必须为 not_applicable")
            if self.bbox_xywh is not None:
                raise ProtocolValidationError("absent 预测不能包含 bbox_xywh")
        if self.target_presence is TargetPresence.UNCERTAIN and self.bbox_xywh is not None:
            raise ProtocolValidationError("uncertain 预测不能输出用于评测的 bbox_xywh")


@dataclass(frozen=True)
class CognitionInfo:
    """可解释文本和记忆更新信息；不参与基础 bbox 指标计算。"""

    target_text: str = ""
    reasoning: str = ""
    memory_update_proposal: Optional[str] = None
    memory_updated: bool = False
    memory_update_reason: str = ""

    def __post_init__(self) -> None:
        for field_name in ("target_text", "reasoning", "memory_update_reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ProtocolValidationError(f"{field_name} 必须是字符串")
        proposal = self.memory_update_proposal
        if proposal is not None:
            if not isinstance(proposal, str) or not proposal.strip():
                raise ProtocolValidationError("memory_update_proposal 必须是非空字符串或 None")
            object.__setattr__(self, "memory_update_proposal", proposal.strip())


@dataclass(frozen=True)
class ContextInfo:
    """复现实验所需的模型、Prompt 和参考帧上下文。"""

    model_name: str = ""
    prompt_name: str = ""
    prompt_version: str = ""
    reference_frames: Tuple[int, ...] = ()
    generation_config: Mapping[str, Any] = field(default_factory=dict)
    #: 实际输入/输出 token 数用于分析 ``memory_update=null`` 快路径与非空
    #: 慢路径的生成开销；拿不到 usage 的后端保持 None。
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    #: 本帧使用的 bbox 坐标协议。同一 Prompt 名下换协议会改变数值结果，因此
    #: 必须逐帧留痕，否则事后无法判断某个 run 的框该怎么解释。
    bbox_protocol: str = ""
    #: processor 实际喂给模型的当前帧尺寸 ``(width, height)``；norm1000 协议下为 None。
    model_image_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        frames = tuple(int(frame_id) for frame_id in self.reference_frames)
        if any(frame_id < 0 for frame_id in frames):
            raise ProtocolValidationError("reference_frames 不能包含负帧号")
        object.__setattr__(self, "reference_frames", frames)
        object.__setattr__(self, "generation_config", dict(self.generation_config))
        for field_name in ("prompt_tokens", "generated_tokens"):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ProtocolValidationError(f"{field_name} 必须是非负整数或 None")


@dataclass(frozen=True)
class FrameResult:
    """可直接写入 JSONL 的 CognitiveTrack v4 帧结果。"""

    sequence: str
    frame_id: int
    is_observation_frame: bool
    execution: ExecutionInfo
    prediction: Optional[Prediction]
    cognition: CognitionInfo = field(default_factory=CognitionInfo)
    context: ContextInfo = field(default_factory=ContextInfo)
    schema_version: str = SCHEMA_VERSION
    raw_model_response: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProtocolValidationError(f"不支持 schema_version={self.schema_version!r}，期望 {SCHEMA_VERSION!r}")
        if not isinstance(self.sequence, str) or not self.sequence.strip():
            raise ProtocolValidationError("sequence 必须是非空字符串")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ProtocolValidationError("frame_id 必须是非负整数")
        if self.execution.status is ExecutionStatus.OK and self.prediction is None:
            raise ProtocolValidationError("execution.status=ok 时必须包含 prediction")
        if self.execution.status is not ExecutionStatus.OK and self.prediction is not None:
            raise ProtocolValidationError("执行失败/跳过时 prediction 必须为 None，不能伪造 absent 等语义预测")

    def to_dict(self, include_raw_response: bool = True) -> Dict[str, Any]:
        """转换成 JSON 可序列化字典，枚举统一写成字符串值。"""

        data = asdict(self)

        def convert(value: Any) -> Any:
            if hasattr(value, "value") and isinstance(value.value, str):
                return value.value
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        output = convert(data)
        if not include_raw_response:
            output.pop("raw_model_response", None)
        return output
