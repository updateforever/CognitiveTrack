"""面向长时跟踪的轻量内部状态机。

这里的 ``confirmed/uncertain/searching`` 是 tracker 内部控制阶段，不是数据集
真值标签，也不会恢复旧工程中缺少监督的六分类状态。状态机只根据成功解析
的认知预测推进；模型/解析错误不会被当成目标消失。
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from ..protocol.bbox import BBoxXYWH, validate_xywh
from ..protocol.enums import ExecutionStatus, IdentityMatch, TargetPresence
from ..protocol.schema import Prediction


class CognitivePhase(str, Enum):
    """跟踪控制器当前所处的内部阶段。"""

    UNINITIALIZED = "uninitialized"
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    SEARCHING = "searching"


@dataclass(frozen=True)
class StateMachineConfig:
    """基于离散判断的状态转换配置。"""

    confirmations_to_recover: int = 1
    uncertain_frames_to_search: int = 3

    def __post_init__(self) -> None:
        for field_name in ("confirmations_to_recover", "uncertain_frames_to_search"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} 必须是正整数")


@dataclass(frozen=True)
class CognitiveState:
    """可写入调试日志的完整状态快照。"""

    phase: CognitivePhase = CognitivePhase.UNINITIALIZED
    frame_id: int = -1
    last_seen_frame: Optional[int] = None
    last_trusted_bbox: Optional[BBoxXYWH] = None
    absent_duration: int = 0
    uncertain_duration: int = 0
    error_duration: int = 0
    consecutive_present: int = 0


@dataclass(frozen=True)
class StateTransition:
    """一次状态转换及其原因，便于排查错误漂移。"""

    previous: CognitiveState
    current: CognitiveState
    reason: str


class CognitiveStateMachine:
    """维护 last_seen、缺失时长和全局搜索阶段。"""

    def __init__(self, config: Optional[StateMachineConfig] = None) -> None:
        self.config = config or StateMachineConfig()
        self._state = CognitiveState()

    @property
    def state(self) -> CognitiveState:
        return self._state

    def initialize(self, frame_id: int, bbox_xywh: BBoxXYWH) -> StateTransition:
        """用首帧真值初始化；同一实例只能初始化一次。"""

        if self._state.phase is not CognitivePhase.UNINITIALIZED:
            raise RuntimeError("CognitiveStateMachine 已初始化，不能覆盖身份状态")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id 必须是非负整数")
        bbox = validate_xywh(bbox_xywh)
        previous = self._state
        self._state = CognitiveState(
            phase=CognitivePhase.CONFIRMED,
            frame_id=frame_id,
            last_seen_frame=frame_id,
            last_trusted_bbox=bbox,
            consecutive_present=1,
        )
        return StateTransition(previous, self._state, "使用初始化真值建立身份锚点")

    def _validate_next_frame(self, frame_id: int) -> None:
        if self._state.phase is CognitivePhase.UNINITIALIZED:
            raise RuntimeError("必须先调用 initialize()")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError("frame_id 必须是整数")
        if frame_id <= self._state.frame_id:
            raise ValueError(f"状态机要求帧号严格递增：上一帧 {self._state.frame_id}，当前 {frame_id}")

    def step(
        self,
        frame_id: int,
        execution_status: ExecutionStatus,
        prediction: Optional[Prediction],
    ) -> StateTransition:
        """根据一次帧级执行推进内部状态。"""

        self._validate_next_frame(frame_id)
        status = ExecutionStatus(execution_status)
        previous = self._state

        if status is ExecutionStatus.SKIPPED:
            if prediction is not None:
                raise ValueError("跳过认知观测时 prediction 必须为 None")
            # 稀疏执行中的正常跳帧不提供新语义证据，因此只推进帧号，完整保留
            # 上一次可信状态；它既不是模型错误，也不是目标 absent。
            self._state = replace(previous, frame_id=frame_id)
            return StateTransition(previous, self._state, "本帧未执行认知观测，保持原状态")

        if status is not ExecutionStatus.OK:
            if prediction is not None:
                raise ValueError("执行失败时 prediction 必须为 None")
            # 工程错误只增加 error_duration，不能增加 absent_duration。
            self._state = replace(
                previous,
                phase=CognitivePhase.UNCERTAIN,
                frame_id=frame_id,
                uncertain_duration=previous.uncertain_duration + 1,
                error_duration=previous.error_duration + 1,
                consecutive_present=0,
            )
            return StateTransition(previous, self._state, f"执行状态为 {status.value}，不作 absent 判断")

        if prediction is None:
            raise ValueError("execution_status=ok 时 prediction 不能为空")

        trusted_present = (
            prediction.target_presence is TargetPresence.PRESENT
            and prediction.identity_match is IdentityMatch.SAME
            and prediction.bbox_xywh is not None
        )
        if trusted_present:
            consecutive = previous.consecutive_present + 1
            phase = (
                CognitivePhase.CONFIRMED
                if consecutive >= self.config.confirmations_to_recover
                else CognitivePhase.UNCERTAIN
            )
            self._state = CognitiveState(
                phase=phase,
                frame_id=frame_id,
                last_seen_frame=frame_id,
                last_trusted_bbox=prediction.bbox_xywh,
                absent_duration=0,
                uncertain_duration=0,
                error_duration=0,
                consecutive_present=consecutive,
            )
            return StateTransition(previous, self._state, "present + same 且 bbox 合法，确认目标身份")

        if prediction.target_presence is TargetPresence.ABSENT:
            self._state = CognitiveState(
                phase=CognitivePhase.SEARCHING,
                frame_id=frame_id,
                last_seen_frame=previous.last_seen_frame,
                last_trusted_bbox=previous.last_trusted_bbox,
                absent_duration=previous.absent_duration + 1,
                uncertain_duration=0,
                error_duration=0,
                consecutive_present=0,
            )
            return StateTransition(previous, self._state, "模型判定 absent，进入全图搜索")

        uncertain_duration = previous.uncertain_duration + 1
        phase = (
            CognitivePhase.SEARCHING
            if uncertain_duration >= self.config.uncertain_frames_to_search
            else CognitivePhase.UNCERTAIN
        )
        self._state = CognitiveState(
            phase=phase,
            frame_id=frame_id,
            last_seen_frame=previous.last_seen_frame,
            last_trusted_bbox=previous.last_trusted_bbox,
            absent_duration=previous.absent_duration,
            uncertain_duration=uncertain_duration,
            error_duration=0,
            consecutive_present=0,
        )
        return StateTransition(previous, self._state, "存在性或身份证据不足")
