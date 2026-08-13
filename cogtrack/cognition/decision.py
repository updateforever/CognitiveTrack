"""把原始 VLM 响应转换成标准帧结果，并同步内部状态机。"""

from dataclasses import dataclass
from typing import Optional, Tuple

from ..prompts.common import PromptSpec
from ..protocol.enums import ExecutionStatus
from ..protocol.exceptions import ModelOutputParseError
from ..protocol.schema import (
    CognitionInfo,
    ContextInfo,
    ExecutionInfo,
    FrameResult,
    Prediction,
)
from ..vlm.base import GenerationConfig, VLMResponse
from ..vlm.parser import parse_tracking_output
from .state_machine import CognitiveStateMachine, StateTransition


@dataclass(frozen=True)
class CognitiveDecisionResult:
    """外部结果和可选内部状态转换。"""

    frame_result: FrameResult
    transition: Optional[StateTransition]


class CognitiveDecisionEngine:
    """集中处理严格解析与 execution/prediction 解耦规则。"""

    def __init__(self, state_machine: Optional[CognitiveStateMachine] = None) -> None:
        self.state_machine = state_machine

    def _transition(
        self,
        frame_id: int,
        status: ExecutionStatus,
        prediction: Optional[Prediction],
    ) -> Optional[StateTransition]:
        if self.state_machine is None:
            return None
        return self.state_machine.step(frame_id, status, prediction)

    def from_vlm_response(
        self,
        *,
        sequence: str,
        frame_id: int,
        image_width: int,
        image_height: int,
        is_observation_frame: bool,
        response: VLMResponse,
        prompt: PromptSpec,
        reference_frames: Tuple[int, ...] = (),
        generation_config: Optional[GenerationConfig] = None,
    ) -> CognitiveDecisionResult:
        """严格解析成功响应；失败时产生 parse_error 而不是 absent。"""

        context = ContextInfo(
            model_name=response.model_name,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            reference_frames=reference_frames,
            generation_config=(generation_config or GenerationConfig()).to_dict(),
            prompt_tokens=response.prompt_tokens,
            generated_tokens=response.generated_tokens,
            bbox_protocol=prompt.bbox_protocol,
            model_image_size=response.current_frame_size(),
        )
        try:
            parsed = parse_tracking_output(
                response.text,
                image_width,
                image_height,
                bbox_protocol=prompt.bbox_protocol,
                # 模型看当前帧用的尺寸由后端如实上报；qwen_abs_pixel 协议下缺失会
                # 在解析层直接报 parse_error，不会退化成带偏移的“正常”预测。
                model_image_size=response.current_frame_size(),
                require_memory_update=prompt.include_memory_update,
                # 在线推理分层校验：第三字段非法只拒绝记忆，不丢弃合法的核心跟踪
                # 判断。训练数据验收会显式启用 strict_memory_update。
                strict_memory_update=False,
            )
        except ModelOutputParseError as error:
            execution = ExecutionInfo.failure(
                ExecutionStatus.PARSE_ERROR,
                error,
                latency_ms=response.latency_ms,
            )
            frame_result = FrameResult(
                sequence=sequence,
                frame_id=frame_id,
                is_observation_frame=is_observation_frame,
                execution=execution,
                prediction=None,
                cognition=CognitionInfo(),
                context=context,
                raw_model_response=response.text,
            )
            transition = self._transition(frame_id, ExecutionStatus.PARSE_ERROR, None)
            return CognitiveDecisionResult(frame_result, transition)

        execution = ExecutionInfo.success(latency_ms=response.latency_ms)
        frame_result = FrameResult(
            sequence=sequence,
            frame_id=frame_id,
            is_observation_frame=is_observation_frame,
            execution=execution,
            prediction=parsed.prediction,
            cognition=parsed.cognition,
            context=context,
            raw_model_response=response.text,
        )
        transition = self._transition(frame_id, ExecutionStatus.OK, parsed.prediction)
        return CognitiveDecisionResult(frame_result, transition)

    def from_execution_error(
        self,
        *,
        sequence: str,
        frame_id: int,
        is_observation_frame: bool,
        status: ExecutionStatus,
        error: BaseException,
        context: Optional[ContextInfo] = None,
        latency_ms: Optional[float] = None,
    ) -> CognitiveDecisionResult:
        """把图像/模型/API 异常写成标准失败结果。"""

        status = ExecutionStatus(status)
        if status in (ExecutionStatus.OK, ExecutionStatus.PARSE_ERROR):
            raise ValueError("该接口仅用于 image/model/api/skipped 等非解析执行状态")
        execution = ExecutionInfo.failure(status, error, latency_ms=latency_ms)
        frame_result = FrameResult(
            sequence=sequence,
            frame_id=frame_id,
            is_observation_frame=is_observation_frame,
            execution=execution,
            prediction=None,
            context=context or ContextInfo(),
        )
        transition = self._transition(frame_id, status, None)
        return CognitiveDecisionResult(frame_result, transition)
