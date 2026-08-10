"""基于本地 Qwen-VL 的标准 pytracking 认知跟踪器。

该类只负责组件编排：视觉上下文、模型推理、严格解析、内部状态机和记忆门控
均由 ``cogtrack`` 中的独立模块完成。它不保存结果、不读取后续 GT，也不计算指标。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from cogtrack.cognition import CognitiveDecisionEngine, CognitiveStateMachine, StateMachineConfig
from cogtrack.context import TrackingContextBuilder
from cogtrack.memory import (
    GatedMemoryUpdatePolicy,
    IdentityAnchor,
    MemoryBank,
    MemoryBankConfig,
    MemoryCandidate,
    MemoryKind,
    MemorySource,
    MemoryUpdateDecision,
    MemoryUpdatePolicyConfig,
)
from cogtrack.protocol import (
    BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    CognitionInfo,
    ContextInfo,
    ExecutionInfo,
    ExecutionStatus,
    FrameResult,
    IdentityMatch,
    Localizability,
    Prediction,
    TargetPresence,
    validate_bbox_protocol,
    validate_xywh,
)
from cogtrack.vlm import (
    VLMBackendError,
    VLMDependencyError,
    VLMLoadError,
    build_backend,
    describe_backend,
    path_fields_for,
    resolve_backend_name,
)
from pytracking.trackers.base import BaseTracker, TrackerParams


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} 必须是 mapping")
    return dict(value)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"模型配置不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"模型配置顶层必须是 mapping: {path}")
    payload["_config_path"] = str(path)
    return payload


def _json_safe_state(state: Any) -> dict[str, Any]:
    """把状态机 dataclass 转成不含 Enum 的普通 JSON 字典。"""

    output = asdict(state)
    for key, value in tuple(output.items()):
        if hasattr(value, "value"):
            output[key] = value.value
        elif isinstance(value, tuple):
            output[key] = list(value)
    return output


@dataclass(frozen=True)
class BBoxCommitDecision:
    """单帧候选框的可审计提交结果。"""

    accepted: bool
    reason: str
    candidate_bbox: tuple[float, float, float, float] | None
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "source": self.source,
            "required": {
                "target_status": TargetPresence.PRESENT.value,
                "bbox": "valid",
            },
        }


class CognitiveVLMTracker(BaseTracker):
    """纯 VLM 的 pair/mosaic 认知跟踪 baseline。"""

    def __init__(self, params: Mapping[str, Any] | TrackerParams | None = None) -> None:
        super().__init__(params)
        self.context_mode = str(self.params.get("context_mode", "pair")).lower()
        if self.context_mode not in {"pair", "mosaic"}:
            raise ValueError("context_mode 只允许 pair/mosaic")

        # 零样本评测必须用模型的原生坐标约定：Qwen2.5-VL 起输出的是它自己看到
        # 那张图的绝对像素，让它给 norm1000 等于考它没被训过的格式。norm1000 保
        # 留给用本仓库 SFT/GRPO 数据微调过的模型。
        self.bbox_protocol = validate_bbox_protocol(
            str(self.params.get("bbox_protocol", BBOX_PROTOCOL_QWEN_ABS_PIXEL))
        )

        self.save_raw_response = bool(self.params.get("save_raw_response", True))
        if _mapping(self.params.get("bbox_commit"), "bbox_commit"):
            raise ValueError("cogtrack.v2 已移除 bbox_commit 置信度阈值，请删除该配置块")

        memory_config = _mapping(self.params.get("memory"), "memory")
        self.memory_enabled = bool(memory_config.get("enabled", self.context_mode == "mosaic"))
        self.memory_output_enabled = bool(memory_config.get("model_output_enabled", True))
        self.semantic_memory_enabled = bool(memory_config.get("semantic_enabled", True))
        self.history_size = int(memory_config.get("max_positive_records", 3))
        if self.history_size < 0:
            raise ValueError("max_positive_records 不能为负数")

        bank_config = MemoryBankConfig(
            positive_capacity=int(memory_config.get("max_positive_records", 8)),
            negative_capacity=0,
            semantic_capacity=int(memory_config.get("max_semantic_records", 4)),
        )
        policy_config = MemoryUpdatePolicyConfig(
            consecutive_positive_confirmations=int(memory_config.get("confirmations", 2)),
            max_confirmation_gap=int(memory_config.get("max_confirmation_gap", 30)),
            min_bbox_iou_consistency=float(memory_config.get("min_bbox_iou_consistency", 0.0)),
            min_positive_frame_gap=int(memory_config.get("min_positive_frame_gap", 5)),
            min_semantic_frame_gap=int(memory_config.get("min_semantic_frame_gap", 30)),
        )
        self._bank_config = bank_config
        self._memory_policy = GatedMemoryUpdatePolicy(policy_config)

        model_config_path = self._resolve_model_config_path()
        model_payload = _load_yaml_mapping(model_config_path)
        # backend 由模型配置决定：本地权重走 huggingface_qwen，vLLM/远程服务走
        # openai_api。两者都要把相对目录名解析到本机的权重根目录，只是字段名
        # 不同（本地是 model_path，API 是用于确定坐标系的 processor_path）。
        self.backend_name = resolve_backend_name(model_payload)
        for field_name in path_fields_for(self.backend_name):
            value = model_payload.get(field_name)
            if isinstance(value, str):
                model_payload[field_name] = str(self._resolve_model_path(value, model_config_path))
        # Tracker 依赖抽象 VLMBackend；生成参数由工厂单独返回，测试替身或 API
        # backend 不需要伪造 Qwen 专属的 ``config`` 属性。
        self.backend, self.generation_config = build_backend(model_payload)

        state_config = _mapping(self.params.get("state_machine"), "state_machine")
        self.state_machine = CognitiveStateMachine(
            StateMachineConfig(
                confirmations_to_recover=int(state_config.get("confirmations_to_recover", 1)),
                uncertain_frames_to_search=int(state_config.get("uncertain_frames_to_search", 3)),
            )
        )
        self.decision_engine = CognitiveDecisionEngine(self.state_machine)
        self.sequence_name = ""
        self.target_text = ""
        self.anchor: IdentityAnchor | None = None
        self.memory_bank: MemoryBank | None = None
        self.context_builder: TrackingContextBuilder | None = None

    def _resolve_model_config_path(self) -> Path:
        value = self.params.get("model_config")
        if not value:
            raise ValueError("cognitive_vlm tracker 配置必须提供 model_config")
        path = Path(os.path.expandvars(os.path.expanduser(str(value))))
        if path.is_absolute():
            return path.resolve()
        owner = self.params.get("_config_path")
        if not owner:
            raise ValueError("相对 model_config 需要 tracker 参数中的 _config_path")
        return (Path(str(owner)).resolve().parent / path).resolve()

    def _resolve_model_path(self, value: str, model_config_path: Path) -> Path:
        """把模型目录名解析到当前机器的权重根目录。

        可提交的模型配置应只写 ``Qwen2.5-VL-7B-Instruct`` 这类相对目录名。
        根目录按 ``COGTRACK_MODEL_ROOT``、runner 注入的 ``runtime.model_root``
        依次解析；显式绝对路径仍可用于一次性实验。若两者都没有，则回退为
        相对模型 YAML 的路径，便于把小模型与配置一起打包。
        """

        expanded = Path(os.path.expandvars(os.path.expanduser(value)))
        if expanded.is_absolute():
            return expanded.resolve(strict=False)

        env_root = os.environ.get("COGTRACK_MODEL_ROOT")
        runtime = _mapping(self.params.get("runtime"), "runtime")
        runtime_root = runtime.get("model_root")
        root = env_root or runtime_root
        if root:
            return (Path(str(root)).expanduser() / expanded).resolve(strict=False)
        return (model_config_path.parent / expanded).resolve(strict=False)

    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        bbox = validate_xywh(info["init_bbox"])
        self.sequence_name = str(info.get("sequence_name") or info.get("seq_name") or "unknown")
        self.target_text = str(info.get("init_nlp") or "").strip()
        frame_path = str(info.get("frame_path") or "")
        self.anchor = IdentityAnchor(
            frame_id=0,
            bbox_xywh=bbox,
            target_text=self.target_text,
            image_ref=frame_path,
            image=image.copy(),
        )
        self.memory_bank = MemoryBank(self.anchor, self._bank_config)
        self.context_builder = TrackingContextBuilder(self.anchor, bbox_protocol=self.bbox_protocol)
        transition = self.state_machine.initialize(0, bbox)

        prediction = Prediction(
            target_presence=TargetPresence.PRESENT,
            identity_match=IdentityMatch.SAME,
            localizability=Localizability.LOCALIZABLE,
            bbox_xywh=bbox,
        )
        frame_result = FrameResult(
            sequence=self.sequence_name,
            frame_id=0,
            is_observation_frame=True,
            execution=ExecutionInfo.success(latency_ms=0.0),
            prediction=prediction,
            cognition=CognitionInfo(
                target_text=self.target_text,
                reasoning="使用第一帧真值建立不可变身份锚点。",
                memory_updated=False,
                memory_update_reason="初始化真值存入 IdentityAnchor，不进入动态记忆。",
            ),
            context=ContextInfo(
                model_name=self.backend.model_name,
                prompt_name="initialization",
                prompt_version="1.0.0",
                reference_frames=(0,),
            ),
        )
        return self._to_tracker_output(
            frame_result,
            transition.current,
            memory_decision=None,
            is_initialization=True,
        )

    def _require_initialized(self) -> tuple[MemoryBank, TrackingContextBuilder]:
        if self.memory_bank is None or self.context_builder is None:
            raise RuntimeError("必须先调用 initialize()")
        return self.memory_bank, self.context_builder

    def _build_context(self, image: np.ndarray):
        bank, builder = self._require_initialized()
        semantic_memory = "\n".join(
            f"- {record.text}"
            for record in bank.records(MemoryKind.SEMANTIC)
            if record.text.strip()
        )
        if self.context_mode == "mosaic":
            records = bank.select_positive(self.history_size)
            return builder.build_mosaic(
                image,
                records,
                self.target_text,
                semantic_memory,
                self.memory_output_enabled,
            )
        return builder.build_pair(
            image,
            self.target_text,
            semantic_memory,
            self.memory_output_enabled,
        )

    def track(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        bank, _ = self._require_initialized()
        frame_id = int(info.get("frame_num", self.state_machine.state.frame_id + 1))
        is_observation = bool(info.get("is_observation_frame", True))

        if not is_observation:
            transition = self.state_machine.step(frame_id, ExecutionStatus.SKIPPED, None)
            frame_result = FrameResult(
                sequence=self.sequence_name,
                frame_id=frame_id,
                is_observation_frame=False,
                execution=ExecutionInfo(status=ExecutionStatus.SKIPPED),
                prediction=None,
                cognition=CognitionInfo(memory_update_reason="本帧未执行 VLM 观测。"),
                context=ContextInfo(model_name=self.backend.model_name),
            )
            return self._to_tracker_output(frame_result, transition.current, memory_decision=None)

        context = self._build_context(image)
        if len(context.images) != context.prompt.expected_image_count:
            raise RuntimeError(
                f"Prompt {context.prompt.name} 期望 {context.prompt.expected_image_count} 张图，"
                f"上下文实际生成 {len(context.images)} 张"
            )
        try:
            response = self.backend.generate(
                context.images,
                context.prompt.user_prompt,
                system_prompt=context.prompt.system_prompt,
            )
            decision = self.decision_engine.from_vlm_response(
                sequence=self.sequence_name,
                frame_id=frame_id,
                image_width=int(image.shape[1]),
                image_height=int(image.shape[0]),
                is_observation_frame=True,
                response=response,
                prompt=context.prompt,
                reference_frames=context.reference_frames,
                generation_config=self.generation_config,
            )
        except VLMBackendError as error:
            status = ExecutionStatus.MODEL_ERROR
            # 缺依赖/权重属于模型环境错误，而不是目标 absent。
            if isinstance(error, (VLMDependencyError, VLMLoadError)):
                status = ExecutionStatus.MODEL_ERROR
            decision = self.decision_engine.from_execution_error(
                sequence=self.sequence_name,
                frame_id=frame_id,
                is_observation_frame=True,
                status=status,
                error=error,
                context=ContextInfo(
                    model_name=self.backend.model_name,
                    prompt_name=context.prompt.name,
                    prompt_version=context.prompt.version,
                    reference_frames=context.reference_frames,
                    generation_config=self.generation_config.to_dict(),
                ),
            )

        frame_result = decision.frame_result
        memory_decisions: dict[str, Any] = {}
        if frame_result.execution.status is ExecutionStatus.OK and frame_result.prediction:
            prediction = frame_result.prediction
            should_process_memory = self.memory_enabled
        else:
            should_process_memory = False

        if should_process_memory and frame_result.prediction is not None:
            prediction = frame_result.prediction
            visual_candidate = MemoryCandidate(
                kind=MemoryKind.POSITIVE,
                frame_id=frame_id,
                source=MemorySource.VLM_PREDICTION,
                execution_status=frame_result.execution.status,
                target_presence=prediction.target_presence,
                identity_match=prediction.identity_match,
                bbox_xywh=prediction.bbox_xywh,
                text="",
                image_ref=str(info.get("frame_path") or ""),
                image=image.copy(),
                metadata={"effective_context_mode": context.effective_mode},
            )
            memory_decisions["visual"] = self._memory_policy.process(bank, visual_candidate)

        proposal = frame_result.cognition.memory_update_proposal
        if frame_result.execution.status is not ExecutionStatus.OK or frame_result.prediction is None:
            semantic_decision = MemoryUpdateDecision(False, "本帧执行未成功，禁止写入语义记忆")
        elif not self.memory_output_enabled:
            semantic_decision = MemoryUpdateDecision(False, "二字段 presence-only 协议未请求语义记忆")
        elif proposal is None:
            semantic_decision = MemoryUpdateDecision(False, "模型选择 memory_update=null")
        elif not self.semantic_memory_enabled:
            semantic_decision = MemoryUpdateDecision(False, "配置禁用语义记忆写入")
        else:
            prediction = frame_result.prediction
            semantic_candidate = MemoryCandidate(
                kind=MemoryKind.SEMANTIC,
                frame_id=frame_id,
                source=MemorySource.VLM_PREDICTION,
                execution_status=frame_result.execution.status,
                target_presence=prediction.target_presence,
                identity_match=prediction.identity_match,
                bbox_xywh=prediction.bbox_xywh,
                text=proposal,
                image_ref=str(info.get("frame_path") or ""),
                image=image.copy(),
                metadata={"effective_context_mode": context.effective_mode},
            )
            semantic_decision = self._memory_policy.process(bank, semantic_candidate)
        memory_decisions["semantic"] = semantic_decision

        # FrameResult 不可变，重新构造 cognition 记录模型提议与最终门控结果。
        frame_result = FrameResult(
            sequence=frame_result.sequence,
            frame_id=frame_result.frame_id,
            is_observation_frame=frame_result.is_observation_frame,
            execution=frame_result.execution,
            prediction=frame_result.prediction,
            cognition=CognitionInfo(
                target_text=frame_result.cognition.target_text,
                reasoning=frame_result.cognition.reasoning,
                memory_update_proposal=proposal,
                memory_updated=bool(semantic_decision.accepted),
                memory_update_reason=semantic_decision.reason,
            ),
            context=frame_result.context,
            raw_model_response=frame_result.raw_model_response,
        )

        return self._to_tracker_output(frame_result, self.state_machine.state, memory_decisions)

    def _decide_bbox_commit(
        self,
        frame_result: FrameResult,
        *,
        is_initialization: bool,
    ) -> BBoxCommitDecision:
        """将结构化 prediction 作为候选，独立决定是否发布 target_bbox。"""

        prediction = frame_result.prediction
        candidate_bbox = prediction.bbox_xywh if prediction is not None else None
        source = "initialization_ground_truth" if is_initialization else "vlm_prediction"

        def decision(accepted: bool, reason: str) -> BBoxCommitDecision:
            return BBoxCommitDecision(
                accepted=accepted,
                reason=reason,
                candidate_bbox=candidate_bbox,
                source=source if prediction is not None else None,
            )

        if frame_result.execution.status is not ExecutionStatus.OK:
            return decision(False, f"execution.status={frame_result.execution.status.value}，无可提交预测")
        if prediction is None:
            return decision(False, "成功执行但缺少 prediction")
        if is_initialization:
            return decision(True, "初始化真值直接建立首帧已提交框")
        if prediction.target_presence is not TargetPresence.PRESENT:
            return decision(False, "target_presence 不是 present")
        if candidate_bbox is None:
            return decision(False, "target_status=present 但缺少合法 bbox")
        return decision(True, "通过 target_status=present + 合法 bbox 门控")

    def _to_tracker_output(
        self,
        frame_result: FrameResult,
        cognitive_state: Any,
        memory_decision: Any,
        *,
        is_initialization: bool = False,
    ) -> dict[str, Any]:
        payload = frame_result.to_dict(include_raw_response=self.save_raw_response)
        commit_decision = self._decide_bbox_commit(frame_result, is_initialization=is_initialization)
        candidate_bbox = (
            list(commit_decision.candidate_bbox)
            if commit_decision.candidate_bbox is not None
            else None
        )
        target_bbox = candidate_bbox if commit_decision.accepted else None
        # 原始 prediction 描述 VLM 对最佳候选的判断；对外目标存在性必须
        # 经过同一提交门控。工程失败不能擅自推断初始化目标一定 absent。
        if commit_decision.accepted:
            committed_presence = TargetPresence.PRESENT.value
        elif (
            frame_result.execution.status is ExecutionStatus.OK
            and frame_result.prediction is not None
            and frame_result.prediction.target_presence is TargetPresence.ABSENT
        ):
            committed_presence = TargetPresence.ABSENT.value
        elif frame_result.execution.status is ExecutionStatus.OK:
            committed_presence = TargetPresence.UNCERTAIN.value
        else:
            committed_presence = None
        bank_snapshot = self.memory_bank.snapshot() if self.memory_bank is not None else None
        return {
            "target_bbox": target_bbox,
            "candidate_bbox": candidate_bbox,
            "committed_target_presence": committed_presence,
            "commit_decision": commit_decision.to_dict(),
            "schema_version": payload["schema_version"],
            "execution": payload["execution"],
            "prediction": payload["prediction"],
            "cognition": payload["cognition"],
            "context": payload["context"],
            "raw_model_response": payload.get("raw_model_response"),
            "cognitive_state": _json_safe_state(cognitive_state),
            "memory": bank_snapshot,
            "memory_decision": self._serialize_memory_decision(memory_decision),
        }

    @staticmethod
    def _serialize_memory_decision(memory_decision: Any) -> Any:
        """序列化单项或 visual/semantic 双通道记忆门控结果。"""

        if memory_decision is None:
            return None
        if isinstance(memory_decision, Mapping):
            return {
                str(name): CognitiveVLMTracker._serialize_memory_decision(decision)
                for name, decision in memory_decision.items()
            }
        return {
            "accepted": memory_decision.accepted,
            "reason": memory_decision.reason,
            "confirmations": memory_decision.confirmations,
            "kind": memory_decision.record.kind.value if memory_decision.record else None,
            "record_id": memory_decision.record.record_id if memory_decision.record else None,
            "evicted_record_id": memory_decision.evicted_record_id,
        }

    def describe_runtime(self) -> dict[str, Any]:
        """上报 backend 与坐标系事实，写入 run manifest。

        ``bbox_protocol`` 和实际生效的视觉网格参数必须留痕：同一份 prompt 配
        不同协议会得到完全不同的数值结果，事后没有这条记录就无法判断一次实验
        到底跑的是什么。``describe_backend`` 不会写出 API key。
        """

        return {
            "context_mode": self.context_mode,
            "bbox_protocol": self.bbox_protocol,
            "memory_output_enabled": self.memory_output_enabled,
            "vlm_backend": describe_backend(self.backend),
            "generation": self.generation_config.to_dict(),
        }

    def close(self) -> None:
        # 评测器每个序列都会 close tracker。Qwen backend 只释放句柄，
        # 进程级权重缓存留给下一序列；其他 backend 仍遵循 unload 接口。
        release = getattr(self.backend, "release", None)
        if callable(release):
            release()
        else:
            self.backend.unload()


def get_tracker_class() -> type[CognitiveVLMTracker]:
    """pytracking 动态 tracker 工厂约定。"""

    return CognitiveVLMTracker
