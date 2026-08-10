"""SUTrack 连续定位与关键帧 VLM 身份裁决的混合跟踪器。

SUTrack 在每一帧运行，保证标准 benchmark 获得稠密候选框；CognitiveVLMTracker
同样在每帧收到调用，但由 runner 的 ``is_observation_frame`` 决定是否真正执行
昂贵推理。融合层严格区分语义拒绝与工程失败，且只发布一个顶层 benchmark bbox。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from cogtrack.protocol import ExecutionStatus, validate_xywh
from pytracking.trackers.base import BaseTracker, TrackerParams
from pytracking.trackers.cognitive_vlm import CognitiveVLMTracker
from pytracking.trackers.sutrack_adapter import SUTrackAdapter

_ACTIONS = {"suppress", "fallback"}


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} 必须是 mapping")
    return dict(value)


@dataclass(frozen=True)
class HybridFusionConfig:
    """关键帧语义拒绝和 SUTrack 回退策略。"""

    semantic_rejection_action: str = "suppress"
    uncommitted_action: str = "fallback"
    correct_sutrack_on_commit: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "HybridFusionConfig":
        payload = dict(values or {})
        allowed = {
            "semantic_rejection_action",
            "uncommitted_action",
            "correct_sutrack_on_commit",
        }
        unknown = set(payload) - allowed
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"fusion 含未知参数: {names}")

        semantic_action = str(payload.get("semantic_rejection_action", "suppress")).lower()
        uncommitted_action = str(payload.get("uncommitted_action", "fallback")).lower()
        if semantic_action not in _ACTIONS:
            raise ValueError("fusion.semantic_rejection_action 只允许 suppress/fallback")
        if uncommitted_action not in _ACTIONS:
            raise ValueError("fusion.uncommitted_action 只允许 suppress/fallback")
        correction = payload.get("correct_sutrack_on_commit", True)
        if not isinstance(correction, bool):
            raise ValueError("fusion.correct_sutrack_on_commit 必须是 bool")

        return cls(
            semantic_rejection_action=semantic_action,
            uncommitted_action=uncommitted_action,
            correct_sutrack_on_commit=correction,
        )


def _child_params(parent: TrackerParams, field_name: str) -> TrackerParams:
    """构造隔离的子 tracker 参数，并显式继承配置归属和 runner 上下文。"""

    child = _mapping(parent.get(field_name), field_name)
    if "_config_path" not in child and parent.get("_config_path") is not None:
        child["_config_path"] = parent["_config_path"]

    parent_runtime = parent.get("runtime") or {}
    child_runtime = child.get("runtime") or {}
    if not isinstance(parent_runtime, Mapping):
        raise TypeError("runtime 必须是 mapping")
    if not isinstance(child_runtime, Mapping):
        raise TypeError(f"{field_name}.runtime 必须是 mapping")
    merged_runtime = {**dict(parent_runtime), **dict(child_runtime)}
    if merged_runtime:
        child["runtime"] = merged_runtime
    return TrackerParams(child)


class HybridCognitiveTracker(BaseTracker):
    """逐帧 SUTrack + 稀疏 VLM 认知判别的身份安全混合 tracker。"""

    def __init__(self, params: Mapping[str, Any] | TrackerParams | None = None) -> None:
        super().__init__(params)
        fusion = self.params.get("fusion")
        if fusion is not None and not isinstance(fusion, Mapping):
            raise TypeError("fusion 必须是 mapping")
        self.fusion_config = HybridFusionConfig.from_mapping(fusion)
        self.sutrack = SUTrackAdapter(_child_params(self.params, "sutrack"))
        self.vlm = CognitiveVLMTracker(_child_params(self.params, "vlm"))

    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        """两个分支都使用同一首帧真值初始化，首帧语义固定为 present。"""

        sutrack_output = self._safe_call(self.sutrack, "initialize", image, info, "sutrack")
        vlm_output = self._safe_call(self.vlm, "initialize", image, info, "vlm")
        return self._fuse(
            image=image,
            info=info,
            sutrack_output=sutrack_output,
            vlm_output=vlm_output,
            is_initialization=True,
        )

    def track(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        """每帧调用两个分支；非关键帧 VLM 自行返回 skipped，不产生 NaN。"""

        # 先运行连续定位分支；即使它异常，_safe_call 也会记录错误并继续调用 VLM。
        sutrack_output = self._safe_call(self.sutrack, "track", image, info, "sutrack")
        vlm_output = self._safe_call(self.vlm, "track", image, info, "vlm")
        return self._fuse(
            image=image,
            info=info,
            sutrack_output=sutrack_output,
            vlm_output=vlm_output,
            is_initialization=False,
        )

    @staticmethod
    def _safe_call(
        tracker: BaseTracker,
        method_name: str,
        image: np.ndarray,
        info: dict[str, Any],
        branch_name: str,
    ) -> dict[str, Any]:
        """把单分支意外异常限制在当前分支，允许另一分支提供可审计降级。"""

        try:
            result = getattr(tracker, method_name)(image, dict(info))
            if result is None:
                output: dict[str, Any] = {}
            elif isinstance(result, Mapping):
                output = dict(result)
            else:
                raise TypeError(f"输出必须是 mapping 或 None，实际为 {type(result).__name__}")

            raw_execution = output.get("execution")
            if raw_execution is None:
                execution = {"status": ExecutionStatus.OK.value}
            elif isinstance(raw_execution, Mapping):
                execution = dict(raw_execution)
                execution.setdefault("status", ExecutionStatus.OK.value)
            elif isinstance(raw_execution, str):
                execution = {"status": raw_execution}
            else:
                raise TypeError("execution 必须是 mapping 或字符串")
            execution["status"] = ExecutionStatus(execution["status"]).value
            output["execution"] = execution

            if output.get("target_bbox") is not None:
                output["target_bbox"] = list(validate_xywh(output["target_bbox"]))
            return output
        except Exception as error:
            return {
                "target_bbox": None,
                "execution": {
                    "status": ExecutionStatus.INTERNAL_ERROR.value,
                    "error_type": type(error).__name__,
                    "error_message": f"{branch_name} 分支调用失败: {error}",
                },
            }

    def _fuse(
        self,
        *,
        image: np.ndarray,
        info: dict[str, Any],
        sutrack_output: dict[str, Any],
        vlm_output: dict[str, Any],
        is_initialization: bool,
    ) -> dict[str, Any]:
        is_observation = bool(info.get("is_observation_frame", True))
        sutrack_bbox = sutrack_output.get("target_bbox")
        vlm_bbox = vlm_output.get("target_bbox")
        sutrack_execution = dict(sutrack_output["execution"])
        vlm_execution = dict(vlm_output["execution"])
        sutrack_ok = sutrack_execution["status"] == ExecutionStatus.OK.value and sutrack_bbox is not None
        vlm_ok = vlm_execution["status"] == ExecutionStatus.OK.value
        committed = vlm_ok and vlm_bbox is not None and _commit_accepted(vlm_output)
        correction: dict[str, Any] = {"attempted": False, "supported": None, "applied": False}

        selected_bbox: list[float] | None
        selected_execution: dict[str, Any]
        selected_source: str
        reason: str
        baseline_assumed_present = False

        if is_initialization:
            if committed:
                selected_bbox = vlm_bbox
                selected_execution = vlm_execution
                selected_source = "vlm"
                reason = "首帧 VLM 已提交初始化真值"
            else:
                selected_bbox, selected_execution, selected_source, reason = self._fallback_to_sutrack(
                    sutrack_bbox,
                    sutrack_execution,
                    sutrack_ok,
                    "首帧 VLM 不可用，回退 SUTrack 初始化框",
                )
            committed_presence = "present" if selected_bbox is not None else "uncertain"
        elif not is_observation:
            # CognitiveVLMTracker 仍然被调用并应返回 skipped；最终连续框只来自 SUTrack。
            selected_bbox, selected_execution, selected_source, reason = self._fallback_to_sutrack(
                sutrack_bbox,
                sutrack_execution,
                sutrack_ok,
                "非关键帧采用逐帧 SUTrack 连续定位",
            )
            baseline_assumed_present = selected_bbox is not None
            committed_presence = "present" if baseline_assumed_present else "uncertain"
        elif not vlm_ok:
            selected_bbox, selected_execution, selected_source, reason = self._fallback_to_sutrack(
                sutrack_bbox,
                sutrack_execution,
                sutrack_ok,
                f"VLM 工程状态为 {vlm_execution['status']}，回退 SUTrack",
            )
            # 工程失败时没有新的身份证据，即使为 benchmark 延续框也不声称已确认目标。
            committed_presence = "uncertain"
        elif committed:
            selected_bbox = vlm_bbox
            selected_execution = vlm_execution
            selected_source = "vlm"
            reason = "VLM 候选已通过 present + same + localizable 提交门控"
            committed_presence = "present"
            if self.fusion_config.correct_sutrack_on_commit:
                correction = self._correct_sutrack(image, vlm_bbox, info)
        else:
            rejection = self._semantic_rejection(vlm_output)
            action = (
                self.fusion_config.semantic_rejection_action
                if rejection is not None
                else self.fusion_config.uncommitted_action
            )
            semantic_reason = rejection or "VLM 未提交可靠目标框"
            committed_presence = _uncommitted_presence(vlm_output)
            if action == "suppress":
                selected_bbox = None
                selected_execution = vlm_execution
                selected_source = "suppressed"
                reason = f"{semantic_reason}；按身份安全策略抑制 SUTrack 候选"
            else:
                selected_bbox, selected_execution, selected_source, reason = self._fallback_to_sutrack(
                    sutrack_bbox,
                    sutrack_execution,
                    sutrack_ok,
                    f"{semantic_reason}；按配置回退 SUTrack",
                )
                # 融合层既然选择了发布 baseline 框，就不能同时对外宣称
                # 目标已确认 absent。保留原始 VLM absent 供 diagnostics，融合决策为 uncertain。
                if selected_bbox is not None:
                    committed_presence = "uncertain"

        diagnostics = {
            "backend": "hybrid_cognitive",
            "selected_source": selected_source,
            "selection_reason": reason,
            "is_observation_frame": is_observation,
            "baseline_assumed_present": baseline_assumed_present,
            "sutrack_bbox": sutrack_bbox,
            "vlm_committed_bbox": vlm_bbox if committed else None,
            "sutrack_execution": sutrack_execution,
            # 模型/API/解析错误完整保留在这里，不能被 SUTrack 的 ok 覆盖掉。
            "vlm_execution": vlm_execution,
            "sutrack_correction": correction,
        }
        output: dict[str, Any] = {
            "target_bbox": selected_bbox,
            "committed_target_presence": committed_presence,
            "execution": selected_execution,
            "prediction": vlm_output.get("prediction"),
            "diagnostics": diagnostics,
        }
        # 保留认知分支的可解释与记忆字段，但不复制它的 target_bbox；顶层框是
        # 唯一 benchmark 输入，避免评测脚本误取未提交候选。
        for key in (
            "schema_version",
            "candidate_bbox",
            "commit_decision",
            "cognition",
            "context",
            "raw_model_response",
            "cognitive_state",
            "memory",
            "memory_decision",
        ):
            if key in vlm_output:
                output[key] = vlm_output[key]
        return output

    @staticmethod
    def _fallback_to_sutrack(
        sutrack_bbox: list[float] | None,
        sutrack_execution: dict[str, Any],
        sutrack_ok: bool,
        reason: str,
    ) -> tuple[list[float] | None, dict[str, Any], str, str]:
        if sutrack_ok:
            return sutrack_bbox, sutrack_execution, "sutrack", reason
        execution = sutrack_execution
        if execution["status"] == ExecutionStatus.OK.value:
            execution = {
                "status": ExecutionStatus.INTERNAL_ERROR.value,
                "error_type": "MissingBBox",
                "error_message": "SUTrack 执行成功但没有可用 target_bbox",
            }
        return None, execution, "none", f"{reason}；但 SUTrack 当前不可用"

    def _semantic_rejection(self, vlm_output: Mapping[str, Any]) -> str | None:
        """按 VLM 的离散状态执行语义拒绝，不解释其自报概率。"""

        prediction = vlm_output.get("prediction")
        if not isinstance(prediction, Mapping):
            return None
        presence = str(prediction.get("target_presence", ""))
        identity = str(prediction.get("identity_match", ""))

        if presence == "absent":
            return "VLM 判定目标 absent"
        if identity == "different":
            return "VLM 判定当前候选为 different 实例"
        if presence == "uncertain":
            return "VLM 主动拒答目标存在性"
        if identity == "uncertain":
            return "VLM 主动拒答实例身份"
        return None

    def _correct_sutrack(
        self,
        image: np.ndarray,
        bbox_xywh: list[float],
        info: dict[str, Any],
    ) -> dict[str, Any]:
        callback = getattr(self.sutrack, "correct", None)
        if not callable(callback):
            return {
                "attempted": True,
                "supported": False,
                "applied": False,
                "reason": "当前 SUTrack adapter 未提供 correct()",
            }
        try:
            result = callback(image, bbox_xywh, dict(info))
            if isinstance(result, Mapping):
                return {"attempted": True, **dict(result)}
            if result is None:
                return {"attempted": True, "supported": True, "applied": True}
            if isinstance(result, bool):
                return {"attempted": True, "supported": True, "applied": result}
            raise TypeError(f"correct() 返回不支持的类型 {type(result).__name__}")
        except Exception as error:
            # 回灌失败不能推翻当前已经提交的 VLM 框，但必须完整记录以便定位漂移。
            return {
                "attempted": True,
                "supported": True,
                "applied": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    def close(self) -> None:
        """确保两个分支都释放；一个分支异常不能阻止另一个分支 close。"""

        errors: list[Exception] = []
        for tracker in (self.vlm, self.sutrack):
            try:
                tracker.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"hybrid tracker 关闭分支失败: {errors[0]}") from errors[0]


def _commit_accepted(vlm_output: Mapping[str, Any]) -> bool:
    decision = vlm_output.get("commit_decision")
    if decision is None:
        # target_bbox 是 CognitiveVLMTracker 的已提交公开字段；测试替身可省略审计细节。
        return True
    return isinstance(decision, Mapping) and decision.get("accepted") is True


def _uncommitted_presence(vlm_output: Mapping[str, Any]) -> str:
    """提交三态：different 只是否定身份，绝不能自动改写成目标 absent。"""

    prediction = vlm_output.get("prediction")
    if isinstance(prediction, Mapping) and str(prediction.get("target_presence")) == "absent":
        return "absent"
    return "uncertain"


def get_tracker_class() -> type[HybridCognitiveTracker]:
    """遵循 pytracking 动态 tracker 工厂约定。"""

    return HybridCognitiveTracker
