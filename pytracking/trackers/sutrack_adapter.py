"""把独立 SUTrack 运行时接入 CognitiveTrack 的 pytracking 生命周期。

本模块不依赖任何具体 SUTrack 源码。实现模块由配置指定，并延迟到首帧初始化时
导入；迁移网络代码时只需新增插件包，无需污染 runner 或数据集层。
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from cogtrack.models.sutrack import (
    SUTrackAdapterConfig,
    SUTrackOutputError,
    SUTrackPluginLoadError,
)
from cogtrack.protocol import ExecutionStatus, validate_xywh, xyxy_to_xywh
from pytracking.trackers.base import BaseTracker, TrackerParams


class SUTrackAdapter(BaseTracker):
    """插件式 SUTrack adapter；自身不持有网络结构或 checkpoint 约定。"""

    def __init__(self, params: Mapping[str, Any] | TrackerParams | None = None) -> None:
        super().__init__(params)
        self.adapter_config = SUTrackAdapterConfig.from_mapping(self.params)
        self._runtime: Any | None = None
        self._initialized = False

    @property
    def runtime_loaded(self) -> bool:
        """供诊断和测试使用；读取本属性不会触发插件导入。"""

        return self._runtime is not None

    def _load_runtime(self) -> Any:
        """首次使用时导入插件并调用工厂，避免仅检查数据时加载模型。"""

        if self._runtime is not None:
            return self._runtime
        config = self.adapter_config
        try:
            module = importlib.import_module(config.module)
        except Exception as error:
            raise SUTrackPluginLoadError(
                f"无法导入 SUTrack 实现模块 {config.module!r}；请检查插件是否已安装及其依赖"
            ) from error

        factory = getattr(module, config.factory, None)
        if not callable(factory):
            raise SUTrackPluginLoadError(
                f"SUTrack 模块 {config.module!r} 中不存在可调用工厂 {config.factory!r}"
            )
        try:
            runtime = factory(params=self.params.clone(), **dict(config.factory_kwargs))
        except Exception as error:
            raise SUTrackPluginLoadError(
                f"SUTrack 工厂 {config.module}.{config.factory} 构造运行时失败: {error}"
            ) from error

        missing = [name for name in ("initialize", "track") if not callable(getattr(runtime, name, None))]
        if missing:
            names = ", ".join(missing)
            raise SUTrackPluginLoadError(f"SUTrack 工厂返回对象缺少必要方法: {names}")
        self._runtime = runtime
        return runtime

    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        """加载插件并建立首帧状态；允许官方 pytracking 风格的空返回。"""

        init_bbox = info.get("init_bbox")
        if init_bbox is None:
            raise ValueError("SUTrackAdapter.initialize() 需要 info['init_bbox']")
        normalized_init_bbox = list(validate_xywh(_box_values(init_bbox, "info.init_bbox")))
        runtime = self._load_runtime()
        raw_output = runtime.initialize(image, info)
        output = self._normalize_output(raw_output, fallback_bbox=normalized_init_bbox, stage="initialize")
        if output["execution"]["status"] == ExecutionStatus.OK.value:
            self._initialized = True
        return output

    def track(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        """执行一帧 SUTrack 推理并输出统一的像素 xywh 与工程状态。"""

        if not self._initialized or self._runtime is None:
            raise RuntimeError("SUTrackAdapter 必须先成功调用 initialize()")
        raw_output = self._runtime.track(image, info)
        return self._normalize_output(raw_output, fallback_bbox=None, stage="track")

    def correct(
        self,
        image: np.ndarray,
        bbox_xywh: Sequence[float],
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """可选地用外部可靠框校正 SUTrack 状态。

        具体运行时可以实现 ``correct(image, bbox_xywh, info)``，用于重设运动状态
        或重新提取模板；没有该方法时返回 ``supported=False``，不会假装已经完成
        回灌。返回 ``None`` 代表成功，布尔值或 ``{'applied': bool, 'reason': str}``
        也会被规范化为稳定诊断字段。
        """

        if not self._initialized or self._runtime is None:
            raise RuntimeError("SUTrackAdapter 必须先成功调用 initialize() 才能校正")
        normalized_bbox = list(validate_xywh(_box_values(bbox_xywh, "correction.bbox_xywh")))
        callback = getattr(self._runtime, "correct", None)
        if not callable(callback):
            return {
                "supported": False,
                "applied": False,
                "reason": "SUTrack 运行时未实现可选 correct() 接口",
            }

        try:
            result = callback(image, normalized_bbox, dict(info or {}))
        except Exception as error:
            raise RuntimeError(f"SUTrack 运行时 correct() 校正失败: {error}") from error

        if result is None:
            applied = True
            reason = "SUTrack 运行时已接受外部可靠框"
        elif isinstance(result, bool):
            applied = result
            reason = "SUTrack 运行时返回校正结果"
        elif isinstance(result, Mapping):
            raw_applied = result.get("applied", True)
            if not isinstance(raw_applied, bool):
                raise SUTrackOutputError("SUTrack correct() 输出中的 applied 必须是 bool")
            applied = raw_applied
            reason = str(result.get("reason") or "SUTrack 运行时返回校正结果")
        else:
            raise SUTrackOutputError(
                "SUTrack correct() 必须返回 None、bool 或包含 applied 的 mapping"
            )
        return {"supported": True, "applied": applied, "reason": reason}

    def _normalize_output(
        self,
        raw_output: Mapping[str, Any] | None,
        *,
        fallback_bbox: Sequence[float] | None,
        stage: str,
    ) -> dict[str, Any]:
        """将插件的 bbox、执行状态和置信度收敛到稳定且可序列化的字段。"""

        if raw_output is None:
            payload: dict[str, Any] = {}
        elif isinstance(raw_output, Mapping):
            payload = dict(raw_output)
        else:
            raise SUTrackOutputError(
                f"SUTrack {stage} 输出必须是 mapping 或 None，实际为 {type(raw_output).__name__}"
            )

        output_config = self.adapter_config.output
        execution = _normalize_execution(payload.get(output_config.execution_key))
        status = execution["status"]

        raw_bbox = payload.get(output_config.bbox_key)
        bbox_from_fallback = False
        if raw_bbox is None and fallback_bbox is not None and status == ExecutionStatus.OK.value:
            raw_bbox = fallback_bbox
            # init_bbox 已经是框架内部 xywh，不能再按插件声明的原始格式转换。
            bbox_from_fallback = True
        if status == ExecutionStatus.OK.value and raw_bbox is None:
            raise SUTrackOutputError(f"SUTrack {stage} 执行成功但没有输出 {output_config.bbox_key!r}")
        if status != ExecutionStatus.OK.value and raw_bbox is not None:
            raise SUTrackOutputError(
                f"SUTrack {stage} execution.status={status!r} 时不能同时输出 bbox"
            )

        target_bbox: list[float] | None = None
        if raw_bbox is not None:
            values = _box_values(raw_bbox, output_config.bbox_key)
            if output_config.bbox_format == "xyxy" and not bbox_from_fallback:
                values = xyxy_to_xywh(values)
            target_bbox = list(validate_xywh(values))

        normalized: dict[str, Any] = {
            "target_bbox": target_bbox,
            "execution": execution,
            "diagnostics": {
                "backend": "sutrack_plugin",
                "implementation_module": self.adapter_config.module,
                "implementation_factory": self.adapter_config.factory,
            },
        }
        score_key = output_config.score_key
        if score_key is not None and score_key in payload:
            normalized["confidence"] = _finite_score(payload[score_key], score_key)
        return normalized

    def close(self) -> None:
        """若插件持有设备资源，则在 runner 结束序列时转发释放调用。"""

        runtime = self._runtime
        try:
            close = getattr(runtime, "close", None) if runtime is not None else None
            if callable(close):
                close()
        finally:
            self._runtime = None
            self._initialized = False


def _normalize_execution(value: Any) -> dict[str, Any]:
    """严格规范工程执行状态，不把 lost 等语义状态误当作运行错误。"""

    if value is None:
        payload: dict[str, Any] = {"status": ExecutionStatus.OK.value}
    elif isinstance(value, Mapping):
        payload = dict(value)
        payload.setdefault("status", ExecutionStatus.OK.value)
    elif isinstance(value, (str, ExecutionStatus)):
        payload = {"status": str(value)}
    elif hasattr(value, "status"):
        payload = {"status": str(value.status)}
        for key in ("latency_ms", "error_type", "error_message"):
            field = getattr(value, key, None)
            if field is not None:
                payload[key] = field
    else:
        raise SUTrackOutputError(
            f"SUTrack execution 必须是字符串、mapping 或含 status 的对象，实际为 {type(value).__name__}"
        )

    try:
        status = ExecutionStatus(payload["status"])
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in ExecutionStatus)
        raise SUTrackOutputError(
            f"未知 SUTrack execution.status={payload.get('status')!r}；允许值为: {allowed}"
        ) from error
    payload["status"] = status.value

    latency = payload.get("latency_ms")
    if latency is not None:
        payload["latency_ms"] = _nonnegative_finite(latency, "execution.latency_ms")
    if status is ExecutionStatus.OK and (payload.get("error_type") or payload.get("error_message")):
        raise SUTrackOutputError("SUTrack execution.status='ok' 时不能携带错误信息")
    return payload


def _box_values(value: Any, field_name: str) -> tuple[float, float, float, float]:
    """兼容 list、numpy array 和 CPU tensor 的四维 bbox，不引入 torch 依赖。"""

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise SUTrackOutputError(f"{field_name} 必须是长度为 4 的数值序列")
    return tuple(value)  # type: ignore[return-value]


def _finite_score(value: Any, field_name: str) -> float:
    """把标量 tensor/数字转换为 JSON 可序列化的有限浮点数。"""

    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SUTrackOutputError(f"{field_name} 必须是标量数值")
    score = float(value)
    if not math.isfinite(score):
        raise SUTrackOutputError(f"{field_name} 必须是有限数值")
    return score


def _nonnegative_finite(value: Any, field_name: str) -> float:
    number = _finite_score(value, field_name)
    if number < 0.0:
        raise SUTrackOutputError(f"{field_name} 必须是非负数值")
    return number


def get_tracker_class() -> type[SUTrackAdapter]:
    """遵循 pytracking 动态 tracker 工厂约定。"""

    return SUTrackAdapter
