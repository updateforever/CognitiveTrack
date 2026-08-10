"""SUTrack 实现与 pytracking 适配层之间的最小契约。

这里不包含网络结构、权重加载或旧工程兼容逻辑。具体 SUTrack 实现只要由工厂
构造，并提供标准 ``initialize/track`` 方法，即可接入 CognitiveTrack。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

_DOTTED_MODULE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_PYTHON_NAME = re.compile(r"^[A-Za-z_]\w*$")


class SUTrackConfigurationError(ValueError):
    """SUTrack adapter 参数不完整或不安全。"""


class SUTrackPluginLoadError(RuntimeError):
    """SUTrack 插件模块、工厂或运行时无法加载。"""


class SUTrackOutputError(TypeError):
    """SUTrack 运行时输出不符合 adapter 协议。"""


@runtime_checkable
class SUTrackRuntime(Protocol):
    """具体 SUTrack 推理实现必须满足的结构化接口。

    ``close`` 有意不放进强制协议：纯 Python/CPU 测试实现通常无需释放资源；
    adapter 会在对象提供 ``close()`` 时自动调用。
    """

    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> Mapping[str, Any] | None:
        """使用首帧 RGB 图像和 ``info['init_bbox']`` 初始化。"""

        ...

    def track(self, image: np.ndarray, info: dict[str, Any]) -> Mapping[str, Any] | None:
        """处理后续 RGB 图像并返回跟踪结果。"""

        ...


@dataclass(frozen=True)
class SUTrackOutputConfig:
    """声明插件原始输出中各字段的含义，禁止按数值猜测 bbox 格式。"""

    bbox_key: str = "target_bbox"
    bbox_format: str = "xywh"
    execution_key: str = "execution"
    score_key: str | None = "best_score"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "SUTrackOutputConfig":
        payload = dict(values or {})
        unknown = set(payload) - {"bbox_key", "bbox_format", "execution_key", "score_key"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise SUTrackConfigurationError(f"SUTrack output 含未知参数: {names}")

        bbox_key = _required_name(payload.get("bbox_key", "target_bbox"), "output.bbox_key")
        execution_key = _required_name(payload.get("execution_key", "execution"), "output.execution_key")
        bbox_format = str(payload.get("bbox_format", "xywh")).lower()
        if bbox_format not in {"xywh", "xyxy"}:
            raise SUTrackConfigurationError("output.bbox_format 只允许 xywh 或 xyxy")

        raw_score_key = payload.get("score_key", "best_score")
        score_key = None if raw_score_key is None else _required_name(raw_score_key, "output.score_key")
        return cls(
            bbox_key=bbox_key,
            bbox_format=bbox_format,
            execution_key=execution_key,
            score_key=score_key,
        )


@dataclass(frozen=True)
class SUTrackAdapterConfig:
    """SUTrack 插件定位、工厂参数和输出声明。"""

    module: str
    factory: str = "build_sutrack_runtime"
    factory_kwargs: Mapping[str, Any] = field(default_factory=dict)
    output: SUTrackOutputConfig = field(default_factory=SUTrackOutputConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SUTrackAdapterConfig":
        """从 tracker 顶层参数解析严格配置。

        预期结构为 ``implementation: {module, factory, kwargs}`` 与可选的
        ``output``。其他 tracker 公共参数会被保留给插件，因此只严格检查这两个
        子块，不限制顶层键。
        """

        implementation = values.get("implementation")
        if not isinstance(implementation, Mapping):
            raise SUTrackConfigurationError("SUTrack adapter 必须提供 implementation mapping")
        payload = dict(implementation)
        unknown = set(payload) - {"module", "factory", "kwargs"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise SUTrackConfigurationError(f"SUTrack implementation 含未知参数: {names}")

        module = str(payload.get("module", "")).strip()
        if not module or not _DOTTED_MODULE.fullmatch(module):
            raise SUTrackConfigurationError(
                "implementation.module 必须是绝对 Python 模块名，例如 research_sutrack.runtime"
            )
        factory = _required_name(payload.get("factory", "build_sutrack_runtime"), "implementation.factory")

        raw_kwargs = payload.get("kwargs", {})
        if not isinstance(raw_kwargs, Mapping):
            raise SUTrackConfigurationError("implementation.kwargs 必须是 mapping")
        factory_kwargs = dict(raw_kwargs)
        if "params" in factory_kwargs:
            raise SUTrackConfigurationError(
                "implementation.kwargs.params 是保留参数；adapter 会自动向工厂传入 params"
            )

        raw_output = values.get("output", {})
        if not isinstance(raw_output, Mapping):
            raise SUTrackConfigurationError("SUTrack output 必须是 mapping")
        return cls(
            module=module,
            factory=factory,
            factory_kwargs=factory_kwargs,
            output=SUTrackOutputConfig.from_mapping(raw_output),
        )


def _required_name(value: Any, field_name: str) -> str:
    """校验用作输出键或 Python 属性名的非空字符串。"""

    name = str(value).strip()
    if not name or not _PYTHON_NAME.fullmatch(name):
        raise SUTrackConfigurationError(f"{field_name} 必须是合法的 Python 标识符")
    return name
