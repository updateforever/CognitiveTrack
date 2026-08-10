"""跟踪器统一生命周期与结构化参数容器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np


class TrackerParams(dict[str, Any]):
    """同时支持 ``params.key`` 与 ``params['key']`` 的递归参数映射。

    pytracking/SUTrack 生态普遍使用属性访问，而 YAML 天然产生字典。这个小型
    适配层避免 tracker 中充斥格式转换，同时保留标准字典的序列化能力。
    """

    def __init__(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        merged = {**dict(values or {}), **kwargs}
        for key, value in merged.items():
            self[key] = self._convert(value)

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, TrackerParams):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._convert(value)

    def clone(self) -> "TrackerParams":
        return TrackerParams(deepcopy(dict(self)))


class BaseTracker(ABC):
    """所有单目标 tracker 必须遵守的最小接口。

    输入图像为 ``H x W x 3`` 的 RGB ``numpy.ndarray``；bbox 为像素级 xywh。
    输出至少可含 ``target_bbox``，其余认知字段由具体 tracker 自由扩展。
    """

    def __init__(self, params: Mapping[str, Any] | TrackerParams | None = None) -> None:
        self.params = params if isinstance(params, TrackerParams) else TrackerParams(params)

    @abstractmethod
    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any] | None:
        """使用首帧及 ``info['init_bbox']`` 初始化 tracker。"""

    @abstractmethod
    def track(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any] | None:
        """处理一帧并更新内部状态。runner 保证所有原始帧都会调用本方法。"""

    def describe_runtime(self) -> dict[str, Any]:
        """返回复现本次运行所需的运行时信息，由 runner 写入 manifest。

        用于记录那些只有 tracker 构造完才知道、且不在配置文件里的东西：实际
        连上的服务地址、生效的模型标识、坐标系参数等。默认为空，需要时覆盖。

        实现必须只返回可 JSON 序列化的值，且绝不包含凭据。
        """

        return {}

    def close(self) -> None:
        """释放可选资源；默认无操作，backend 可覆盖。"""

        return None
