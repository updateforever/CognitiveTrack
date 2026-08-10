"""跟踪器描述、YAML 参数加载与动态构建。"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from pytracking.trackers.base import BaseTracker, TrackerParams

from .environment import EnvironmentSettings

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class TrackerSpec:
    """一次 tracker 实验的可序列化说明。

    默认模块约定为 ``pytracking.trackers.<name>``，并调用模块的
    ``get_tracker_class()``。也可显式传 ``module``/``class_name`` 接入外部实现。
    """

    name: str
    parameter_name: str = "default"
    run_id: int | None = None
    display_name: str | None = None
    config_path: str | Path | None = None
    module: str | None = None
    class_name: str | None = None

    def __post_init__(self) -> None:
        for label, value in (("tracker name", self.name), ("parameter name", self.parameter_name)):
            if not _SAFE_NAME.fullmatch(value):
                raise ValueError(f"{label} 含不安全字符: {value!r}")
        if self.run_id is not None and self.run_id < 0:
            raise ValueError("run_id 不能为负数")

    @property
    def result_parameter_name(self) -> str:
        if self.run_id is None:
            return self.parameter_name
        return f"{self.parameter_name}_{self.run_id:03d}"

    def result_directory(self, environment: EnvironmentSettings, dataset_name: str) -> Path:
        return environment.results_path / self.name / self.result_parameter_name / dataset_name


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"tracker 配置顶层必须是 mapping: {path}")
    # 允许文件含 model/dataset 等并列配置；BaseTracker 获得完整快照。
    return payload


def resolve_tracker_config(spec: TrackerSpec, environment: EnvironmentSettings) -> Path | None:
    """解析参数文件；``default`` 且文件不存在时允许使用空参数。"""

    if spec.config_path is not None:
        path = Path(spec.config_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"tracker 配置不存在: {path}")
        return path.resolve()

    root = environment.project_root / "configs" / "trackers"
    candidates = (
        root / spec.name / f"{spec.parameter_name}.yaml",
        root / f"{spec.parameter_name}.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    if spec.parameter_name == "default":
        return None
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"未找到参数 {spec.parameter_name!r}；检查过: {locations}")


def load_tracker_params(
    spec: TrackerSpec,
    environment: EnvironmentSettings,
    *,
    runtime: Mapping[str, Any] | None = None,
) -> TrackerParams:
    config_path = resolve_tracker_config(spec, environment)
    payload = _load_yaml(config_path) if config_path else {}
    if config_path is not None:
        # 高层 tracker 可据此把 model_config 等相对引用解析到当前 YAML 目录。
        payload.setdefault("_config_path", str(config_path))
    payload.setdefault("tracker_name", spec.name)
    payload.setdefault("parameter_name", spec.parameter_name)
    payload.setdefault("run_id", spec.run_id)
    if runtime:
        payload.setdefault("runtime", {}).update(dict(runtime))
    return TrackerParams(payload)


def get_tracker_class(spec: TrackerSpec) -> type[BaseTracker]:
    """动态导入 tracker 类，并验证其遵循 ``BaseTracker`` 接口。"""

    module_name = spec.module or f"pytracking.trackers.{spec.name}"
    module = importlib.import_module(module_name)
    if spec.class_name:
        tracker_class = getattr(module, spec.class_name)
    elif hasattr(module, "get_tracker_class"):
        tracker_class = module.get_tracker_class()
    else:
        raise AttributeError(f"{module_name} 缺少 get_tracker_class()；请在 TrackerSpec 中指定 class_name")
    if not isinstance(tracker_class, type) or not issubclass(tracker_class, BaseTracker):
        raise TypeError(f"{module_name} 返回的 tracker 必须继承 BaseTracker")
    return tracker_class


def build_tracker(
    spec: TrackerSpec,
    environment: EnvironmentSettings,
    *,
    dataset_name: str,
    runtime: Mapping[str, Any] | None = None,
) -> BaseTracker:
    """为一个序列构建全新的 tracker 实例。"""

    environment_runtime = {
        "project_root": str(environment.project_root),
        "model_root": str(environment.model_root) if environment.model_root is not None else None,
    }
    params = load_tracker_params(
        spec,
        environment,
        # 机器相关根目录始终由 EnvironmentSettings 注入；显式 runtime 仍可
        # 覆盖序列名、结果目录等一次性上下文，但不需要每个调用方重复传路径。
        runtime={
            "dataset_name": dataset_name,
            **environment_runtime,
            **dict(runtime or {}),
        },
    )
    return get_tracker_class(spec)(params)
