"""基于 YAML 的可移植环境配置。

配置只描述机器相关路径，不包含 tracker 算法超参数。加载优先级为：

1. 调用方显式传入的 ``config_path``；
2. 环境变量 ``COGTRACK_ENV``；
3. ``configs/env.local.yaml``；
4. ``configs/env.yaml``；
5. 以项目目录为基准的安全默认值。

YAML 中可使用 ``${ENV_VAR}`` 和 ``~``。相对路径统一相对 ``project_root``，避免
因启动命令所在目录不同而改变含义。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

_DATASET_ALIASES = {
    "videocube": "mgit",
    "videocube_test": "mgit",
    "videocube_val": "mgit",
}


def _project_root() -> Path:
    # .../CognitiveTrack/pytracking/evaluation/environment.py -> CognitiveTrack
    return Path(__file__).resolve().parents[2]


def _expand_path(value: str | os.PathLike[str], *, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve(strict=False)


@dataclass(frozen=True)
class EnvironmentSettings:
    """运行环境的不可变快照。

    ``dataset_roots`` 的标准键为 ``cognitivebench``、``lasot``、``tnl2k`` 和
    ``mgit``。不可变 dataclass 可以防止一次实验在运行中途被 tracker 意外改写。
    """

    project_root: Path
    results_path: Path
    dataset_roots: Mapping[str, Path] = field(default_factory=dict)
    model_root: Path | None = None
    source_file: Path | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def dataset_root(self, name: str) -> Path:
        """返回数据集根目录，不存在配置时立即失败而不是延迟到读图阶段。"""

        key = _DATASET_ALIASES.get(name.lower(), name.lower())
        try:
            root = self.dataset_roots[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.dataset_roots)) or "<无>"
            raise KeyError(f"数据集 {name!r} 未配置；当前可用: {available}") from exc
        return Path(root)

    def with_results_path(self, path: str | os.PathLike[str]) -> "EnvironmentSettings":
        """返回仅覆盖结果目录的新配置，便于 CLI 临时指定输出位置。"""

        return EnvironmentSettings(
            project_root=self.project_root,
            results_path=_expand_path(path, base=self.project_root),
            dataset_roots=self.dataset_roots,
            model_root=self.model_root,
            source_file=self.source_file,
            extras=self.extras,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"环境配置顶层必须是 mapping: {path}")
    return payload


def _find_config(explicit: str | os.PathLike[str] | None, project_root: Path) -> Path | None:
    if explicit is not None:
        path = Path(os.path.expandvars(os.path.expanduser(str(explicit))))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"环境配置不存在: {path}")
        return path.resolve()

    env_path = os.environ.get("COGTRACK_ENV")
    if env_path:
        return _find_config(env_path, project_root)

    for candidate in (project_root / "configs/env.local.yaml", project_root / "configs/env.yaml"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_environment(
    config_path: str | os.PathLike[str] | None = None,
    *,
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
) -> EnvironmentSettings:
    """加载并规范化环境配置。

    兼容两种常见 YAML 写法：推荐的 ``datasets: {lasot: ...}``，以及迁移阶段
    常见的 ``lasot_path: ...``。环境变量 ``COGTRACK_<NAME>_ROOT`` 可覆盖单个
    数据集路径，适合容器或集群作业。
    """

    default_project = _project_root()
    source_file = _find_config(config_path, default_project)
    payload = _read_yaml(source_file) if source_file else {}

    raw_project = payload.get("project_root", payload.get("workspace_root", default_project))
    project_root = _expand_path(raw_project, base=source_file.parent if source_file else default_project)

    raw_results = payload.get(
        "results_path",
        payload.get("results_root", payload.get("output_root", project_root / "outputs/tracking_results")),
    )
    results_path = _expand_path(raw_results, base=project_root)

    raw_datasets = payload.get("datasets", payload.get("dataset_roots", {})) or {}
    if not isinstance(raw_datasets, Mapping):
        raise TypeError("环境配置中的 datasets/dataset_roots 必须是 mapping")

    data_root_raw = (
        payload.get("dataset_root")
        or payload.get("data_root")
        or os.environ.get("COGTRACK_DATASET_ROOT")
        or os.environ.get("COGTRACK_DATA_ROOT")
    )
    data_root = _expand_path(data_root_raw, base=project_root) if data_root_raw else None
    dataset_roots: dict[str, Path] = {}
    for name in ("cognitivebench", "lasot", "tnl2k", "mgit"):
        raw_value = raw_datasets.get(name, payload.get(f"{name}_path"))
        env_value = os.environ.get(f"COGTRACK_{name.upper()}_ROOT")
        raw_value = env_value or raw_value
        if raw_value is None and data_root is not None:
            conventional_name = {
                "cognitivebench": "CognitiveBench",
                "lasot": "lasot",
                "tnl2k": "TNL2K",
                "mgit": "MGIT",
            }[name]
            raw_value = data_root / conventional_name
        if raw_value is not None:
            dataset_roots[name] = _expand_path(raw_value, base=project_root)

    model_raw = os.environ.get("COGTRACK_MODEL_ROOT") or payload.get("model_root")
    model_root = _expand_path(model_raw, base=project_root) if model_raw else None

    output_override = os.environ.get("COGTRACK_OUTPUT_ROOT")
    if output_override:
        results_path = _expand_path(output_override, base=project_root)

    if overrides:
        if "results_path" in overrides:
            results_path = _expand_path(overrides["results_path"], base=project_root)
        normalized_overrides = {
            _DATASET_ALIASES.get(str(name).lower(), str(name).lower()): value
            for name, value in overrides.items()
            if name != "results_path"
        }
        for name in ("cognitivebench", "lasot", "tnl2k", "mgit"):
            if name in normalized_overrides:
                dataset_roots[name] = _expand_path(normalized_overrides[name], base=project_root)

    known_keys = {
        "project_root",
        "workspace_root",
        "results_path",
        "results_root",
        "output_root",
        "datasets",
        "dataset_roots",
        "dataset_root",
        "data_root",
        "model_root",
        "cognitivebench_path",
        "lasot_path",
        "tnl2k_path",
        "mgit_path",
    }
    extras = {key: value for key, value in payload.items() if key not in known_keys}
    return EnvironmentSettings(
        project_root=project_root,
        results_path=results_path,
        dataset_roots=dataset_roots,
        model_root=model_root,
        source_file=source_file,
        extras=extras,
    )
