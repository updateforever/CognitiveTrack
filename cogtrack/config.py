"""项目配置加载工具。

本模块只负责把 YAML 转换为普通字典，并处理环境变量覆盖与相对路径。
它刻意不依赖模型、数据集或 tracker，避免配置系统反向耦合业务代码。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_OVERRIDES = {
    "dataset_root": "COGTRACK_DATASET_ROOT",
    "model_root": "COGTRACK_MODEL_ROOT",
    "output_root": "COGTRACK_OUTPUT_ROOT",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 映射，并把配置来源记录在 ``_config_path`` 中。

    Args:
        path: YAML 文件路径。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: YAML 顶层不是映射。
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {config_path}")

    result = dict(data)
    result["_config_path"] = str(config_path)
    return result


def load_environment(path: str | Path) -> dict[str, Any]:
    """读取本机环境配置，并应用显式环境变量覆盖。

    环境变量只覆盖机器相关的三个根目录，不允许静默改写实验参数。
    数据集独立路径仍可通过 YAML 的 ``datasets`` 映射指定。
    """

    config = load_yaml(path)
    for key, env_name in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            config[key] = str(Path(value).expanduser().resolve())
    return config


def resolve_config_reference(owner: Mapping[str, Any], value: str | Path) -> Path:
    """相对拥有者配置文件解析另一个配置路径。

    例如 tracker YAML 中的 ``../models/qwen25vl_7b.yaml`` 应相对 tracker
    YAML 所在目录，而不是相对当前 shell 工作目录。
    """

    ref = Path(value).expanduser()
    if ref.is_absolute():
        return ref.resolve()

    owner_path = owner.get("_config_path")
    if not owner_path:
        raise ValueError("解析相对配置引用时缺少 _config_path")
    return (Path(str(owner_path)).parent / ref).resolve()


def require_keys(config: Mapping[str, Any], *keys: str) -> None:
    """检查必要配置项，并一次性报告全部缺失字段。"""

    missing = [key for key in keys if key not in config or config[key] in (None, "")]
    if missing:
        source = config.get("_config_path", "<memory>")
        raise ValueError(f"配置 {source} 缺少必要字段: {', '.join(missing)}")
