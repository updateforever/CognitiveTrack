"""按模型配置里的 ``backend`` 字段构造后端。

配置的 ``backend`` 决定推理走哪条路，其余字段由对应后端自己校验。新增后端只需
在 :data:`_BACKEND_BUILDERS` 注册，tracker 不需要改动。

非法 backend 名会直接报错并列出可用值，不静默回退 —— 静默回退到本地权重会在
无 GPU 的机器上表现为一次莫名的 OOM 或长时间挂起。
"""

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .base import GenerationConfig, VLMBackend
from .openai_api import OpenAIVLMBackend, OpenAIVLMConfig
from .qwen_vl import QwenVLBackend, QwenVLConfig

#: 需要按机器解析路径的字段，按 backend 分组。
#: tracker 拿到这些字段名后，用它自己的模型根目录规则做解析。
BACKEND_PATH_FIELDS: Dict[str, Tuple[str, ...]] = {
    "huggingface_qwen": ("model_path", "adapter_path"),
    # API 后端不加载权重，但仍需要 image processor 来确定模型像素空间。
    "openai_api": ("processor_path",),
}


def _build_huggingface_qwen(payload: Mapping[str, Any]) -> Tuple[VLMBackend, GenerationConfig]:
    config = QwenVLConfig.from_mapping(payload)
    return QwenVLBackend(config), config.generation


def _build_openai_api(payload: Mapping[str, Any]) -> Tuple[VLMBackend, GenerationConfig]:
    config = OpenAIVLMConfig.from_mapping(payload)
    return OpenAIVLMBackend(config), config.generation


_BACKEND_BUILDERS: Dict[str, Callable[[Mapping[str, Any]], Tuple[VLMBackend, GenerationConfig]]] = {
    "huggingface_qwen": _build_huggingface_qwen,
    "openai_api": _build_openai_api,
}

#: 模型配置省略 ``backend`` 时使用的默认值，保持既有配置向后兼容。
DEFAULT_BACKEND = "huggingface_qwen"


def available_backends() -> Tuple[str, ...]:
    """返回全部可用 backend 名。"""

    return tuple(sorted(_BACKEND_BUILDERS))


def resolve_backend_name(payload: Mapping[str, Any]) -> str:
    """取出并校验模型配置里的 backend 名。"""

    name = str(payload.get("backend", DEFAULT_BACKEND))
    if name not in _BACKEND_BUILDERS:
        raise ValueError(f"未知 backend {name!r}；可用值: {list(available_backends())}")
    return name


def build_backend(payload: Mapping[str, Any]) -> Tuple[VLMBackend, GenerationConfig]:
    """构造后端，同时返回它的生成配置。

    生成配置单独返回，这样 tracker 不必假设后端持有 Qwen 专属的 ``config``
    属性，测试替身也不需要伪造它。
    """

    name = resolve_backend_name(payload)
    data = dict(payload)
    data["backend"] = name
    return _BACKEND_BUILDERS[name](data)


def path_fields_for(backend_name: str) -> Tuple[str, ...]:
    """返回该 backend 中需要按机器解析的路径字段名。"""

    return BACKEND_PATH_FIELDS.get(backend_name, ())


def describe_backend(backend: VLMBackend, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """生成写入 run manifest 的后端描述。

    API 后端记录服务地址和坐标系参数，本地后端记录权重路径 —— 这些是复现一次
    实验的必要信息。任何情况下都不写 API key。
    """

    info: Dict[str, Any] = {
        "type": type(backend).__name__,
        "model_name": backend.model_name,
    }
    if isinstance(backend, OpenAIVLMBackend):
        info.update(
            backend="openai_api",
            base_url=backend.config.base_url,
            processor_path=backend.config.processor_path,
            image_format=backend.config.image_format,
            visual_grid=dict(
                factor=backend.grid_spec.factor,
                min_pixels=backend.grid_spec.min_pixels,
                max_pixels=backend.grid_spec.max_pixels,
            ),
        )
    else:
        info["backend"] = "huggingface_qwen"
        model_path = getattr(getattr(backend, "config", None), "model_path", None)
        if model_path:
            info["model_path"] = str(Path(model_path))
        adapter_path = getattr(getattr(backend, "config", None), "adapter_path", None)
        if adapter_path:
            info["adapter_path"] = str(Path(adapter_path))
    return info
