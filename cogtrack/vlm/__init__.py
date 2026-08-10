"""VLM 后端、生成配置和严格输出解析。"""

from .base import (
    GenerationConfig,
    VLMBackend,
    VLMBackendError,
    VLMDependencyError,
    VLMInferenceError,
    VLMLoadError,
    VLMResponse,
)
from .factory import (
    DEFAULT_BACKEND,
    available_backends,
    build_backend,
    describe_backend,
    path_fields_for,
    resolve_backend_name,
)
from .openai_api import (
    OpenAIVLMBackend,
    OpenAIVLMConfig,
    VisualGridSpec,
    load_visual_grid_spec,
    smart_resize_fixed_point,
)
from .parser import (
    ParsedIdentityOutput,
    ParsedTrackingOutput,
    parse_identity_output,
    parse_tracking_output,
)
from .qwen_vl import (
    HuggingFaceQwenVLBackend,
    QwenVLBackend,
    QwenVLConfig,
    clear_qwen_model_cache,
    qwen_model_cache_size,
)

__all__ = [
    "DEFAULT_BACKEND",
    "GenerationConfig",
    "HuggingFaceQwenVLBackend",
    "OpenAIVLMBackend",
    "OpenAIVLMConfig",
    "ParsedIdentityOutput",
    "ParsedTrackingOutput",
    "QwenVLBackend",
    "QwenVLConfig",
    "VLMBackend",
    "VLMBackendError",
    "VLMDependencyError",
    "VLMInferenceError",
    "VLMLoadError",
    "VLMResponse",
    "VisualGridSpec",
    "available_backends",
    "build_backend",
    "clear_qwen_model_cache",
    "describe_backend",
    "load_visual_grid_spec",
    "parse_identity_output",
    "parse_tracking_output",
    "path_fields_for",
    "qwen_model_cache_size",
    "resolve_backend_name",
    "smart_resize_fixed_point",
]
