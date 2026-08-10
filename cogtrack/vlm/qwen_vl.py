"""Hugging Face 本地 Qwen2.5-VL / Qwen3-VL 多图推理后端。

模型采用“进程级共享 + tracker 级无状态句柄”：构造 tracker 时不占用
显存，首次观测才加载权重；同一进程内多个序列构造的 backend 共用同一份
model/processor，但序列记忆、状态机和 Prompt 仍完全属于 tracker 实例。默认
``local_files_only=True``，确保 smoke test 不会意外访问网络。
"""

import atexit
import gc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .base import (
    GenerationConfig,
    ImageInput,
    VLMBackend,
    VLMBackendError,
    VLMDependencyError,
    VLMInferenceError,
    VLMLoadError,
    VLMResponse,
)
from .generation import build_qwen_messages, prepare_rgb_images


@dataclass(frozen=True)
class QwenVLConfig:
    """本地 Qwen-VL 权重和运行设备配置。"""

    model_path: str
    adapter_path: Optional[str] = None
    model_name: Optional[str] = None
    model_family: str = "auto"
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"
    device: Optional[str] = None
    attn_implementation: Optional[str] = "flash_attention_2"
    fallback_attention: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = True
    processor_use_fast: bool = False
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    max_image_side: Optional[int] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path 必须是非空字符串")
        if self.model_name is not None and (not isinstance(self.model_name, str) or not self.model_name.strip()):
            raise ValueError("model_name 必须是非空字符串或 None")
        if self.adapter_path is not None and (
            not isinstance(self.adapter_path, str) or not self.adapter_path.strip()
        ):
            raise ValueError("adapter_path 必须是非空字符串或 None")
        if not isinstance(self.generation, GenerationConfig):
            raise TypeError("generation 必须是 GenerationConfig")
        for field_name in ("device_map", "device", "attn_implementation"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} 必须是非空字符串或 None")
        for field_name in (
            "fallback_attention",
            "trust_remote_code",
            "local_files_only",
            "processor_use_fast",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} 必须是 bool")
        valid_families = {"auto", "qwen2_5_vl", "qwen3_vl"}
        if self.model_family not in valid_families:
            raise ValueError(f"model_family 必须是 {sorted(valid_families)} 之一")
        valid_dtypes = {"auto", "bfloat16", "float16", "float32"}
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"torch_dtype 必须是 {sorted(valid_dtypes)} 之一")
        for field_name in ("min_pixels", "max_pixels", "max_image_side"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{field_name} 必须是正整数或 None")
        if self.min_pixels and self.max_pixels and self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels 不能大于 max_pixels")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QwenVLConfig":
        """从项目 YAML 的扁平模型配置构造后端配置。

        ``max_new_tokens`` 等生成字段在 YAML 中保持扁平，读取时集中组装为
        ``GenerationConfig``；未知字段会明确报错，避免拼写错误被静默忽略。
        """

        data = dict(values)
        data.pop("backend", None)
        data.pop("_config_path", None)
        generation_fields = {
            key: data.pop(key)
            for key in (
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_p",
                "repetition_penalty",
            )
            if key in data
        }
        if "generation" in data and generation_fields:
            raise ValueError("generation 嵌套配置与扁平生成字段不能同时使用")
        if "generation" in data and isinstance(data["generation"], Mapping):
            data["generation"] = GenerationConfig(**dict(data["generation"]))
        elif generation_fields:
            data["generation"] = GenerationConfig(**generation_fields)

        valid_fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise ValueError(f"QwenVLConfig 包含未知字段: {unknown}")
        return cls(**data)


RuntimeFactory = Callable[[QwenVLConfig], Tuple[Any, Any]]


@dataclass
class _SharedQwenRuntime:
    """进程级共享的模型资源。

    ``inference_lock`` 与模型实例绑定，因此不同 backend 句柄也不会
    并发进入一个非线程安全的 ``generate``。``factory`` 保留强引用，
    避免测试注入的短生命周期 callable 被回收后 ``id`` 复用。
    """

    model: Any
    processor: Any
    factory: RuntimeFactory = field(repr=False)
    inference_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_MODEL_CACHE: Dict[Tuple[Any, ...], _SharedQwenRuntime] = {}
_MODEL_CACHE_LOCK = threading.RLock()


def _resolve_family(config: QwenVLConfig) -> str:
    if config.model_family != "auto":
        return config.model_family
    lower_name = config.model_path.lower().replace("-", "_")
    return "qwen3_vl" if "qwen3" in lower_name else "qwen2_5_vl"


def _resolve_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch_module, dtype_name)


def _load_huggingface_runtime(config: QwenVLConfig) -> Tuple[Any, Any]:
    """加载一份 Hugging Face model/processor。

    该函数是默认 runtime factory；独立函数形式便于单元测试注入
    fake factory，在无 GPU/无真实权重时验证缓存复用。
    """

    try:
        import torch
        from transformers import AutoProcessor

        if _resolve_family(config) == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError as error:
        raise VLMDependencyError("本地 Qwen-VL 推理需要 torch、transformers 及对应 Qwen 模型类") from error

    model_kwargs: Dict[str, Any] = {
        # Transformers 4.57 起正式使用 ``dtype``。
        "dtype": _resolve_dtype(torch, config.torch_dtype),
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    processor_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        # 固定预处理实现，避免 Transformers 默认值变化影响复现。
        "use_fast": config.processor_use_fast,
    }
    if config.min_pixels is not None:
        processor_kwargs["min_pixels"] = config.min_pixels
    if config.max_pixels is not None:
        processor_kwargs["max_pixels"] = config.max_pixels

    try:
        try:
            model = ModelClass.from_pretrained(config.model_path, **model_kwargs)
        except Exception:
            if not config.attn_implementation or not config.fallback_attention:
                raise
            # flash-attn 未安装或 GPU 不支持时移除 attention 参数重试。
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            model = ModelClass.from_pretrained(config.model_path, **fallback_kwargs)
        if config.device_map is None and config.device:
            model = model.to(config.device)
        if config.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise VLMDependencyError("加载 LoRA adapter 需要 peft") from error
            model = PeftModel.from_pretrained(
                model,
                config.adapter_path,
                is_trainable=False,
            )
        model.eval()
        processor = AutoProcessor.from_pretrained(config.model_path, **processor_kwargs)
    except Exception as error:
        raise VLMLoadError(f"加载本地 Qwen-VL 失败: {config.model_path}") from error
    return model, processor


def _runtime_cache_key(config: QwenVLConfig, factory: RuntimeFactory) -> Tuple[Any, ...]:
    """只使用影响权重/预处理的字段生成缓存键。

    生成长度、temperature 等属于单次请求，不应导致重复加载权重。
    """

    canonical_path = str(Path(config.model_path).expanduser().resolve(strict=False))
    canonical_adapter = (
        str(Path(config.adapter_path).expanduser().resolve(strict=False))
        if config.adapter_path is not None
        else None
    )
    return (
        id(factory),
        canonical_path,
        canonical_adapter,
        _resolve_family(config),
        config.torch_dtype,
        config.device_map,
        config.device,
        config.attn_implementation,
        config.fallback_attention,
        config.trust_remote_code,
        config.local_files_only,
        config.processor_use_fast,
        config.min_pixels,
        config.max_pixels,
    )


def _release_runtime(runtime: _SharedQwenRuntime) -> None:
    """等待在途推理完成后清空一份共享资源。"""

    with runtime.inference_lock:
        runtime.model = None
        runtime.processor = None


def _collect_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def clear_qwen_model_cache() -> int:
    """显式清空进程内所有 Qwen 权重，返回清理的 runtime 数量。

    常规的单序列 ``tracker.close()`` 只释放轻量句柄，不调用本函数；
    只有整个评测任务结束或需要切换大模型时才应显式清理。
    """

    with _MODEL_CACHE_LOCK:
        runtimes = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
        for runtime in runtimes:
            _release_runtime(runtime)
    if runtimes:
        _collect_accelerator_memory()
    return len(runtimes)


def qwen_model_cache_size() -> int:
    """返回当前进程已缓存的模型 runtime 数，便于运行时自检。"""

    with _MODEL_CACHE_LOCK:
        return len(_MODEL_CACHE)


class HuggingFaceQwenVLBackend(VLMBackend):
    """同时支持 Qwen2.5-VL 和 Qwen3-VL 的本地后端。"""

    def __init__(self, config: QwenVLConfig, *, runtime_factory: Optional[RuntimeFactory] = None) -> None:
        self.config = config
        self._runtime_factory = runtime_factory or _load_huggingface_runtime
        self._cache_key = _runtime_cache_key(config, self._runtime_factory)
        self._runtime: Optional[_SharedQwenRuntime] = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """默认使用本地目录名作为简洁模型标识。"""

        if self.config.model_name:
            return self.config.model_name
        path = Path(self.config.model_path.rstrip("/"))
        return path.name or self.config.model_path

    @property
    def is_loaded(self) -> bool:
        runtime = self._runtime
        return runtime is not None and runtime.model is not None and runtime.processor is not None

    def _load_once(self) -> None:
        """双重检查加锁，从进程缓存取得或创建共享 runtime。"""

        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            with _MODEL_CACHE_LOCK:
                runtime = _MODEL_CACHE.get(self._cache_key)
                if runtime is None or runtime.model is None or runtime.processor is None:
                    try:
                        model, processor = self._runtime_factory(self.config)
                    except VLMBackendError:
                        raise
                    except Exception as error:
                        raise VLMLoadError(f"加载本地 Qwen-VL 失败: {self.config.model_path}") from error
                    if model is None or processor is None:
                        raise VLMLoadError("Qwen runtime factory 必须同时返回 model 和 processor")
                    runtime = _SharedQwenRuntime(model, processor, self._runtime_factory)
                    _MODEL_CACHE[self._cache_key] = runtime
                # 最后赋值，加载失败时 backend 仍保持未加载。
                self._runtime = runtime

    @staticmethod
    def _model_device(model: Any) -> Any:
        """寻找非 meta 参数所在设备，适配 device_map='auto'。"""

        for parameter in model.parameters():
            if getattr(parameter.device, "type", None) != "meta":
                return parameter.device
        return model.device

    @staticmethod
    def _vision_inputs(messages: Any, fallback_images: Sequence[Any]) -> Tuple[Any, Any]:
        """优先用 qwen-vl-utils 解析消息，缺失时使用纯图像等价路径。"""

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            # 本任务不含视频，直接传有序 PIL 图像与官方 messages 语义一致。
            return list(fallback_images), None
        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs

    def generate(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> VLMResponse:
        """执行 Qwen-VL 多图生成，返回纯新增 token 对应的文本。"""

        rgb_images = prepare_rgb_images(images, self.config.max_image_side)
        messages = build_qwen_messages(rgb_images, prompt, system_prompt)
        generation = generation_config or self.config.generation
        started = time.perf_counter()

        # 清理缓存可能与请求并发。获得 runtime 锁后再检查一次；如果
        # 资源已被显式清空，释放旧句柄并重新从缓存加载。
        while True:
            self._load_once()
            runtime = self._runtime
            if runtime is None:  # pragma: no cover - 仅用于防御未知竞态
                raise VLMInferenceError("Qwen-VL runtime 未初始化")
            runtime.inference_lock.acquire()
            if runtime.model is not None and runtime.processor is not None:
                break
            runtime.inference_lock.release()
            self._runtime = None

        # HF generate 和部分量化后端不是线程安全的；所有共享句柄串行访问。
        try:
            try:
                import torch

                text = runtime.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = self._vision_inputs(messages, rgb_images)
                processor_kwargs: Dict[str, Any] = {
                    "text": [text],
                    "images": image_inputs,
                    "padding": True,
                    "return_tensors": "pt",
                }
                if video_inputs:
                    processor_kwargs["videos"] = video_inputs
                inputs = runtime.processor(**processor_kwargs)
                # 在搬到显卡前读出 processor 真正使用的视觉网格。模型的绝对
                # 像素坐标输出活在这个空间里，而不是我们传进去的图像尺寸。
                image_sizes = self._model_image_sizes(runtime.processor, inputs)
                inputs = inputs.to(self._model_device(runtime.model))

                with torch.inference_mode():
                    generated_ids = runtime.model.generate(
                        **inputs,
                        **generation.to_generate_kwargs(),
                    )
                if hasattr(generated_ids, "sequences"):
                    generated_ids = generated_ids.sequences
                input_length = int(inputs.input_ids.shape[1])
                trimmed_ids = generated_ids[:, input_length:]
                decoded = runtime.processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                output_text = decoded[0] if decoded else ""
                generated_tokens = int(trimmed_ids.shape[1])
            except Exception as error:
                raise VLMInferenceError("Qwen-VL 本地多图生成失败") from error
        finally:
            runtime.inference_lock.release()

        latency_ms = (time.perf_counter() - started) * 1000.0
        return VLMResponse(
            text=output_text,
            model_name=self.model_name,
            latency_ms=latency_ms,
            prompt_tokens=input_length,
            generated_tokens=generated_tokens,
            image_sizes=image_sizes,
        )

    @staticmethod
    def _model_image_sizes(processor: Any, inputs: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
        """从 processor 输出反查每张图在模型里的像素尺寸。

        ``image_grid_thw`` 的单位是 patch，乘 ``patch_size`` 得到像素。这里直接
        用 processor 自己算出的结果，而不是在本仓库重写一份 ``smart_resize``，
        这样 transformers 改变缩放规则时不会产生静默偏移。拿不到就返回 None，
        由调用方决定是否报错。
        """

        grid = None
        try:
            grid = inputs["image_grid_thw"]
        except (KeyError, TypeError):
            grid = getattr(inputs, "image_grid_thw", None)
        if grid is None:
            return None
        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
        if not isinstance(patch_size, int) or patch_size <= 0:
            return None
        sizes = []
        for row in grid.tolist():
            if len(row) != 3:
                return None
            _, grid_h, grid_w = row
            sizes.append((int(grid_w) * patch_size, int(grid_h) * patch_size))
        return tuple(sizes)

    def release(self) -> None:
        """释放当前 backend 的轻量句柄，保留进程级权重缓存。"""

        with self._load_lock:
            self._runtime = None

    def unload(self) -> None:
        """显式清理该配置对应的共享权重。

        该操作会使其他共享同一 runtime 的 backend 句柄失效；它们在下次
        ``generate`` 时会安全重载。逐序列关闭 tracker 应调用 ``release``。
        """

        with _MODEL_CACHE_LOCK:
            runtime = _MODEL_CACHE.pop(self._cache_key, None)
            if runtime is not None:
                _release_runtime(runtime)
        self.release()
        if runtime is not None:
            _collect_accelerator_memory()


# 简短别名便于配置工厂和用户代码引用。
QwenVLBackend = HuggingFaceQwenVLBackend


# 异常退出时也尽力释放大块 CPU/GPU 内存。
atexit.register(clear_qwen_model_cache)
