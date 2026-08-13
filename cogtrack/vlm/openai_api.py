"""OpenAI 兼容 API 后端（本地 vLLM / 远程服务共用）。

与本地 Hugging Face 后端相比，这里唯一真正困难的地方是坐标系。

``qwen_abs_pixel`` 协议约定模型输出的是**模型实际看到那张图**的绝对像素坐标。
本地后端能从 processor 的 ``image_grid_thw`` 反查这个尺寸；但 OpenAI 兼容接口
只回文本，服务端 smart_resize 成了什么尺寸我们无从知道。而这个尺寸是错不起的：
配置注释里记着实测结果 —— 协议弄错时 Qwen2.5-VL 有 68/69 个观测帧 IoU 恰好为 0。

本模块的做法是把这个未知量消掉，而不是去猜它：

    发图之前，先在客户端把图缩放到 smart_resize 的**不动点**（两边都是
    ``patch_size * merge_size`` 的整数倍，且总像素落在 ``[min_pixels,
    max_pixels]`` 内）。此时服务端再跑一次 smart_resize 只会原样返回，于是
    「我们发出去的尺寸」就等于「模型看到的尺寸」。

不动点是调 transformers 自己的 ``smart_resize`` 求出来的，不是在本仓库重写一份
缩放策略：transformers 改规则时我们跟着改，不会产生静默偏移。缩放参数
（factor / min_pixels / max_pixels）也一律从模型自己的 image processor 读，因为
它们随模型族变化 —— Qwen2.5-VL 的 factor 是 28，Qwen3-VL 是 32。

这条推理依赖一个前提：服务端 processor 的配置与我们读到的一致。若用
``--mm-processor-kwargs`` 之类参数改过服务端的 ``min_pixels/max_pixels``，必须
在 ``processor_overrides`` 里镜像同样的值，否则服务端会二次缩放。构造时会做一次
幂等自检，把配置不一致尽早暴露成显式错误。
"""

import base64
import io
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import (
    GenerationConfig,
    ImageInput,
    VLMBackend,
    VLMDependencyError,
    VLMInferenceError,
    VLMLoadError,
    VLMResponse,
)
from .generation import prepare_rgb_images

#: 服务端 processor 的默认视觉参数来源。这些值必须来自模型目录，不能硬编码。
_REQUIRED_PROCESSOR_ATTRS = ("patch_size", "merge_size", "min_pixels", "max_pixels")


@dataclass(frozen=True)
class OpenAIVLMConfig:
    """OpenAI 兼容多模态服务的连接与坐标系配置。"""

    #: 服务地址，例如 ``http://127.0.0.1:8000/v1``。
    base_url: str
    #: 服务端 ``--served-model-name``，会原样进入请求体。
    model_name: str
    #: 求 smart_resize 不动点所需的 image processor 目录。必填：没有它就无法
    #: 确定模型像素空间，``qwen_abs_pixel`` 也就无从谈起。
    processor_path: str
    #: 读取 API key 的环境变量名。本地 vLLM 通常任意值即可，但不写死在配置里。
    api_key_env: str = "LOCAL_VLLM_API_KEY"
    #: 环境变量缺失时使用的兜底 key，仅对不校验 key 的本地服务有意义。
    api_key_fallback: Optional[str] = "local-test-key"
    #: 覆盖服务端 processor 参数。仅当服务端用 mm-processor-kwargs 改过时才需要。
    processor_overrides: Mapping[str, int] = field(default_factory=dict)
    #: 先按最长边预缩，再求不动点。等价于本地后端的同名参数，用于控制视觉 token。
    max_image_side: Optional[int] = None
    #: 图像编码格式。默认 PNG：无损，保证同一帧多次实验的输入逐字节一致。
    image_format: str = "png"
    #: 仅 ``image_format="jpeg"`` 时生效。
    jpeg_quality: int = 95
    timeout_s: float = 120.0
    #: 瞬时故障重试次数。长序列评测里网络抖动不应让整条序列报废。
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        for field_name in ("base_url", "model_name", "processor_path", "api_key_env"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
        if not isinstance(self.generation, GenerationConfig):
            raise TypeError("generation 必须是 GenerationConfig")
        if self.image_format not in ("png", "jpeg"):
            raise ValueError("image_format 只允许 png / jpeg")
        if isinstance(self.jpeg_quality, bool) or not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("jpeg_quality 必须位于 [1, 100]")
        if isinstance(self.timeout_s, bool) or float(self.timeout_s) <= 0.0:
            raise ValueError("timeout_s 必须大于 0")
        if isinstance(self.max_retries, bool) or int(self.max_retries) < 0:
            raise ValueError("max_retries 不能为负数")
        if isinstance(self.retry_backoff_s, bool) or float(self.retry_backoff_s) < 0.0:
            raise ValueError("retry_backoff_s 不能为负数")
        if self.max_image_side is not None and (
            isinstance(self.max_image_side, bool) or int(self.max_image_side) <= 0
        ):
            raise ValueError("max_image_side 必须是正整数或 None")
        for key, value in dict(self.processor_overrides).items():
            if key not in _REQUIRED_PROCESSOR_ATTRS:
                raise ValueError(f"processor_overrides 只允许 {list(_REQUIRED_PROCESSOR_ATTRS)}，收到 {key!r}")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"processor_overrides[{key!r}] 必须是正整数")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OpenAIVLMConfig":
        """从项目 YAML 的扁平模型配置构造。

        与 :class:`QwenVLConfig.from_mapping` 保持同样约定：生成字段在 YAML 里
        保持扁平，读取时集中组装；未知字段显式报错，不让拼写错误被静默忽略。
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

        # vLLM 多卡评测通常是一卡一个服务端口。模型 YAML 保持可提交，具体
        # endpoint 由 worker 的环境变量注入；model_name 也允许同样方式切换
        # base / LoRA。这里只展开连接字段，路径字段仍由 tracker 的统一路径
        # 解析逻辑处理，避免两套规则互相覆盖。
        for key in ("base_url", "model_name", "api_key_env", "api_key_fallback"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = os.path.expandvars(value)

        valid_fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise ValueError(f"OpenAIVLMConfig 包含未知字段: {unknown}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """返回可写入 run manifest 的字典，不含 API key。"""

        payload = asdict(self)
        payload["generation"] = self.generation.to_dict()
        payload["processor_overrides"] = dict(self.processor_overrides)
        return payload


@dataclass(frozen=True)
class VisualGridSpec:
    """服务端 processor 的视觉网格参数。

    ``factor`` 是 smart_resize 的对齐粒度，等于 ``patch_size * merge_size``：
    Qwen2.5-VL 为 ``14 * 2 = 28``，Qwen3-VL 为 ``16 * 2 = 32``。硬编码 28 会
    静默弄坏 Qwen3-VL 这条线，所以一律从模型自己的配置读。
    """

    factor: int
    min_pixels: int
    max_pixels: int

    def __post_init__(self) -> None:
        for field_name in ("factor", "min_pixels", "max_pixels"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"VisualGridSpec.{field_name} 必须是正整数")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels 不能大于 max_pixels")


def load_visual_grid_spec(
    processor_path: str,
    overrides: Optional[Mapping[str, int]] = None,
    *,
    local_files_only: bool = True,
) -> VisualGridSpec:
    """从模型目录读取视觉网格参数。

    只加载 image processor：那是几 KB 的 JSON，不碰权重，也不需要 GPU。
    """

    try:
        from transformers import AutoImageProcessor
    except ImportError as error:  # pragma: no cover - 仅在缺少运行依赖时触发
        raise VLMDependencyError("读取视觉网格参数需要 transformers") from error

    resolved = Path(os.path.expandvars(os.path.expanduser(processor_path)))
    try:
        processor = AutoImageProcessor.from_pretrained(
            str(resolved),
            local_files_only=local_files_only,
            # 与本地后端保持一致：固定预处理实现，避免默认值变化影响复现。
            use_fast=False,
        )
    except Exception as error:
        raise VLMLoadError(f"加载 image processor 失败: {resolved}") from error

    values: Dict[str, int] = {}
    for name in _REQUIRED_PROCESSOR_ATTRS:
        value = getattr(processor, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise VLMLoadError(
                f"image processor 未提供可用的 {name}（读到 {value!r}）；"
                f"请在 processor_overrides 里显式给出。来源: {resolved}"
            )
        values[name] = int(value)
    values.update({key: int(val) for key, val in dict(overrides or {}).items()})

    return VisualGridSpec(
        factor=values["patch_size"] * values["merge_size"],
        min_pixels=values["min_pixels"],
        max_pixels=values["max_pixels"],
    )


def smart_resize_fixed_point(height: int, width: int, spec: VisualGridSpec) -> Tuple[int, int]:
    """返回 ``(height, width)`` 在 smart_resize 下的不动点。

    直接调 transformers 的实现，不在本仓库复制一份缩放策略。返回值满足
    ``smart_resize(h, w) == (h, w)``，因此把图缩到该尺寸后，服务端的预处理
    对它是恒等变换。
    """

    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    except ImportError as error:  # pragma: no cover - 仅在缺少运行依赖时触发
        raise VLMDependencyError("求 smart_resize 不动点需要 transformers 的 Qwen2-VL image processor") from error

    return smart_resize(
        height,
        width,
        factor=spec.factor,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
    )


def _verify_fixed_point(spec: VisualGridSpec) -> None:
    """自检：确认在本 spec 下不动点性质真的成立。

    这是对「服务端不会二次缩放」这个核心假设的直接检验。若 transformers 换了
    缩放策略、或 spec 参数自相矛盾，这里会立刻炸，而不是让坐标悄悄偏掉。
    """

    probes = ((1080, 1920), (720, 1280), (480, 640), (240, 1200), (57, 57))
    for height, width in probes:
        once = smart_resize_fixed_point(height, width, spec)
        twice = smart_resize_fixed_point(once[0], once[1], spec)
        if once != twice:
            raise VLMLoadError(
                "smart_resize 在当前视觉网格参数下不幂等，"
                f"{height}x{width} -> {once} -> {twice}；"
                f"qwen_abs_pixel 协议无法在 API 后端成立。spec={spec}"
            )


def _encode_image(image: Any, image_format: str, jpeg_quality: int) -> str:
    """把 PIL 图像编码成 data URI。"""

    buffer = io.BytesIO()
    if image_format == "jpeg":
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        mime = "image/jpeg"
    else:
        # PNG 无损：同一帧在多次实验里逐字节一致，符合本项目的可复现要求。
        image.save(buffer, format="PNG", optimize=False)
        mime = "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OpenAIVLMBackend(VLMBackend):
    """通过 OpenAI 兼容接口访问多模态模型。

    无状态：不持有权重，因此可以在多线程/多进程评测里自由复制。客户端对象
    本身复用，避免每帧重建连接池。
    """

    def __init__(self, config: OpenAIVLMConfig, *, client: Optional[Any] = None) -> None:
        self.config = config
        self.grid_spec = load_visual_grid_spec(config.processor_path, config.processor_overrides)
        # 构造时就检验核心假设，而不是等第一帧坐标偏了才发现。
        _verify_fixed_point(self.grid_spec)
        self._client = client
        self._last_retries = 0

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def is_loaded(self) -> bool:
        """远程后端没有本地权重；这里表示「客户端已就绪」。"""

        return self._client is not None

    @property
    def last_retries(self) -> int:
        """上一次 ``generate`` 实际重试的次数，便于运行日志记录服务稳定性。"""

        return self._last_retries

    def _resolve_api_key(self) -> str:
        key = os.environ.get(self.config.api_key_env)
        if key:
            return key
        if self.config.api_key_fallback:
            return self.config.api_key_fallback
        raise VLMLoadError(
            f"环境变量 {self.config.api_key_env} 未设置，且未配置 api_key_fallback"
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as error:
            raise VLMDependencyError(
                "API 后端需要 openai 包：pip install openai"
            ) from error
        self._client = OpenAI(
            api_key=self._resolve_api_key(),
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            # 重试在本类内部统一处理，避免两层退避叠加导致超时行为难以解释。
            max_retries=0,
        )
        return self._client

    def _prepare_images(self, images: Sequence[ImageInput]) -> Tuple[List[Any], Tuple[Tuple[int, int], ...]]:
        """预缩到 smart_resize 不动点，返回图像与其 ``(width, height)`` 列表。

        返回的尺寸即模型看到的尺寸，因为服务端对不动点尺寸的预处理是恒等的。
        """

        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover
            raise VLMDependencyError("图像编码需要 Pillow") from error

        prepared = prepare_rgb_images(images, self.config.max_image_side)
        resized: List[Any] = []
        sizes: List[Tuple[int, int]] = []
        for image in prepared:
            width, height = image.size
            target_height, target_width = smart_resize_fixed_point(height, width, self.grid_spec)
            if (target_width, target_height) != (width, height):
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized.append(image)
            sizes.append((target_width, target_height))
        return resized, tuple(sizes)

    def _build_messages(
        self,
        images: Sequence[Any],
        prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict[str, Any]]:
        """构造 OpenAI 兼容的多图消息。

        图像顺序与本地后端一致：pair/mosaic 都把待搜索的当前帧放在最后一张。
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")
        content: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": _encode_image(image, self.config.image_format, self.config.jpeg_quality)
                },
            }
            for image in images
        ]
        content.append({"type": "text", "text": prompt.strip()})
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _request_kwargs(generation: GenerationConfig) -> Dict[str, Any]:
        """把 GenerationConfig 映射到 OpenAI 请求参数。

        ``do_sample=False`` 对应 ``temperature=0``：OpenAI 协议没有单独的贪心
        开关，温度为 0 是各实现（含 vLLM）公认的确定性解码写法。
        """

        kwargs: Dict[str, Any] = {"max_tokens": generation.max_new_tokens}
        if generation.do_sample:
            kwargs["temperature"] = generation.temperature
            kwargs["top_p"] = generation.top_p
        else:
            kwargs["temperature"] = 0.0
        if generation.repetition_penalty != 1.0:
            # 非 OpenAI 标准字段，vLLM 通过 extra_body 接受。
            kwargs["extra_body"] = {"repetition_penalty": generation.repetition_penalty}
        return kwargs

    @staticmethod
    def _is_retryable(error: BaseException) -> bool:
        """只重试瞬时故障。

        400 类错误（提示过长、参数非法）重试多少次都是同样结果，重试只会把
        一次快速失败拖成几十秒，并掩盖真正的配置问题。
        """

        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status == 408 or status == 429 or status >= 500
        # 连接层异常没有 status_code，按可重试处理。
        return True

    def generate(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> VLMResponse:
        """执行一次多图生成。"""

        generation = generation_config or self.config.generation
        prepared, image_sizes = self._prepare_images(images)
        messages = self._build_messages(prepared, prompt, system_prompt)
        request_kwargs = self._request_kwargs(generation)
        client = self._ensure_client()

        # 计时包含重试：latency_ms 反映 tracker 真实等待的墙钟时间。
        started = time.perf_counter()
        self._last_retries = 0
        last_error: Optional[BaseException] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    **request_kwargs,
                )
                break
            except Exception as error:  # noqa: BLE001 - SDK 异常层次随版本变化
                last_error = error
                if attempt >= self.config.max_retries or not self._is_retryable(error):
                    raise VLMInferenceError(
                        f"OpenAI 兼容接口调用失败（尝试 {attempt + 1} 次）: {self.config.base_url}"
                    ) from error
                self._last_retries = attempt + 1
                if self.config.retry_backoff_s:
                    time.sleep(self.config.retry_backoff_s * (2**attempt))
        else:  # pragma: no cover - 循环必然 break 或 raise
            raise VLMInferenceError("OpenAI 兼容接口调用失败") from last_error

        latency_ms = (time.perf_counter() - started) * 1000.0
        text, prompt_tokens, generated_tokens = self._extract_completion(completion)
        return VLMResponse(
            text=text,
            model_name=self.model_name,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            # 不动点保证服务端预处理是恒等变换，所以发出去的尺寸就是模型看到的尺寸。
            image_sizes=image_sizes,
        )

    @staticmethod
    def _extract_completion(completion: Any) -> Tuple[str, Optional[int], Optional[int]]:
        """从响应中取出文本与 token 统计。

        推理模型（如 Qwen3-VL-Thinking）会把思维链放在 ``reasoning_content``，
        正式答案仍在 ``content``。这里只取 ``content``：解析器要的是 JSON，
        把思维链拼进去只会让「抽取首个 JSON 对象」抽到错的那个。
        """

        try:
            choices = completion.choices
            if not choices:
                raise VLMInferenceError("响应中没有 choices")
            message = choices[0].message
            text = getattr(message, "content", None)
        except AttributeError as error:
            raise VLMInferenceError("响应结构不符合 OpenAI 兼容格式") from error

        if text is None:
            reasoning = getattr(message, "reasoning_content", None)
            raise VLMInferenceError(
                "响应 content 为空"
                + ("（只返回了 reasoning_content，可能是 max_tokens 太小被截断）" if reasoning else "")
            )

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        generated_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return str(text), prompt_tokens, generated_tokens

    def unload(self) -> None:
        """释放 HTTP 客户端。无本地权重，不涉及显存。"""

        client = self._client
        self._client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - 关闭失败不应影响评测收尾
                    pass
