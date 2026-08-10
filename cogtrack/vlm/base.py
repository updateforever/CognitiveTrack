"""视觉语言模型后端的抽象接口。

跟踪器只依赖 :class:`VLMBackend`，不需要知道模型来自 Hugging Face、vLLM
还是远程 API。当前首版实现本地 Hugging Face Qwen-VL，后续后端可遵循相同
请求/响应结构扩展。
"""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

ImageInput = Any


class VLMBackendError(RuntimeError):
    """VLM 后端错误基类。"""


class VLMDependencyError(VLMBackendError):
    """缺少模型推理所需的可选依赖。"""


class VLMLoadError(VLMBackendError):
    """本地模型或 processor 加载失败。"""


class VLMInferenceError(VLMBackendError):
    """模型已加载，但一次生成调用失败。"""


@dataclass(frozen=True)
class GenerationConfig:
    """各模型后端共享的最小生成配置。"""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须是正整数")
        for field_name in ("temperature", "top_p", "repetition_penalty"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} 必须是有限数值")
        if self.temperature < 0.0:
            raise ValueError("temperature 不能为负数")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty 必须大于 0")

    def to_generate_kwargs(self) -> Dict[str, Any]:
        """生成传给 ``model.generate`` 的参数，避免贪心模式无效警告。"""

        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        """返回便于写入 run manifest 的普通字典。"""

        return asdict(self)


@dataclass(frozen=True)
class VLMResponse:
    """一次 VLM 生成结果及其可复现元数据。"""

    text: str
    model_name: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    #: 各输入图在模型内部实际占用的像素尺寸 ``(width, height)``，按输入顺序。
    #: Qwen 系列 processor 会把图 smart_resize 到 28 的整数倍，模型的绝对像素
    #: 坐标输出就活在这个空间里，因此解析 ``qwen_abs_pixel`` 协议必须拿到它，
    #: 不能用原图尺寸代替。后端拿不到该信息时为 ``None``。
    image_sizes: Optional[Tuple[Tuple[int, int], ...]] = None

    def current_frame_size(self) -> Optional[Tuple[int, int]]:
        """返回最后一张输入图的模型像素尺寸。

        pair 和 mosaic 上下文都把待搜索的当前帧放在最后一张，因此这里等价于
        “模型看当前帧时用的尺寸”。
        """

        if not self.image_sizes:
            return None
        return self.image_sizes[-1]


class VLMBackend(ABC):
    """所有 VLM 推理实现必须遵循的抽象接口。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回写入实验记录的模型标识。"""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """模型权重是否已经加载到内存/显存。"""

    @abstractmethod
    def generate(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> VLMResponse:
        """对一组有序图像和文本 Prompt 执行一次生成。"""

    def generate_text(
        self,
        images: Sequence[ImageInput],
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """只需要文本时使用的轻量便利接口。"""

        return self.generate(
            images,
            prompt,
            system_prompt=system_prompt,
            generation_config=generation_config,
        ).text

    def unload(self) -> None:
        """可选释放资源；无状态远程后端无需覆盖。"""

        return None
