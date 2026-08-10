"""VLM 多模态输入的通用构造工具。"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import ImageInput, VLMDependencyError


def to_rgb_pil(image: ImageInput) -> Any:
    """把路径、PIL 图像或 numpy 数组统一转换为独立的 RGB PIL 图像。

    numpy 数组约定为 RGB；OpenCV 的 BGR 数据必须由调用方先显式转换。这样
    可以避免颜色空间在跟踪链路中被悄悄交换。
    """

    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:  # pragma: no cover - 仅在缺少运行依赖时触发
        raise VLMDependencyError("图像预处理需要 Pillow 和 numpy") from error

    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"图像不存在: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB").copy()

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()

    if isinstance(image, np.ndarray):
        if image.ndim not in (2, 3):
            raise ValueError(f"numpy 图像维度必须为 2 或 3，实际为 {image.ndim}")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(f"numpy 图像通道数必须为 1、3 或 4，实际为 {image.shape[2]}")
        array = image
        if array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError("numpy 图像必须是数值数组")
            if not np.isfinite(array).all():
                raise ValueError("numpy 图像包含 NaN 或 Inf")
            # 浮点输入若位于 [0, 1]，按常见图像约定映射至 [0, 255]。
            if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    raise TypeError(f"不支持的图像输入类型；期望路径、PIL.Image 或 numpy.ndarray，实际为 {type(image).__name__}")


def prepare_rgb_images(
    images: Sequence[ImageInput],
    max_image_side: Optional[int] = None,
) -> List[Any]:
    """校验并转换非空图像列表，可选限制最长边以降低视觉 token 数。"""

    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
        raise TypeError("images 必须是有序图像序列")
    if not images:
        raise ValueError("VLM 推理至少需要一张图像")
    if max_image_side is not None:
        if isinstance(max_image_side, bool) or not isinstance(max_image_side, int) or max_image_side <= 0:
            raise ValueError("max_image_side 必须是正整数或 None")

    prepared = [to_rgb_pil(image) for image in images]
    if max_image_side is None:
        return prepared

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise VLMDependencyError("图像缩放需要 Pillow") from error
    resized = []
    for image in prepared:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_image_side:
            resized.append(image)
            continue
        scale = max_image_side / float(longest)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized.append(image.resize(new_size, Image.Resampling.LANCZOS))
    return resized


def build_qwen_messages(
    images: Sequence[Any],
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按图像顺序构造 Qwen-VL chat messages。"""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt 必须是非空字符串")
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt.strip()})
    messages.append({"role": "user", "content": content})
    return messages
