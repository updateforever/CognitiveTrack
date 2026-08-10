"""边界框格式校验和显式坐标转换。

框架内部统一使用像素坐标 ``xywh``，其中右/下边界采用半开区间语义。VLM
侧支持两种协议，由调用方显式选择，本模块从不根据数值大小猜测坐标系：

``norm1000``
    ``[0, 1000]`` 相对 ``xyxy``。Qwen2-VL 和 Qwen3-VL 使用该协议；本仓库的
    Qwen3-VL 训练视图与推理配置也使用它。
``qwen_abs_pixel``
    模型输入图自身像素网格里的绝对 ``xyxy``。Qwen2.5-VL 使用该协议，其零样本、
    SFT 和微调后推理都必须保持一致。

注意 ``qwen_abs_pixel`` 的参考系是**模型真正看到的那张图**，不是原始视频
帧：processor 会把图 smart_resize 到 28 的整数倍。所以转换需要同时给出模型
像素空间尺寸和原图尺寸，见 :func:`model_pixel_xyxy_to_pixel_xywh`。
"""

import math
from numbers import Real
from typing import Optional, Sequence, Tuple

from .exceptions import BoundingBoxError

BBoxXYWH = Tuple[float, float, float, float]
BBoxXYXY = Tuple[float, float, float, float]

#: Qwen2-VL / Qwen3-VL 使用的 [0,1000] 相对坐标协议。
BBOX_PROTOCOL_NORM1000 = "norm1000"
#: Qwen2.5-VL 协议：模型输入图自身像素网格里的绝对坐标。
BBOX_PROTOCOL_QWEN_ABS_PIXEL = "qwen_abs_pixel"

BBOX_PROTOCOLS = (BBOX_PROTOCOL_NORM1000, BBOX_PROTOCOL_QWEN_ABS_PIXEL)

#: 各协议在模型 JSON 里使用的字段名。字段名必须随协议改变，否则模型和读代码
#: 的人都会被 ``norm1000`` 这个名字误导成需要归一化。
_BBOX_PROTOCOL_JSON_KEYS = {
    BBOX_PROTOCOL_NORM1000: "bbox_norm1000_xyxy",
    BBOX_PROTOCOL_QWEN_ABS_PIXEL: "bbox_pixel_xyxy",
}


def validate_bbox_protocol(protocol: str) -> str:
    """校验 bbox 协议名，非法值直接报错而不静默回退。"""

    if not isinstance(protocol, str) or protocol not in BBOX_PROTOCOLS:
        raise BoundingBoxError(f"bbox_protocol 必须是 {list(BBOX_PROTOCOLS)} 之一，实际为 {protocol!r}")
    return protocol


def bbox_protocol_json_key(protocol: str) -> str:
    """返回该协议在模型 JSON 输出里使用的 bbox 字段名。"""

    return _BBOX_PROTOCOL_JSON_KEYS[validate_bbox_protocol(protocol)]


def _as_finite_box(values: Sequence[float], name: str) -> Tuple[float, float, float, float]:
    """把输入严格转换为四个有限浮点数。"""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BoundingBoxError(f"{name} 必须是长度为 4 的数值序列")
    if len(values) != 4:
        raise BoundingBoxError(f"{name} 长度必须为 4，实际为 {len(values)}")

    converted = []
    for index, value in enumerate(values):
        # bool 是 int 的子类，但在坐标协议中没有合理含义，因此显式拒绝。
        if isinstance(value, bool) or not isinstance(value, Real):
            raise BoundingBoxError(f"{name}[{index}] 必须是数值，实际为 {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number):
            raise BoundingBoxError(f"{name}[{index}] 必须是有限数值")
        converted.append(number)
    return tuple(converted)  # type: ignore[return-value]


def validate_image_size(width: int, height: int) -> Tuple[int, int]:
    """校验图像尺寸并返回标准整数二元组。"""

    if isinstance(width, bool) or isinstance(height, bool):
        raise BoundingBoxError("图像宽高不能是布尔值")
    if not isinstance(width, int) or not isinstance(height, int):
        raise BoundingBoxError("图像宽高必须是整数")
    if width <= 0 or height <= 0:
        raise BoundingBoxError(f"图像宽高必须大于 0，实际为 ({width}, {height})")
    return width, height


def validate_xywh(
    bbox: Sequence[float],
    image_size: Optional[Tuple[int, int]] = None,
) -> BBoxXYWH:
    """校验像素 ``xywh`` 框，可选检查其是否位于图像内。"""

    x, y, width, height = _as_finite_box(bbox, "bbox_xywh")
    if width <= 0.0 or height <= 0.0:
        raise BoundingBoxError("bbox_xywh 的宽高必须大于 0")
    if image_size is not None:
        image_width, image_height = validate_image_size(*image_size)
        tolerance = 1e-6
        if x < -tolerance or y < -tolerance:
            raise BoundingBoxError("bbox_xywh 左上角不能位于图像外")
        if x + width > image_width + tolerance or y + height > image_height + tolerance:
            raise BoundingBoxError("bbox_xywh 右下角超出图像边界")
    return x, y, width, height


def validate_xyxy(bbox: Sequence[float]) -> BBoxXYXY:
    """校验一般 ``xyxy`` 框，不对其坐标范围作假设。"""

    x1, y1, x2, y2 = _as_finite_box(bbox, "bbox_xyxy")
    if x2 <= x1 or y2 <= y1:
        raise BoundingBoxError("bbox_xyxy 必须满足 x2 > x1 且 y2 > y1")
    return x1, y1, x2, y2


def validate_norm1000_xyxy(bbox: Sequence[float]) -> BBoxXYXY:
    """校验 ``[0, 1000]`` 归一化 ``xyxy`` 模型框。"""

    normalized = validate_xyxy(bbox)
    if any(value < 0.0 or value > 1000.0 for value in normalized):
        raise BoundingBoxError("bbox_norm1000_xyxy 的每个坐标都必须位于 [0, 1000]")
    return normalized


def xywh_to_xyxy(bbox: Sequence[float]) -> BBoxXYXY:
    """把像素 ``xywh`` 转为同坐标系的 ``xyxy``。"""

    x, y, width, height = validate_xywh(bbox)
    return x, y, x + width, y + height


def xyxy_to_xywh(bbox: Sequence[float]) -> BBoxXYWH:
    """把 ``xyxy`` 转为同坐标系的 ``xywh``。"""

    x1, y1, x2, y2 = validate_xyxy(bbox)
    return x1, y1, x2 - x1, y2 - y1


def clip_xywh(bbox: Sequence[float], image_width: int, image_height: int) -> BBoxXYWH:
    """把像素框裁剪到图像边界；完全落在图像外时抛出异常。"""

    image_width, image_height = validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = xywh_to_xyxy(bbox)
    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise BoundingBoxError("边界框裁剪后没有有效面积")
    return x1, y1, x2 - x1, y2 - y1


def norm1000_xyxy_to_pixel_xywh(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> BBoxXYWH:
    """把模型的 norm1000 ``xyxy`` 转成内部像素 ``xywh``。"""

    image_width, image_height = validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = validate_norm1000_xyxy(bbox)
    pixel_xyxy = (
        x1 * image_width / 1000.0,
        y1 * image_height / 1000.0,
        x2 * image_width / 1000.0,
        y2 * image_height / 1000.0,
    )
    return validate_xywh(xyxy_to_xywh(pixel_xyxy), (image_width, image_height))


def model_pixel_xyxy_to_pixel_xywh(
    bbox: Sequence[float],
    model_width: int,
    model_height: int,
    image_width: int,
    image_height: int,
) -> BBoxXYWH:
    """把模型输入图像素空间的 ``xyxy`` 线性映回原始帧像素 ``xywh``。

    ``model_width/model_height`` 必须是 processor 实际喂给模型的尺寸（由
    ``image_grid_thw * patch_size`` 得到），不能用原图尺寸代替，否则会引入
    一个与目标位置相关的系统性偏移。

    与 norm1000 路径不同，这里允许框轻微越界后裁剪：grounding 模型给出略微
    超出图像边界的坐标是常见现象，norm1000 路径靠 ``[0, 1000]`` 值域隐式做了
    同样的事，这里显式裁剪以保持两条协议行为一致。裁剪后无有效面积仍然报错。
    """

    model_width, model_height = validate_image_size(model_width, model_height)
    image_width, image_height = validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = validate_xyxy(bbox)
    scale_x = image_width / float(model_width)
    scale_y = image_height / float(model_height)
    pixel_xyxy = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
    return clip_xywh(xyxy_to_xywh(pixel_xyxy), image_width, image_height)


def pixel_xywh_to_model_pixel_xyxy(
    bbox: Sequence[float],
    model_width: int,
    model_height: int,
    image_width: int,
    image_height: int,
    decimals: Optional[int] = 2,
) -> BBoxXYXY:
    """把原始帧像素 ``xywh`` 转成模型输入图像素空间的 ``xyxy``。"""

    model_width, model_height = validate_image_size(model_width, model_height)
    image_width, image_height = validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = xywh_to_xyxy(validate_xywh(bbox, (image_width, image_height)))
    scale_x = model_width / float(image_width)
    scale_y = model_height / float(image_height)
    output = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
    if decimals is not None:
        output = tuple(round(value, decimals) for value in output)
    return validate_xyxy(output)


def pixel_xywh_to_norm1000_xyxy(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    decimals: Optional[int] = 2,
) -> BBoxXYXY:
    """把内部像素 ``xywh`` 转成模型使用的 norm1000 ``xyxy``。"""

    image_width, image_height = validate_image_size(image_width, image_height)
    x1, y1, x2, y2 = xywh_to_xyxy(validate_xywh(bbox, (image_width, image_height)))
    output = (
        x1 * 1000.0 / image_width,
        y1 * 1000.0 / image_height,
        x2 * 1000.0 / image_width,
        y2 * 1000.0 / image_height,
    )
    if decimals is not None:
        output = tuple(round(value, decimals) for value in output)
    return validate_norm1000_xyxy(output)


def bbox_iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    """计算两个像素 ``xywh`` 框的 IoU。"""

    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union
