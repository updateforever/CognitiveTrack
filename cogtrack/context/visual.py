"""训练与在线推理共享的视觉指代渲染。

新范式把目标框直接画在所有过去参考图上，但当前搜索图始终保持无标注。绘框和
mosaic 如果在线、离线各写一份，很容易在线宽、颜色或 resize 顺序上发生漂移；因此
两条链路必须只调用本模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from cogtrack.protocol import clip_xywh

REFERENCE_MODE_BBOX_TEXT = "bbox_text"
REFERENCE_MODE_VISUAL_BOX = "visual_box"
REFERENCE_MODES = frozenset({REFERENCE_MODE_BBOX_TEXT, REFERENCE_MODE_VISUAL_BOX})

# 版本号会写入训练 metadata 和运行 manifest。改变颜色、线宽规则或 mosaic 布局时必须
# 提升版本，不能在同一个数据版本中静默改变视觉任务定义。
VISUAL_MARKER_VERSION = "red_box_v1"


@dataclass(frozen=True)
class VisualMarkerStyle:
    """视觉指代框样式。

    首版固定为 RGB 红框。线宽随短边自适应，避免同一固定像素宽度在低分辨率图上遮挡
    目标、在高清视频上又几乎不可见。首个 baseline 暂不做颜色增强，先保证复现性；
    样式鲁棒性后续作为独立消融。
    """

    color_rgb: tuple[int, int, int] = (255, 0, 0)
    min_thickness: int = 2
    short_side_divisor: float = 180.0

    def __post_init__(self) -> None:
        if (
            len(self.color_rgb) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
                for value in self.color_rgb
            )
        ):
            raise ValueError("color_rgb 必须是三个 [0,255] 整数")
        if isinstance(self.min_thickness, bool) or self.min_thickness <= 0:
            raise ValueError("min_thickness 必须是正整数")
        if isinstance(self.short_side_divisor, bool) or self.short_side_divisor <= 0:
            raise ValueError("short_side_divisor 必须为正数")

    def thickness_for(self, image: np.ndarray) -> int:
        height, width = _validate_rgb_image(image).shape[:2]
        return max(self.min_thickness, int(round(min(width, height) / self.short_side_divisor)))


DEFAULT_VISUAL_MARKER_STYLE = VisualMarkerStyle()


def validate_reference_mode(value: str) -> str:
    """规范化并校验过去参考图的指代方式。"""

    normalized = str(value).strip().lower()
    if normalized not in REFERENCE_MODES:
        raise ValueError(f"reference_mode 必须是 {sorted(REFERENCE_MODES)} 之一，实际为 {value!r}")
    return normalized


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("图像必须是 HxWx3 numpy.ndarray")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("图像宽高必须为正数")
    return image


def draw_reference_box(
    image: np.ndarray,
    bbox_xywh: Sequence[float],
    *,
    style: VisualMarkerStyle = DEFAULT_VISUAL_MARKER_STYLE,
) -> np.ndarray:
    """在 RGB 图像副本上绘制目标框，不修改调用方保存的原始帧。"""

    source = _validate_rgb_image(image)
    if not isinstance(style, VisualMarkerStyle):
        raise TypeError("style 必须是 VisualMarkerStyle")
    height, width = source.shape[:2]
    x, y, box_width, box_height = clip_xywh(bbox_xywh, width, height)
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(x1, min(width - 1, int(round(x + box_width))))
    y2 = max(y1, min(height - 1, int(round(y + box_height))))
    output = np.ascontiguousarray(source.copy())
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        style.color_rgb,
        style.thickness_for(source),
        lineType=cv2.LINE_8,
    )
    return output


def build_history_mosaic(
    panels: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    panel_height: int = 240,
    style: VisualMarkerStyle = DEFAULT_VISUAL_MARKER_STYLE,
) -> np.ndarray:
    """按输入顺序构造带框历史 mosaic。

    调用方负责保证 panels 严格来自 current 之前。这里不接收帧号，也不把任何绝对时间
    写进图像；时间只由 panel 排列顺序表达。
    """

    if isinstance(panel_height, bool) or not isinstance(panel_height, int) or panel_height <= 0:
        raise ValueError("panel_height 必须是正整数")
    if not panels:
        raise ValueError("构造 mosaic 至少需要一个带框历史 panel")

    rendered: list[np.ndarray] = []
    for image, bbox_xywh in panels:
        panel = draw_reference_box(image, bbox_xywh, style=style)
        height, width = panel.shape[:2]
        resized_width = max(1, int(round(width * panel_height / float(height))))
        interpolation = cv2.INTER_AREA if panel_height < height else cv2.INTER_LINEAR
        rendered.append(cv2.resize(panel, (resized_width, panel_height), interpolation=interpolation))

    columns = 1 if len(rendered) <= 2 else 2
    rows = (len(rendered) + columns - 1) // columns
    cell_width = max(panel.shape[1] for panel in rendered)
    canvas = np.full((rows * panel_height, columns * cell_width, 3), 220, dtype=np.uint8)
    for index, panel in enumerate(rendered):
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - panel.shape[1]) // 2
        y = row * panel_height
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


__all__ = [
    "DEFAULT_VISUAL_MARKER_STYLE",
    "REFERENCE_MODE_BBOX_TEXT",
    "REFERENCE_MODE_VISUAL_BOX",
    "REFERENCE_MODES",
    "VISUAL_MARKER_VERSION",
    "VisualMarkerStyle",
    "build_history_mosaic",
    "draw_reference_box",
    "validate_reference_mode",
]
