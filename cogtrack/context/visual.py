"""训练与在线推理共享的视觉指代渲染。

新范式把目标框直接画在所有过去参考图上，但当前搜索图始终保持无标注。绘框和
mosaic 如果在线、离线各写一份，很容易在线宽、颜色或 resize 顺序上发生漂移；因此
两条链路必须只调用本模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import cv2
import numpy as np

from cogtrack.protocol import clip_xywh

REFERENCE_MODE_BBOX_TEXT = "bbox_text"
REFERENCE_MODE_VISUAL_BOX = "visual_box"
REFERENCE_MODES = frozenset({REFERENCE_MODE_BBOX_TEXT, REFERENCE_MODE_VISUAL_BOX})

# 绘框版本会写入训练 metadata 和运行 manifest。改变颜色或线宽规则时必须提升；
# mosaic 布局由下方独立版本号记录，不能把两类变化混为同一字段。
VISUAL_MARKER_VERSION = "red_box_v1"

# 历史布局是模型输入协议的一部分，必须独立版本化。visual-v5/旧实验继续使用紧凑
# 网格；VLT-v6.3 固定使用左到右的三帧近期历史条带。v1 的 panel 直接相接，v2 在
# panel 间加入白色分隔带，避免连续背景被误读成一张超宽图。
HISTORY_LAYOUT_COMPACT_GRID_V1 = "compact_grid_v1"
HISTORY_LAYOUT_RECENT_STRIP_3_V1 = "recent_strip_3_v1"
HISTORY_LAYOUT_RECENT_STRIP_3_V2 = "recent_strip_3_v2"
HISTORY_LAYOUTS = frozenset(
    {
        HISTORY_LAYOUT_COMPACT_GRID_V1,
        HISTORY_LAYOUT_RECENT_STRIP_3_V1,
        HISTORY_LAYOUT_RECENT_STRIP_3_V2,
    }
)

# 分隔宽度随 panel 高度缩放，使离线数据生成和在线推理在不同分辨率下保持同一比例。
# 白色只用于分隔，不承载时间标签；时间顺序仍由 panel 的左到右排列和 Prompt 定义。
HISTORY_STRIP_SEPARATOR_RATIO = 0.03
HISTORY_STRIP_SEPARATOR_COLOR_RGB = (255, 255, 255)

_HistoryItem = TypeVar("_HistoryItem")


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


def arrange_history_items(
    items: Sequence[_HistoryItem],
    *,
    layout: str,
) -> tuple[_HistoryItem, ...]:
    """按版本化布局选择并补齐历史项。

    ``recent_strip_3_v1`` 和 ``recent_strip_3_v2`` 都只保留最近三项，输入必须已经按
    时间从旧到新排列。少于三项时在右侧复制最近可用项，例如
    ``[h1, h2] -> [h1, h2, h2]``。重复项只是视觉 padding，不表示额外发生了一次观测。
    """

    normalized_layout = str(layout).strip().lower()
    if normalized_layout not in HISTORY_LAYOUTS:
        raise ValueError(f"layout 必须是 {sorted(HISTORY_LAYOUTS)} 之一")
    arranged = tuple(items)
    if not arranged:
        raise ValueError("排列历史至少需要一个条目")
    if normalized_layout == HISTORY_LAYOUT_COMPACT_GRID_V1:
        return arranged

    recent = list(arranged[-3:])
    while len(recent) < 3:
        recent.append(recent[-1])
    return tuple(recent)


def build_history_mosaic(
    panels: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    panel_height: int = 240,
    style: VisualMarkerStyle = DEFAULT_VISUAL_MARKER_STYLE,
    layout: str = HISTORY_LAYOUT_COMPACT_GRID_V1,
) -> np.ndarray:
    """按输入顺序构造带框历史 mosaic。

    调用方负责保证 panels 严格来自 current 之前。这里不接收帧号，也不把任何绝对时间
    写进图像；时间只由 panel 排列顺序表达。
    """

    if isinstance(panel_height, bool) or not isinstance(panel_height, int) or panel_height <= 0:
        raise ValueError("panel_height 必须是正整数")
    normalized_layout = str(layout).strip().lower()
    arranged_panels = arrange_history_items(panels, layout=normalized_layout)

    rendered: list[np.ndarray] = []
    for image, bbox_xywh in arranged_panels:
        panel = draw_reference_box(image, bbox_xywh, style=style)
        height, width = panel.shape[:2]
        resized_width = max(1, int(round(width * panel_height / float(height))))
        interpolation = cv2.INTER_AREA if panel_height < height else cv2.INTER_LINEAR
        rendered.append(cv2.resize(panel, (resized_width, panel_height), interpolation=interpolation))

    strip_layouts = {
        HISTORY_LAYOUT_RECENT_STRIP_3_V1,
        HISTORY_LAYOUT_RECENT_STRIP_3_V2,
    }
    if normalized_layout in strip_layouts:
        columns, rows = 3, 1
    else:
        columns = 1 if len(rendered) <= 2 else 2
        rows = (len(rendered) + columns - 1) // columns
    cell_width = max(panel.shape[1] for panel in rendered)
    separator_width = (
        max(1, int(round(panel_height * HISTORY_STRIP_SEPARATOR_RATIO)))
        if normalized_layout == HISTORY_LAYOUT_RECENT_STRIP_3_V2
        else 0
    )
    canvas_width = columns * cell_width + max(0, columns - 1) * separator_width
    canvas = np.full((rows * panel_height, canvas_width, 3), 220, dtype=np.uint8)
    if separator_width:
        for separator_index in range(1, columns):
            start = separator_index * cell_width + (separator_index - 1) * separator_width
            canvas[:, start : start + separator_width] = HISTORY_STRIP_SEPARATOR_COLOR_RGB
    for index, panel in enumerate(rendered):
        row, column = divmod(index, columns)
        x = column * (cell_width + separator_width) + (cell_width - panel.shape[1]) // 2
        y = row * panel_height
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


__all__ = [
    "DEFAULT_VISUAL_MARKER_STYLE",
    "HISTORY_LAYOUT_COMPACT_GRID_V1",
    "HISTORY_LAYOUT_RECENT_STRIP_3_V1",
    "HISTORY_LAYOUT_RECENT_STRIP_3_V2",
    "HISTORY_LAYOUTS",
    "HISTORY_STRIP_SEPARATOR_COLOR_RGB",
    "HISTORY_STRIP_SEPARATOR_RATIO",
    "REFERENCE_MODE_BBOX_TEXT",
    "REFERENCE_MODE_VISUAL_BOX",
    "REFERENCE_MODES",
    "VISUAL_MARKER_VERSION",
    "VisualMarkerStyle",
    "arrange_history_items",
    "build_history_mosaic",
    "draw_reference_box",
    "validate_reference_mode",
]
