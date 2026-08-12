"""构造 pair/mosaic 视觉上下文。

本模块只负责图像组织，不调用模型、不更新记忆，也不读取数据集真值。初始化
身份锚点由 tracker 显式传入，后续历史只能来自已经通过门控的 ``MemoryRecord``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from cogtrack.memory import IdentityAnchor, MemoryRecord
from cogtrack.prompts import PromptSpec, build_mosaic_prompt, build_pair_prompt
from cogtrack.protocol import (
    BBOX_PROTOCOL_NORM1000,
    clip_xywh,
    pixel_xywh_to_norm1000_xyxy,
    validate_bbox_protocol,
)


@dataclass(frozen=True)
class ContextBuildResult:
    """一次 VLM 输入的图像、Prompt 和可复现帧号。"""

    images: tuple[np.ndarray, ...]
    prompt: PromptSpec
    reference_frames: tuple[int, ...]
    effective_mode: str


class TrackingContextBuilder:
    """从永久锚点和可信正记忆构造多图输入。

    ``mosaic`` 在尚无可信历史时自动退化为 ``pair``。这不是静默改变实验：
    ``effective_mode`` 和 Prompt 版本会写入逐帧结果，后续可以精确统计预热阶段。
    """

    def __init__(
        self,
        anchor: IdentityAnchor,
        *,
        mosaic_panel_height: int = 240,
        bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    ) -> None:
        if not isinstance(anchor, IdentityAnchor):
            raise TypeError("anchor 必须是 IdentityAnchor")
        if isinstance(mosaic_panel_height, bool) or mosaic_panel_height <= 0:
            raise ValueError("mosaic_panel_height 必须是正整数")
        if not isinstance(anchor.image, np.ndarray):
            raise TypeError("在线上下文构造要求 IdentityAnchor.image 为 numpy.ndarray")
        self.anchor = anchor
        self.mosaic_panel_height = int(mosaic_panel_height)
        self.bbox_protocol = validate_bbox_protocol(bbox_protocol)
        anchor_height, anchor_width = anchor.image.shape[:2]
        self._anchor_bbox_norm1000 = pixel_xywh_to_norm1000_xyxy(
            anchor.bbox_xywh,
            anchor_width,
            anchor_height,
        )
        # 永久锚点保留完整原始场景，不画框、不裁剪。初始化 bbox 仅通过 Prompt
        # 坐标传给模型，训练与在线推理使用同一表示。
        self._anchor_image = np.ascontiguousarray(anchor.image.copy())

    @staticmethod
    def _draw_bbox(
        image: np.ndarray,
        bbox_xywh: Sequence[float],
        *,
        color: tuple[int, int, int],
        thickness: int = 3,
    ) -> np.ndarray:
        """在 RGB 图像副本上绘框，不修改原始帧或记忆图像。"""

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("图像必须是 HxWx3 numpy.ndarray")
        height, width = image.shape[:2]
        x, y, box_width, box_height = clip_xywh(bbox_xywh, width, height)
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + box_width)), int(round(y + box_height))
        output = np.ascontiguousarray(image.copy())
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        return output

    def _mosaic(self, records: Sequence[MemoryRecord]) -> np.ndarray:
        panels: list[np.ndarray] = []
        for record in records:
            if not isinstance(record.image, np.ndarray) or record.bbox_xywh is None:
                # 只允许带图像和可靠定位框的正记忆进入视觉历史。
                continue
            panel = self._draw_bbox(record.image, record.bbox_xywh, color=(255, 0, 0), thickness=2)
            height, width = panel.shape[:2]
            new_width = max(1, int(round(width * self.mosaic_panel_height / float(height))))
            panel = cv2.resize(panel, (new_width, self.mosaic_panel_height), interpolation=cv2.INTER_AREA)
            # 不把绝对帧号写进视觉输入；frame_id 仍由运行时 metadata 审计，避免模型
            # 学习数据集位置偏置。
            panels.append(panel)

        if not panels:
            raise ValueError("构造 mosaic 至少需要一条带图像和 bbox 的可信正记忆")
        columns = 1 if len(panels) <= 2 else 2
        rows = (len(panels) + columns - 1) // columns
        cell_width = max(panel.shape[1] for panel in panels)
        canvas = np.full(
            (rows * self.mosaic_panel_height, columns * cell_width, 3),
            220,
            dtype=np.uint8,
        )
        for index, panel in enumerate(panels):
            row, column = divmod(index, columns)
            x = column * cell_width + (cell_width - panel.shape[1]) // 2
            y = row * self.mosaic_panel_height
            canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        return canvas

    def build_pair(
        self,
        current_image: np.ndarray,
        target_text: str = "",
        semantic_memory: str = "",
        include_memory_update: bool = True,
    ) -> ContextBuildResult:
        prompt = build_pair_prompt(
            target_text=target_text,
            semantic_memory=semantic_memory,
            reference_has_box=False,
            reference_bbox_norm1000_xyxy=self._anchor_bbox_norm1000,
            bbox_protocol=self.bbox_protocol,
            include_memory_update=include_memory_update,
        )
        return ContextBuildResult(
            images=(self._anchor_image.copy(), current_image),
            prompt=prompt,
            reference_frames=(self.anchor.frame_id,),
            effective_mode="pair",
        )

    def build_mosaic(
        self,
        current_image: np.ndarray,
        records: Sequence[MemoryRecord],
        target_text: str = "",
        semantic_memory: str = "",
        include_memory_update: bool = True,
    ) -> ContextBuildResult:
        usable = tuple(record for record in records if record.image is not None and record.bbox_xywh is not None)
        if not usable:
            return self.build_pair(
                current_image,
                target_text,
                semantic_memory,
                include_memory_update,
            )
        mosaic = self._mosaic(usable)
        prompt = build_mosaic_prompt(
            history_count=len(usable),
            target_text=target_text,
            semantic_memory=semantic_memory,
            reference_bbox_norm1000_xyxy=self._anchor_bbox_norm1000,
            bbox_protocol=self.bbox_protocol,
            include_memory_update=include_memory_update,
        )
        return ContextBuildResult(
            images=(self._anchor_image.copy(), mosaic, current_image),
            prompt=prompt,
            reference_frames=(self.anchor.frame_id, *(record.frame_id for record in usable)),
            effective_mode="mosaic",
        )
