"""构造 pair/mosaic 视觉上下文。

本模块只负责图像组织，不调用模型、不更新记忆，也不读取数据集真值。初始化
身份锚点由 tracker 显式传入，后续历史只能来自已经通过门控的 ``MemoryRecord``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cogtrack.memory import IdentityAnchor, MemoryRecord
from cogtrack.prompts import (
    PromptSpec,
    build_mosaic_prompt,
    build_pair_prompt,
    build_visual_tracking_prompt,
)
from cogtrack.protocol import (
    BBOX_PROTOCOL_NORM1000,
    pixel_xywh_to_norm1000_xyxy,
    validate_bbox_protocol,
)

from .visual import (
    REFERENCE_MODE_BBOX_TEXT,
    REFERENCE_MODE_VISUAL_BOX,
    VISUAL_MARKER_VERSION,
    build_history_mosaic,
    draw_reference_box,
    validate_reference_mode,
)


@dataclass(frozen=True)
class ContextBuildResult:
    """一次 VLM 输入的图像、Prompt 和可复现帧号。"""

    images: tuple[np.ndarray, ...]
    prompt: PromptSpec
    reference_frames: tuple[int, ...]
    effective_mode: str
    reference_mode: str
    visual_marker_version: str | None


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
        reference_mode: str = REFERENCE_MODE_BBOX_TEXT,
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
        self.reference_mode = validate_reference_mode(reference_mode)
        anchor_height, anchor_width = anchor.image.shape[:2]
        self._anchor_bbox_norm1000 = pixel_xywh_to_norm1000_xyxy(
            anchor.bbox_xywh,
            anchor_width,
            anchor_height,
        )
        # v5 将框画进完整首帧；旧实验仍可显式选择 bbox_text 复现坐标文本输入。
        self._anchor_image = (
            draw_reference_box(anchor.image, anchor.bbox_xywh)
            if self.reference_mode == REFERENCE_MODE_VISUAL_BOX
            else np.ascontiguousarray(anchor.image.copy())
        )

    def _mosaic(self, records: Sequence[MemoryRecord]) -> np.ndarray:
        panels: list[tuple[np.ndarray, Sequence[float]]] = []
        for record in records:
            if not isinstance(record.image, np.ndarray) or record.bbox_xywh is None:
                # 只允许带图像和可靠定位框的正记忆进入视觉历史。
                continue
            panels.append((record.image, record.bbox_xywh))

        if not panels:
            raise ValueError("构造 mosaic 至少需要一条带图像和 bbox 的可信正记忆")
        return build_history_mosaic(panels, panel_height=self.mosaic_panel_height)

    def build_pair(
        self,
        current_image: np.ndarray,
        target_text: str = "",
        semantic_memory: str = "",
        include_memory_update: bool = True,
    ) -> ContextBuildResult:
        if self.reference_mode == REFERENCE_MODE_VISUAL_BOX:
            prompt = build_visual_tracking_prompt(
                history_count=0,
                target_text=target_text,
                semantic_memory=semantic_memory,
                bbox_protocol=self.bbox_protocol,
                include_memory_update=include_memory_update,
            )
        else:
            prompt = build_pair_prompt(
                target_text=target_text,
                semantic_memory=semantic_memory,
                reference_has_box=False,
                reference_bbox_norm1000_xyxy=self._anchor_bbox_norm1000,
                bbox_protocol=self.bbox_protocol,
                include_memory_update=include_memory_update,
            )
        return ContextBuildResult(
            images=(self._anchor_image.copy(), np.ascontiguousarray(current_image.copy())),
            prompt=prompt,
            reference_frames=(self.anchor.frame_id,),
            effective_mode="pair",
            reference_mode=self.reference_mode,
            visual_marker_version=(
                VISUAL_MARKER_VERSION if self.reference_mode == REFERENCE_MODE_VISUAL_BOX else None
            ),
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
        if self.reference_mode == REFERENCE_MODE_VISUAL_BOX:
            prompt = build_visual_tracking_prompt(
                history_count=len(usable),
                target_text=target_text,
                semantic_memory=semantic_memory,
                bbox_protocol=self.bbox_protocol,
                include_memory_update=include_memory_update,
            )
        else:
            prompt = build_mosaic_prompt(
                history_count=len(usable),
                target_text=target_text,
                semantic_memory=semantic_memory,
                reference_bbox_norm1000_xyxy=self._anchor_bbox_norm1000,
                bbox_protocol=self.bbox_protocol,
                include_memory_update=include_memory_update,
            )
        return ContextBuildResult(
            images=(self._anchor_image.copy(), mosaic, np.ascontiguousarray(current_image.copy())),
            prompt=prompt,
            reference_frames=(self.anchor.frame_id, *(record.frame_id for record in usable)),
            effective_mode="mosaic",
            reference_mode=self.reference_mode,
            visual_marker_version=(
                VISUAL_MARKER_VERSION if self.reference_mode == REFERENCE_MODE_VISUAL_BOX else None
            ),
        )
