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
    build_vlt_tracking_prompt,
)
from cogtrack.protocol import (
    BBOX_PROTOCOL_NORM1000,
    pixel_xywh_to_norm1000_xyxy,
    validate_bbox_protocol,
)

from .visual import (
    HISTORY_LAYOUT_COMPACT_GRID_V1,
    HISTORY_LAYOUT_RECENT_STRIP_3_V2,
    REFERENCE_MODE_BBOX_TEXT,
    REFERENCE_MODE_VISUAL_BOX,
    VISUAL_MARKER_VERSION,
    arrange_history_items,
    build_history_mosaic,
    draw_reference_box,
    validate_reference_mode,
)

PROMPT_PROFILE_VISUAL_V5 = "visual_v5"
PROMPT_PROFILE_VLT_V6 = "vlt_v6"
PROMPT_PROFILES = frozenset({PROMPT_PROFILE_VISUAL_V5, PROMPT_PROFILE_VLT_V6})

# 可以安全用作初始化文本的 ``language_scope``。判据是该文本必须只描述初始化时刻
# 可见的目标，不能包含后续或整段视频的剧情。
#
# - ``initial_target``：LaSOT/TNL2K 的初始目标描述；
# - ``first_action_description``：MGIT 官方 action 层的第一段描述。其
#   ``start_frame`` 经全量核对为 0（91/91 序列），因此不泄漏未来信息。
#
# 训练导出与在线 tracker 必须共用本集合；两处若各自维护会导致同一序列在训练和
# 推理时得到不同的初始文本。
SAFE_INIT_LANGUAGE_SCOPES = frozenset({"initial_target", "first_action_description"})


def is_unsafe_init_language_scope(scope: str, *, dataset: str = "") -> bool:
    """判断某个 ``language_scope`` 能否作为在线可得的初始化文本。

    未声明 scope 的数据集保持既有保守行为：只有 MGIT 会因为可能存在整段
    ``story`` 描述而被拒绝，其余数据集沿用其 loader 提供的初始描述。
    """

    normalized = str(scope).strip().lower()
    if normalized in SAFE_INIT_LANGUAGE_SCOPES:
        return False
    if normalized == "full_video_story":
        return True
    # MGIT 的描述文件同时含 action/activity/story 三层，scope 缺失时无法确认边界。
    return str(dataset).strip().lower() == "mgit"


def validate_prompt_profile(value: str) -> str:
    """规范化视觉指代 Prompt 版本，防止旧实验被静默升级。"""

    normalized = str(value).strip().lower()
    if normalized not in PROMPT_PROFILES:
        raise ValueError(f"prompt_profile 必须是 {sorted(PROMPT_PROFILES)} 之一")
    return normalized


def history_layout_for_prompt_profile(value: str) -> str:
    """返回指定 Prompt profile 的冻结历史布局版本。"""

    profile = validate_prompt_profile(value)
    return (
        HISTORY_LAYOUT_RECENT_STRIP_3_V2
        if profile == PROMPT_PROFILE_VLT_V6
        else HISTORY_LAYOUT_COMPACT_GRID_V1
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
    history_layout_version: str | None


class TrackingContextBuilder:
    """从永久锚点和可信正记忆构造多图输入。

    默认情况下，``mosaic`` 在尚无可信历史时退化为 ``pair``。VLT-v6.4 可显式启用
    ``force_history_image``，以初始化观测复制补齐三个历史 panel，保持固定三图接口。
    """

    def __init__(
        self,
        anchor: IdentityAnchor,
        *,
        mosaic_panel_height: int = 240,
        bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
        reference_mode: str = REFERENCE_MODE_BBOX_TEXT,
        prompt_profile: str = PROMPT_PROFILE_VISUAL_V5,
        force_history_image: bool = False,
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
        self.prompt_profile = validate_prompt_profile(prompt_profile)
        self.history_layout_version = history_layout_for_prompt_profile(self.prompt_profile)
        if self.prompt_profile == PROMPT_PROFILE_VLT_V6 and self.reference_mode != REFERENCE_MODE_VISUAL_BOX:
            raise ValueError("vlt_v6 只支持把目标框直接画入过去图像的 visual_box 模式")
        if not isinstance(force_history_image, bool):
            raise TypeError("force_history_image 必须是 bool")
        if force_history_image and self.reference_mode != REFERENCE_MODE_VISUAL_BOX:
            raise ValueError("force_history_image 只支持 visual_box 模式")
        self.force_history_image = force_history_image
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

    def _visual_prompt(
        self,
        *,
        history_count: int,
        target_text: str,
        semantic_memory: str,
        include_memory_update: bool,
    ) -> PromptSpec:
        builder = (
            build_vlt_tracking_prompt
            if self.prompt_profile == PROMPT_PROFILE_VLT_V6
            else build_visual_tracking_prompt
        )
        return builder(
            history_count=history_count,
            target_text=target_text,
            semantic_memory=semantic_memory,
            bbox_protocol=self.bbox_protocol,
            include_memory_update=include_memory_update,
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
        return build_history_mosaic(
            panels,
            panel_height=self.mosaic_panel_height,
            layout=self.history_layout_version,
        )

    def build_pair(
        self,
        current_image: np.ndarray,
        target_text: str = "",
        semantic_memory: str = "",
        include_memory_update: bool = True,
    ) -> ContextBuildResult:
        if self.reference_mode == REFERENCE_MODE_VISUAL_BOX:
            prompt = self._visual_prompt(
                history_count=0,
                target_text=target_text,
                semantic_memory=semantic_memory,
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
            history_layout_version=None,
        )

    def build_mosaic(
        self,
        current_image: np.ndarray,
        records: Sequence[MemoryRecord],
        target_text: str = "",
        semantic_memory: str = "",
        include_memory_update: bool = True,
    ) -> ContextBuildResult:
        # records 已由 MemoryBank.select_positive 选过一次，这里再次过滤图像/bbox
        # 完整性并按 frame_id 排序，防止坏记录或调用方乱序破坏时间方向。
        usable = tuple(
            sorted(
                (
                    record
                    for record in records
                    if record.image is not None and record.bbox_xywh is not None
                ),
                key=lambda record: record.frame_id,
            )
        )
        if not usable:
            if self.force_history_image:
                # VLT-v6.4 的正式接口固定三图。尚无动态历史时，Image 2 以初始化
                # 观测复制补齐三个 panel；它不是伪造预测，也不会读取 current GT。
                padded_anchor = arrange_history_items(
                    ((self.anchor.image, self.anchor.bbox_xywh),),
                    layout=self.history_layout_version,
                )
                mosaic = build_history_mosaic(
                    padded_anchor,
                    panel_height=self.mosaic_panel_height,
                    layout=self.history_layout_version,
                )
                prompt = self._visual_prompt(
                    history_count=len(padded_anchor),
                    target_text=target_text,
                    semantic_memory=semantic_memory,
                    include_memory_update=include_memory_update,
                )
                return ContextBuildResult(
                    images=(
                        self._anchor_image.copy(),
                        mosaic,
                        np.ascontiguousarray(current_image.copy()),
                    ),
                    prompt=prompt,
                    reference_frames=(
                        self.anchor.frame_id,
                        *(self.anchor.frame_id for _ in padded_anchor),
                    ),
                    effective_mode="mosaic",
                    reference_mode=self.reference_mode,
                    visual_marker_version=VISUAL_MARKER_VERSION,
                    history_layout_version=self.history_layout_version,
                )
            return self.build_pair(
                current_image,
                target_text,
                semantic_memory,
                include_memory_update,
            )
        arranged_usable = arrange_history_items(
            usable,
            layout=self.history_layout_version,
        )
        mosaic = self._mosaic(arranged_usable)
        if self.reference_mode == REFERENCE_MODE_VISUAL_BOX:
            prompt = self._visual_prompt(
                history_count=len(arranged_usable),
                target_text=target_text,
                semantic_memory=semantic_memory,
                include_memory_update=include_memory_update,
            )
        else:
            prompt = build_mosaic_prompt(
                history_count=len(arranged_usable),
                target_text=target_text,
                semantic_memory=semantic_memory,
                reference_bbox_norm1000_xyxy=self._anchor_bbox_norm1000,
                bbox_protocol=self.bbox_protocol,
                include_memory_update=include_memory_update,
            )
        # Prompt-6.4 图像槽位在这里冻结：Image 1=永久红框锚点，Image 2=按时间
        # 从左到右的三条可信观测，Image 3=当前无框搜索帧。reference_frames 仅记录
        # Image 1/2 的来源帧用于审计，整个元组都不包含当前帧或未来帧。
        return ContextBuildResult(
            images=(self._anchor_image.copy(), mosaic, np.ascontiguousarray(current_image.copy())),
            prompt=prompt,
            reference_frames=(
                self.anchor.frame_id,
                *(record.frame_id for record in arranged_usable),
            ),
            effective_mode="mosaic",
            reference_mode=self.reference_mode,
            visual_marker_version=(
                VISUAL_MARKER_VERSION if self.reference_mode == REFERENCE_MODE_VISUAL_BOX else None
            ),
            history_layout_version=self.history_layout_version,
        )
