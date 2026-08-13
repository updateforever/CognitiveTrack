"""v5 视觉画框统一跟踪 Prompt。

过去参考图通过像素中的红框指认目标，不再向模型提供 reference bbox 坐标文本。最后
一张 current search image 永远无框，也是唯一允许输出 bbox 的图像。
"""

from __future__ import annotations

from ..protocol.bbox import BBOX_PROTOCOL_NORM1000, validate_bbox_protocol
from .common import (
    PromptSpec,
    semantic_memory_section,
    target_text_section,
    tracking_output_schema,
)

VISUAL_TRACKING_PROMPT_VERSION = "5.0.0"
VISUAL_PAIR_PROMPT_NAME = "cognitive_visual_pair"
VISUAL_MOSAIC_PROMPT_NAME = "cognitive_visual_mosaic"

VISUAL_TRACKING_SYSTEM_PROMPT = """You are a rigorous long-term single-object tracking model.
Track the exact instance marked by red boxes in the past reference images. Similar-looking
objects are distractors. Return exactly one JSON object without Markdown or additional text."""


def build_visual_tracking_prompt(
    *,
    history_count: int = 0,
    target_text: str = "",
    semantic_memory: str = "",
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    include_memory_update: bool = True,
) -> PromptSpec:
    """构造两图或三图的统一视觉指代 Prompt。

    ``history_count=0`` 时只有 boxed anchor 与 current 两张图；大于零时 Image 2 是
    chronological history mosaic，最后一张图顺延为 Image 3。两种形式共享相同决策顺序
    和输出协议，避免预热阶段切换成另一个任务。
    """

    if isinstance(history_count, bool) or not isinstance(history_count, int) or history_count < 0:
        raise ValueError("history_count 必须是非负整数")
    validate_bbox_protocol(bbox_protocol)

    if history_count:
        image_description = f"""- Image 1 is the permanent identity anchor. Its red box marks the target.
- Image 2 is a chronological mosaic of {history_count} accepted past observations. Every red box marks
  the tracked target in that past panel; these boxes may be imperfect. The last panel is the most recent.
- Image 3 is the final current full search image. It is unmarked."""
        decision_order = """1. Anchor identity to Image 1.
2. Use Image 2 only as supporting evidence for appearance evolution; Image 1 remains authoritative.
3. Search Image 3 globally and reject same-category distractors.
4. Output present with a box only if the exact target is visible and localizable; otherwise output absent.
5. Propose memory_update only for a new, stable and materially useful visual identity cue."""
        prompt_name = VISUAL_MOSAIC_PROMPT_NAME
        expected_image_count = 3
    else:
        image_description = """- Image 1 is the permanent identity anchor. Its red box marks the target.
- Image 2 is the final current full search image. It is unmarked.
- No dynamic history mosaic is available for this sample."""
        decision_order = """1. Anchor identity to Image 1.
2. Search Image 2 globally and reject same-category distractors.
3. Output present with a box only if the exact target is visible and localizable; otherwise output absent.
4. Propose memory_update only for a new, stable and materially useful visual identity cue."""
        prompt_name = VISUAL_PAIR_PROMPT_NAME
        expected_image_count = 2

    user_prompt = f"""Task: find the referenced target in the final unmarked search image.

All red-boxed images are past observations. Their boxes identify the past target instance; they are
not location hints for the current image. The final image is always the current search image.
{image_description}

Decision order:
{decision_order}

{target_text_section(target_text)}
{semantic_memory_section(semantic_memory)}

{tracking_output_schema(bbox_protocol, include_memory_update=include_memory_update)}"""
    return PromptSpec(
        name=prompt_name,
        version=VISUAL_TRACKING_PROMPT_VERSION,
        system_prompt=VISUAL_TRACKING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=expected_image_count,
        include_memory_update=include_memory_update,
        bbox_protocol=bbox_protocol,
    )


__all__ = [
    "VISUAL_MOSAIC_PROMPT_NAME",
    "VISUAL_PAIR_PROMPT_NAME",
    "VISUAL_TRACKING_PROMPT_VERSION",
    "VISUAL_TRACKING_SYSTEM_PROMPT",
    "build_visual_tracking_prompt",
]
