"""初始锚点 + 可信历史拼图 + 当前帧的时序 Prompt。"""

from typing import Sequence

from ..protocol.bbox import (
    BBOX_PROTOCOL_NORM1000,
    validate_bbox_protocol,
    validate_norm1000_xyxy,
    validate_xyxy,
)
from .common import (
    TRACKING_SYSTEM_PROMPT,
    PromptSpec,
    semantic_memory_section,
    target_text_section,
    tracking_output_schema,
)

MOSAIC_PROMPT_NAME = "cognitive_mosaic"
MOSAIC_PROMPT_VERSION = "4.3.0"


def build_mosaic_prompt(
    history_count: int,
    target_text: str = "",
    semantic_memory: str = "",
    reference_bbox_norm1000_xyxy: Sequence[float] | None = None,
    reference_bbox: Sequence[float] | str | None = None,
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    include_memory_update: bool = True,
) -> PromptSpec:
    """构造三图时序跟踪 Prompt。

    Image 2 必须由 ``ContextBuilder`` 预先拼为历史 mosaic；其中只允许出现经
    门控验证的历史观测。历史图用于适应外观变化，但不能覆盖 Image 1 的永久
    身份锚点。
    """

    if isinstance(history_count, bool) or not isinstance(history_count, int) or history_count <= 0:
        raise ValueError("history_count 必须是正整数")
    validate_bbox_protocol(bbox_protocol)
    if reference_bbox is not None and reference_bbox_norm1000_xyxy is not None:
        raise ValueError("reference_bbox 与 reference_bbox_norm1000_xyxy 不能同时传入")
    reference_note = "the target is marked or tightly cropped"
    if reference_bbox is not None:
        if isinstance(reference_bbox, str):
            if reference_bbox != "<bbox>":
                raise ValueError("字符串 reference_bbox 只允许使用 ms-swift 的 <bbox> 占位符")
            rendered = reference_bbox
        elif bbox_protocol == BBOX_PROTOCOL_NORM1000:
            rendered = str([round(value, 2) for value in validate_norm1000_xyxy(reference_bbox)])
        else:
            rendered = str([round(value, 2) for value in validate_xyxy(reference_bbox)])
        coordinate_text = (
            "normalized 0-to-1000 xyxy coordinates"
            if bbox_protocol == BBOX_PROTOCOL_NORM1000
            else "absolute pixel xyxy coordinates in Image 1's processor-resized pixel grid"
        )
        reference_note = (
            "this is the unmodified full initialization frame; "
            f"the target bbox is {rendered} in {coordinate_text}"
        )
    elif reference_bbox_norm1000_xyxy is not None:
        bbox = validate_norm1000_xyxy(reference_bbox_norm1000_xyxy)
        numbers = [round(value, 2) for value in bbox]
        reference_note = (
            "this is the unmodified full initialization frame; the target bbox is "
            f"{numbers} in normalized 0-to-1000 xyxy coordinates"
        )
    user_prompt = f"""Task: perform long-term identity-aware tracking with trusted temporal history.

Image 1: immutable initialization identity reference; {reference_note}.
Image 2: a chronological mosaic of {history_count} trusted target observations. Mosaic panels
may show legitimate pose, scale, illumination, or viewpoint changes. They are supporting evidence
and must never override identity conflicts with Image 1.
Image 3: current full frame to search globally.

Decision order:
1. Anchor identity to Image 1.
2. Use Image 2 only to explain plausible temporal appearance changes.
3. Search Image 3 globally and reject same-category distractors.
4. Output present and a box only when the initialized instance is visible and localizable.
5. Otherwise output absent and null. Do not invent finer-grained semantic states.

{target_text_section(target_text)}
{semantic_memory_section(semantic_memory)}

{tracking_output_schema(bbox_protocol, include_memory_update=include_memory_update)}"""
    return PromptSpec(
        name=MOSAIC_PROMPT_NAME,
        version=MOSAIC_PROMPT_VERSION,
        system_prompt=TRACKING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=3,
        include_memory_update=include_memory_update,
        bbox_protocol=bbox_protocol,
    )
