"""完整过去参考帧 + 参考坐标 + 当前全图的无时序 pair Prompt。"""

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

PAIR_PROMPT_NAME = "cognitive_pair"
PAIR_PROMPT_VERSION = "4.4.0"


def _reference_note(
    reference_bbox_norm1000_xyxy: Sequence[float] | None,
    *,
    reference_has_box: bool,
    reference_bbox: Sequence[float] | str | None = None,
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
) -> str:
    """渲染过去 reference 框说明。

    ``reference_bbox='<bbox>'`` 专供 ms-swift 官方 grounding 占位符：具体数值由
    模型模板在图像 resize 后填入。数值路径保留给在线推理和历史样本兼容。
    """

    validate_bbox_protocol(bbox_protocol)
    if reference_bbox is not None and reference_bbox_norm1000_xyxy is not None:
        raise ValueError("reference_bbox 与 reference_bbox_norm1000_xyxy 不能同时传入")
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
        return (
            "Image 1 is an unmodified full earlier reference frame. "
            f"The target bbox in Image 1 is {rendered} in {coordinate_text}."
        )
    if reference_bbox_norm1000_xyxy is not None:
        bbox = validate_norm1000_xyxy(reference_bbox_norm1000_xyxy)
        numbers = [round(value, 2) for value in bbox]
        return (
            "Image 1 is an unmodified full earlier reference frame. "
            "The target bbox in Image 1 is "
            f"{numbers} in normalized 0-to-1000 xyxy coordinates."
        )
    if reference_has_box:
        return "The target is marked by a visible bounding box in Image 1."
    return "Image 1 is a tight crop containing only the initialized target."


def build_pair_prompt(
    target_text: str = "",
    semantic_memory: str = "",
    reference_has_box: bool = True,
    reference_bbox_norm1000_xyxy: Sequence[float] | None = None,
    reference_bbox: Sequence[float] | str | None = None,
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    include_memory_update: bool = True,
) -> PromptSpec:
    """构造两图认知跟踪 Prompt。

    图像顺序固定：Image 1 为更早的身份参考，Image 2 为当前完整视频帧。在线推理
    的 Image 1 恰好是初始化帧，训练 pair 则可使用任意严格更早的 present 帧。
    新训练导出传入 ``reference_bbox='<bbox>'``，由 ms-swift 根据具体模型族填入
    官方坐标：Qwen2.5-VL 为 resize 后绝对像素，Qwen3-VL 为 norm1000。
    ``reference_bbox_norm1000_xyxy`` 与 ``reference_has_box`` 仅保留兼容路径。
    """

    reference_note = _reference_note(
        reference_bbox_norm1000_xyxy,
        reference_has_box=reference_has_box,
        reference_bbox=reference_bbox,
        bbox_protocol=bbox_protocol,
    )
    user_prompt = f"""Task: verify and localize the referenced target in the current frame.

Image 1: earlier identity reference. {reference_note}
Image 2: current full frame to search globally.

Decision order:
1. Compare the current frame with the referenced instance in Image 1.
2. Reject same-category distractors using instance-specific appearance.
3. Output present and a box only when that exact target is visible and localizable.
4. Otherwise output absent and null. Do not invent finer-grained semantic states.

{target_text_section(target_text)}
{semantic_memory_section(semantic_memory)}

{tracking_output_schema(bbox_protocol, include_memory_update=include_memory_update)}"""
    return PromptSpec(
        name=PAIR_PROMPT_NAME,
        version=PAIR_PROMPT_VERSION,
        system_prompt=TRACKING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=2,
        include_memory_update=include_memory_update,
        bbox_protocol=bbox_protocol,
    )
