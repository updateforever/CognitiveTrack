"""目标存在性与可选语义记忆任务共享的 Prompt 数据结构和 v4 输出约束。"""

from dataclasses import dataclass

from ..protocol.bbox import (
    BBOX_PROTOCOL_NORM1000,
    BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    bbox_protocol_json_key,
    validate_bbox_protocol,
)


@dataclass(frozen=True)
class PromptSpec:
    """一个可版本化、可记录到实验 manifest 的 Prompt。"""

    name: str
    version: str
    system_prompt: str
    user_prompt: str
    expected_image_count: int
    #: 主跟踪解析器是否要求第三个 ``memory_update`` 字段。身份等独立 Prompt
    #: 不使用该字段；presence-only SFT 推理将其显式设为 False。
    include_memory_update: bool = False
    #: 本 Prompt 要求模型使用的 bbox 坐标协议，解析端必须用同一个值。
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000

    def __post_init__(self) -> None:
        validate_bbox_protocol(self.bbox_protocol)
        if not isinstance(self.include_memory_update, bool):
            raise TypeError("include_memory_update 必须是 bool")


TRACKING_SYSTEM_PROMPT = """You are a rigorous long-term single-object tracking model.
The target is the exact instance marked in the initialization reference. Similar-looking
objects are distractors, not the target. Return exactly one JSON object without Markdown
or additional text."""


_BBOX_RULES = {
    BBOX_PROTOCOL_NORM1000: (
        "- present + localizable: bbox is required and uses [0,1000] normalized xyxy\n"
        "  coordinates relative to the current frame.\n"
        "- Never output pixel coordinates, xywh, Markdown fences, or extra keys."
    ),
    BBOX_PROTOCOL_QWEN_ABS_PIXEL: (
        "- present + localizable: bbox is required and uses absolute pixel xyxy\n"
        "  coordinates in the current frame's own pixel grid.\n"
        "- Never normalize the coordinates, and never output xywh, Markdown fences,\n"
        "  or extra keys."
    ),
}


def tracking_output_schema(
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    *,
    include_memory_update: bool = True,
) -> str:
    """按坐标协议渲染二字段训练或三字段推理输出约束。

    字段名随协议变化：``qwen_abs_pixel`` 用 ``bbox_pixel_xyxy``，避免沿用
    ``bbox_norm1000_xyxy`` 这个会把模型往归一化方向带的名字。
    """

    validate_bbox_protocol(bbox_protocol)
    # 模板本身保持合法 JSON 形态并默认走 null 快路径。需要更新时，模型按下面
    # 的规则把 null 替换为短字符串；不要在 JSON 值位置展示 ``null or ...`` 这
    # 类伪语法，否则零样本模型容易把“不变化”也口头写成字符串。
    memory_line = ',\n  "memory_update": null' if include_memory_update else ""
    memory_rules = (
        """
- memory_update must be the final key.
- The JSON template shows the default null fast path. Replace null with a JSON string only when
  one of the qualifying changes below is actually visible.
- Use null for ordinary motion, scale change, blur, duplicated information, or absent targets.
- When there is no qualifying new cue, output the JSON value null. Never write a string such as
  "no change", "unchanged", "none", or "no update".
- Use a short non-empty string only when the target reveals a materially changed viewpoint,
  configuration, appearance, or a new stable discriminative cue useful for future tracking.
- Describe only the new observable delta, not the full target and not your reasoning.
- Keep memory_update within 30 English words."""
        if include_memory_update
        else ""
    )
    key_count = "three" if include_memory_update else "two"
    return f"""Return exactly these {key_count} keys:
{{
  "target_status": "present | absent",
  "{bbox_protocol_json_key(bbox_protocol)}": [x1, y1, x2, y2] or null{memory_line}
}}

Protocol rules:
{_BBOX_RULES[bbox_protocol]}
- present means the exact initialized target is visible and localizable in the current frame.
- absent means the exact initialized target is not visible or cannot be localized; bbox is null.
- Similar-category distractors do not count as present.
- Do not output identity labels, visibility sub-states, confidence scores, reasoning,
  Markdown, or extra keys.{memory_rules}"""


#: 向后兼容的 norm1000 常量，供既有训练数据构造代码继续引用。
TRACKING_OUTPUT_SCHEMA = tracking_output_schema(BBOX_PROTOCOL_NORM1000)


def target_text_section(target_text: str) -> str:
    """按需加入已有语义身份描述；初始图像始终是最高优先级证据。"""

    text = target_text.strip() if isinstance(target_text, str) else ""
    if not text:
        return "No prior semantic description is available. Derive identity from Image 1."
    return f"Prior semantic description (supporting evidence only; Image 1 has priority):\n{text}"


def semantic_memory_section(semantic_memory: str) -> str:
    """加入已接受的语义变化记忆；永久初始化锚点始终优先。"""

    text = semantic_memory.strip() if isinstance(semantic_memory, str) else ""
    if not text:
        return "Current accepted semantic change memory: (empty)"
    return f"Current accepted semantic change memory (supporting evidence only):\n{text}"
