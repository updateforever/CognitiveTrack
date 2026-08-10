"""已隔离的旧版独立语义记忆 Prompt。

v4 主链路使用 pair/mosaic 输出末尾的 ``memory_update`` 字段，单次完成定位与
记忆开关判定，不再额外调用本 Prompt。本模块仅保留给历史实验复现。
"""

from .common import PromptSpec

MEMORY_PROMPT_NAME = "memory_update"
MEMORY_PROMPT_VERSION = "2.0.0"


def build_memory_prompt(existing_memory: str, requested_type: str = "positive") -> PromptSpec:
    """构造单图记忆更新建议任务。

    该 Prompt 只让 VLM 提议一条简短、可验证的语义记忆，不能自行修改
    ``MemoryBank``。``GatedMemoryUpdatePolicy`` 仍会检查来源和连续性。
    """

    valid_types = {"positive", "negative", "semantic"}
    if requested_type not in valid_types:
        raise ValueError(f"requested_type 必须是 {sorted(valid_types)} 之一")
    current = existing_memory.strip() if isinstance(existing_memory, str) else ""
    current = current or "(empty)"
    system_prompt = """You maintain conservative semantic memory for identity-aware tracking.
Only propose stable, visually supported information. Never invent hidden attributes. Return exactly
one JSON object without Markdown or additional text."""
    user_prompt = f"""Image 1: a candidate observation already processed by the tracker.
Requested memory type: {requested_type}
Existing semantic memory:
{current}

Propose an update only if Image 1 adds stable, discriminative and non-duplicated evidence.
For negative memory, describe cues that distinguish a rejected distractor from the initialized
target. The local policy will independently decide whether this proposal is trusted.

Return exactly:
{{
  "update_memory": true or false,
  "memory_type": "positive | negative | semantic | none",
  "summary": "short factual memory; empty when no update",
  "reasoning": "brief evidence for the proposal"
}}
When update_memory is false, memory_type must be none. Do not return confidence scores or extra keys."""
    return PromptSpec(
        name=MEMORY_PROMPT_NAME,
        version=MEMORY_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_image_count=1,
    )
