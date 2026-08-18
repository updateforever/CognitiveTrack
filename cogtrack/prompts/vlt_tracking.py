"""VLT-v6.4 统一长时跟踪 Prompt。

训练版模型通过 SFT 内化输出字段和坐标协议，因此 system prompt 只定义跟踪任务本身：
身份不可覆盖、利用历史轨迹、分析当前目标状态和按需更新状态记忆。user prompt 只携带
本次序列的不可变初始身份、当前维护状态和一句跟踪触发语。
"""

from __future__ import annotations

from ..protocol.bbox import (
    BBOX_PROTOCOL_NORM1000,
    validate_bbox_protocol,
)
from .common import PromptSpec

VLT_TRACKING_PROMPT_VERSION = "6.4.0"
VLT_PAIR_PROMPT_NAME = "cognitive_vlt_pair"
VLT_MOSAIC_PROMPT_NAME = "cognitive_vlt_mosaic"


def _system_prompt() -> str:
    """渲染训练模型使用的极简任务定义，不重复其已经学过的输出协议。"""

    return """You are a long-term vision-language single-object tracker. Image 1 with the red
bounding box defines the initialized target identity as the permanent anchor. Image 2 shows
the temporally ordered historical prediction trajectory.

Image 3 is the current search frame: determine the target status and localize it precisely
when present. Maintain and update the target-state memory only when significant state changes
occur that would benefit future tracking."""


def _dynamic_value(value: str, *, empty: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    return text or empty


def build_vlt_tracking_prompt(
    *,
    history_count: int = 0,
    target_text: str = "",
    semantic_memory: str = "",
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    include_memory_update: bool = True,
) -> PromptSpec:
    """构造 VLT-v6.4 的极简动态输入。

    ``history_count`` 只用于核对图片数量和兼容旧 pair/mosaic 调用，不把帧数、决策步骤
    等重复写进每条 user 消息，也不会向 System Prompt 暴露双图分支。正式 v6.4 配置会
    复制锚点补齐三个历史 panel，从而始终保持三图接口。
    """

    if isinstance(history_count, bool) or not isinstance(history_count, int) or history_count < 0:
        raise ValueError("history_count 必须是非负整数")
    validate_bbox_protocol(bbox_protocol)

    prompt_name = VLT_MOSAIC_PROMPT_NAME if history_count else VLT_PAIR_PROMPT_NAME
    expected_image_count = 3 if history_count else 2
    initial_identity = _dynamic_value(
        target_text,
        empty="the target marked by the red box in Image 1",
    )
    current_state = _dynamic_value(semantic_memory, empty=initial_identity)
    user_prompt = "\n".join(
        (
            "Initial target identity: " + initial_identity,
            "Current maintained target state: " + current_state,
            "Track output:",
        )
    )
    return PromptSpec(
        name=prompt_name,
        version=VLT_TRACKING_PROMPT_VERSION,
        system_prompt=_system_prompt(),
        user_prompt=user_prompt,
        expected_image_count=expected_image_count,
        include_memory_update=include_memory_update,
        bbox_protocol=bbox_protocol,
    )


__all__ = [
    "VLT_MOSAIC_PROMPT_NAME",
    "VLT_PAIR_PROMPT_NAME",
    "VLT_TRACKING_PROMPT_VERSION",
    "build_vlt_tracking_prompt",
]
