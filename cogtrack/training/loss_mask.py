"""CognitiveTrack SFT 的字段级 loss mask。

第一阶段只学习核心跟踪输出：``target_status`` 与 bbox。训练样本仍保留最终
``memory_update`` 字段，使训练和正式推理共用同一输出协议；只有该字段的 JSON 值
不参与交叉熵。字段名、冒号和对象闭合符继续受监督，从而不牺牲结构稳定性。

这里按本项目导出的紧凑 JSON 边界切分，不解析 ``<bbox>`` 占位符。ms-swift 会在随后
的模型模板阶段把该占位符展开为真实坐标 token，并继承所在前缀的权重 1。
"""

from __future__ import annotations

import json
from typing import Any

SFT_SUPERVISION_FULL = "full"
SFT_SUPERVISION_TRACKING_CORE = "tracking_core"
SFT_SUPERVISION_PROFILES = frozenset(
    {SFT_SUPERVISION_FULL, SFT_SUPERVISION_TRACKING_CORE}
)

_MEMORY_FIELD_MARKER = ',"memory_update":'


def validate_sft_supervision_profile(value: str) -> str:
    """规范化训练监督档位，防止错误拼写退化为 ms-swift 默认全量 loss。"""

    normalized = str(value).strip().lower()
    if normalized not in SFT_SUPERVISION_PROFILES:
        raise ValueError(
            f"SFT supervision profile 必须是 {sorted(SFT_SUPERVISION_PROFILES)} 之一"
        )
    return normalized


def split_tracking_core_response(
    context: Any,
    *,
    require_memory_field: bool = False,
) -> tuple[list[Any], list[float]]:
    """把 assistant response 切成 ``核心前缀 / 记忆值 / 闭合后缀``。

    返回值直接符合 ms-swift ``LossScale.get_loss_scale`` 的约定。只有
    ``memory_update`` 的值权重为 0；例如核心 JSON 前缀权重为 1，``null`` 权重为
    0，最后的 ``}`` 权重为 1。

    两字段历史数据在非严格模式下保持全量监督。若出现字段名却不符合 canonical
    紧凑格式，则直接报错，避免因为空格或字段顺序漂移而静默监督了记忆文本。
    """

    if not isinstance(context, str):
        if require_memory_field:
            raise TypeError("tracking_core loss mask 只接受字符串 assistant response")
        return [context], [1.0]

    marker_index = context.rfind(_MEMORY_FIELD_MARKER)
    if marker_index < 0:
        if require_memory_field or '"memory_update"' in context:
            raise ValueError(
                "tracking_core 样本缺少末尾 canonical memory_update 字段："
                f"期望 {_MEMORY_FIELD_MARKER!r}"
            )
        return [context], [1.0]

    value_start = marker_index + len(_MEMORY_FIELD_MARKER)
    object_end = context.rfind("}")
    if object_end < value_start or context[object_end + 1 :].strip():
        raise ValueError("memory_update 必须是 JSON 对象中的最后一个字段")

    prefix = context[:value_start]
    memory_value = context[value_start:object_end]
    suffix = context[object_end:]
    if not memory_value.strip():
        raise ValueError("memory_update 的 teacher-forced 占位值不能为空")
    try:
        json.loads(memory_value)
    except json.JSONDecodeError as exc:
        raise ValueError("memory_update 必须是最后一个且自身为合法 JSON 值") from exc
    if prefix + memory_value + suffix != context:
        raise AssertionError("loss mask 分段必须无损重组原始 response")
    return [prefix, memory_value, suffix], [1.0, 0.0, 1.0]


__all__ = [
    "SFT_SUPERVISION_FULL",
    "SFT_SUPERVISION_PROFILES",
    "SFT_SUPERVISION_TRACKING_CORE",
    "split_tracking_core_response",
    "validate_sft_supervision_profile",
]
