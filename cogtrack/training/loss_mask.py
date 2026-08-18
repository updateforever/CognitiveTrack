"""CognitiveTrack SFT 的字段级 loss mask。

大规模 tracking SFT 学习存在性与 bbox。训练样本仍保留最终
``memory_update`` 字段，使训练和正式推理共用同一输出协议；只有该字段的 JSON 值
不参与交叉熵。字段名、冒号和对象闭合符继续受监督，从而不牺牲结构稳定性。

这里按本项目导出的紧凑 JSON 边界切分，不解析 ``<bbox>`` 占位符。ms-swift 会在随后
的模型模板阶段把该占位符展开为真实坐标 token，并继承所在前缀的权重 1。
"""

from __future__ import annotations

import json
from typing import Any

from cogtrack.protocol import MEMORY_UPDATE_JSON_KEY

SFT_SUPERVISION_FULL = "full"
# 面向新数据与新训练的唯一正式名称。
SFT_SUPERVISION_TRACKING_SFT = "tracking_sft"
# 可靠状态更新数据：memory_update 值和跟踪字段均全量监督。
SFT_SUPERVISION_STATE_UPDATE_SFT = "state_update_sft"
# 训练侧档位：允许 tracking_sft 与 state_update_sft 行在同一 JSONL/batch 中，
# 但每行仍保留其原始 profile 和三态 metadata，由 loss_scale 逐行裁决。
SFT_SUPERVISION_MIXED_SFT = "mixed_sft"
# 仅用于读取已经生成的历史 release / 命令；新产物不得再写入这个名字。
SFT_SUPERVISION_TRACKING_CORE = "tracking_core"
SFT_SUPERVISION_PROFILES = frozenset(
    {
        SFT_SUPERVISION_FULL,
        SFT_SUPERVISION_TRACKING_SFT,
        SFT_SUPERVISION_STATE_UPDATE_SFT,
        SFT_SUPERVISION_MIXED_SFT,
        SFT_SUPERVISION_TRACKING_CORE,
    }
)

#: 记忆监督三态。
#:
#: - ``masked_unknown``：present 且没有可靠状态标签。占位 ``null`` 不是负标签，
#:   其值不参与 loss，避免把模型教成“永不更新”。
#: - ``verified_hard_null``：标注明确确认本帧不需要替换 memory。``null`` 参与 loss。
#: - ``verified_update``：有可靠完整状态文本，文本参与 loss。
MEMORY_STATE_MASKED_UNKNOWN = "masked_unknown"
MEMORY_STATE_VERIFIED_HARD_NULL = "verified_hard_null"
MEMORY_STATE_VERIFIED_UPDATE = "verified_update"
MEMORY_SUPERVISION_STATES = frozenset(
    {
        MEMORY_STATE_MASKED_UNKNOWN,
        MEMORY_STATE_VERIFIED_HARD_NULL,
        MEMORY_STATE_VERIFIED_UPDATE,
    }
)

#: 需要完整监督 memory 值的状态。
_FULLY_SUPERVISED_MEMORY_STATES = frozenset(
    {MEMORY_STATE_VERIFIED_HARD_NULL, MEMORY_STATE_VERIFIED_UPDATE}
)


def validate_memory_supervision_state(value: str) -> str:
    """规范化记忆监督状态，防止拼写错误退化为静默的错误档位。"""

    normalized = str(value).strip().lower()
    if normalized not in MEMORY_SUPERVISION_STATES:
        raise ValueError(
            f"memory_supervision_state 必须是 {sorted(MEMORY_SUPERVISION_STATES)} 之一"
        )
    return normalized


def decide_memory_supervision_state(
    *,
    status: str,
    memory_update: Any,
    verified_null: bool = False,
) -> str:
    """由 presence 与记忆标签判定三态。**源层与训练视图共用的唯一真源。**

    - ``present/absent`` 且有非空文本 -> ``verified_update``。absent 文本可描述消失
      转折，reappearance 文本可恢复当前动态指代表达。
    - ``present/absent`` 且为 ``None``：默认 ``masked_unknown``，占位 ``null`` 不是负标签。
      只有调用方显式传 ``verified_null=True`` 才升级为 ``verified_hard_null``。该
      参数表示标注**已证明**这一帧不需要更新（例如 MGIT action 分段内部、远离人工
      标注边界的稳定帧），此时 ``null`` 是真负标签，必须参与 loss。默认值刻意为
      ``False``：缺证据时保持 masked，宁可少监督也不能凭空造负标签。

    刻意不接受 ``present`` + 空字符串：那是标签生产环节的 bug，不能静默退化成
    “已验证要把 memory 写成空串”。
    """

    normalized_status = str(status).strip().lower()
    if normalized_status not in {"present", "absent"}:
        raise ValueError(f"status 必须是 present/absent，实际为 {status!r}")
    if verified_null and memory_update is not None:
        raise ValueError("verified_null=True 时 memory_update 必须为 null")
    if memory_update is None:
        if verified_null:
            return MEMORY_STATE_VERIFIED_HARD_NULL
        return MEMORY_STATE_MASKED_UNKNOWN
    if not isinstance(memory_update, str):
        raise ValueError("memory_update 必须是字符串或 null")
    if not memory_update.strip():
        raise ValueError("verified_update 的 memory_update 文本不能为空")
    return MEMORY_STATE_VERIFIED_UPDATE


def assistant_loss_scale_for_state(state: str) -> float | None:
    """返回该状态应写入 ``messages[assistant].loss_scale`` 的值。

    这是 ms-swift 4.3.1 的原生 per-message 通道（``swift/loss_scale/base.py``
    的 ``LossScale._inner_call``），不需要修改 site-packages：

    - 返回 ``None`` 表示**不写**该 key，于是落到外部插件的
      ``[核心前缀, memory 值, JSON 后缀] -> [1, 0, 1]`` 分段，memory 值被 mask；
    - 返回 ``1.0`` 会让 ms-swift 直接返回 ``[context], [1.0]``，**绕过**插件分段，
      对整条 response 全量监督。

    两种行为可以在同一数据集、同一 batch 内共存。由于 ``LossScale.is_binary``
    为 ``True``，只有 ``0.0``/``1.0`` 是安全值。
    """

    normalized = validate_memory_supervision_state(state)
    return 1.0 if normalized in _FULLY_SUPERVISED_MEMORY_STATES else None

_MEMORY_FIELD_MARKER = f',"{MEMORY_UPDATE_JSON_KEY}":'


def validate_sft_supervision_profile(value: str) -> str:
    """规范化训练监督档位；历史 ``tracking_core`` 只作为读取别名。"""

    normalized = str(value).strip().lower()
    if normalized not in SFT_SUPERVISION_PROFILES:
        raise ValueError(
            f"SFT supervision profile 必须是 {sorted(SFT_SUPERVISION_PROFILES)} 之一"
        )
    if normalized == SFT_SUPERVISION_TRACKING_CORE:
        return SFT_SUPERVISION_TRACKING_SFT
    return normalized


def split_tracking_sft_response(
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
            raise TypeError("tracking_sft loss mask 只接受字符串 assistant response")
        return [context], [1.0]

    marker_index = context.rfind(_MEMORY_FIELD_MARKER)
    if marker_index < 0:
        if require_memory_field or f'"{MEMORY_UPDATE_JSON_KEY}"' in context:
            raise ValueError(
                "tracking_sft 样本缺少末尾 canonical memory_update 字段："
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


def split_tracking_core_response(
    context: Any,
    *,
    require_memory_field: bool = False,
) -> tuple[list[Any], list[float]]:
    """历史兼容别名；新代码使用 :func:`split_tracking_sft_response`。"""

    return split_tracking_sft_response(
        context, require_memory_field=require_memory_field
    )


__all__ = [
    "MEMORY_STATE_MASKED_UNKNOWN",
    "MEMORY_STATE_VERIFIED_HARD_NULL",
    "MEMORY_STATE_VERIFIED_UPDATE",
    "MEMORY_SUPERVISION_STATES",
    "SFT_SUPERVISION_FULL",
    "SFT_SUPERVISION_PROFILES",
    "SFT_SUPERVISION_MIXED_SFT",
    "SFT_SUPERVISION_TRACKING_CORE",
    "SFT_SUPERVISION_STATE_UPDATE_SFT",
    "SFT_SUPERVISION_TRACKING_SFT",
    "assistant_loss_scale_for_state",
    "decide_memory_supervision_state",
    "split_tracking_core_response",
    "split_tracking_sft_response",
    "validate_memory_supervision_state",
    "validate_sft_supervision_profile",
]
