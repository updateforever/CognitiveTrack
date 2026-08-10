"""CognitiveTrack 的 ms-swift GRPO outcome reward 插件。

在 ms-swift 4.x 中通过以下参数加载：

.. code-block:: bash

    --external_plugins cogtrack/training/grpo_rewards.py \
    --reward_funcs cogtrack_format cogtrack_presence cogtrack_bbox \
                   cogtrack_consistency

每个 reward 都只依赖 completion 与数据集中的 ``solution``/监督列，互相独立，
便于在实验中调整 ``--reward_weights`` 做消融。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

from cogtrack.training.swift_dataset import parse_training_tracking_answer

try:  # 独立单元测试环境可以不安装 ms-swift。
    from swift.rewards import ORM, orms
except ImportError:  # pragma: no cover - 仅用于无 ms-swift 的轻量测试环境
    class ORM:  # type: ignore[no-redef]
        def __init__(self, args: Any = None, **kwargs: Any):
            self.args = args

    orms: dict[str, Any] = {}


PRESENCE_VALUES = {"present", "absent", "uncertain"}


def _completion_text(value: Any) -> str:
    """兼容 completion 字符串、消息字典或消息列表。"""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("content", value.get("text", "")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in reversed(value):
            if isinstance(item, Mapping) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        if value:
            return _completion_text(value[-1])
    return str(value)


def _parse_payload(value: Any) -> tuple[Optional[dict[str, Any]], bool]:
    """返回解析对象与是否为无 code-fence 的严格 JSON。"""

    if isinstance(value, Mapping):
        return dict(value), True
    text = _completion_text(value).strip()
    strict = not text.startswith("```")
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 字段：{key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"非标准 JSON 常量：{value}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, False
    return (payload, strict) if isinstance(payload, dict) else (None, False)


def _presence(payload: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    value = payload.get("target_presence", payload.get("target_status"))
    value = str(value).strip().lower() if value is not None else None
    return value if value in PRESENCE_VALUES else None


def _bbox_value(payload: Optional[Mapping[str, Any]]) -> tuple[Any, str]:
    if not payload:
        return None, "norm1000_xyxy"
    if "bbox_xywh" in payload:
        return payload["bbox_xywh"], "xywh"
    if "bbox_xyxy" in payload:
        return payload["bbox_xyxy"], "xyxy"
    if "bbox_norm1000_xyxy" in payload:
        return payload["bbox_norm1000_xyxy"], "norm1000_xyxy"
    return payload.get("bbox"), str(payload.get("bbox_format", "norm1000_xyxy"))


def _bbox_xyxy(value: Any, bbox_format: str) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x, y, third, fourth = (float(value[index]) for index in range(4))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in (x, y, third, fourth)):
        return None
    if "xywh" in bbox_format.lower():
        if third <= 0 or fourth <= 0:
            return None
        return x, y, x + third, y + fourth
    if third <= x or fourth <= y:
        return None
    if "norm1000" in bbox_format.lower() and any(
        coordinate < 0.0 or coordinate > 1000.0
        for coordinate in (x, y, third, fourth)
    ):
        return None
    return x, y, third, fourth


def _iou(first: Any, first_format: str, second: Any, second_format: str) -> float:
    box_a = _bbox_xyxy(first, first_format)
    box_b = _bbox_xyxy(second, second_format)
    if box_a is None or box_b is None:
        return 0.0
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _at(value: Any, index: int, batch_size: int) -> Any:
    """读取 batch 列的第 index 项；标量则广播到整个 batch。"""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        if len(value) == batch_size:
            return value[index]
    return value


def _reference_payload(
    index: int,
    batch_size: int,
    solution: Any,
    kwargs: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    candidate = _at(solution, index, batch_size) if solution is not None else None
    payload, _ = _parse_payload(candidate)
    result = dict(payload or {})

    # 某些 ms-swift 数据管线会把原始列直接作为 kwargs 传入 reward。即使
    # solution 未保留，也可以由独立监督列恢复参考答案。
    for key in (
        "target_presence",
        "target_status",
        "bbox",
        "bbox_norm1000_xyxy",
        "bbox_format",
    ):
        if key in kwargs and key not in result:
            result[key] = _at(kwargs[key], index, batch_size)
    return result or None


def _state_consistent(payload: Optional[Mapping[str, Any]]) -> bool:
    """复用推理协议判定二字段与状态是否完整自洽。"""

    if payload is None:
        return False
    try:
        parse_training_tracking_answer(payload)
    except ValueError:
        return False
    return True


class CognitiveFormatReward(ORM):
    """奖励可严格解析、字段取值合法且 bbox/state 一致的 JSON。"""

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        rewards = []
        for completion in completions:
            payload, strict = _parse_payload(completion)
            # bare JSON 的格式约束由 strict 控制；字段与状态约束则完全复用
            # SFT/推理协议。二者不再维护容易漂移的平行规则。
            valid = strict and _state_consistent(payload)
            rewards.append(1.0 if valid else 0.0)
        return rewards


class CognitivePresenceReward(ORM):
    """奖励 target_presence 与参考答案一致。"""

    def __call__(self, completions: Sequence[Any], solution: Any = None, **kwargs: Any) -> list[float]:
        rewards = []
        batch_size = len(completions)
        for index, completion in enumerate(completions):
            prediction, _ = _parse_payload(completion)
            reference = _reference_payload(index, batch_size, solution, kwargs)
            expected = _presence(reference)
            rewards.append(1.0 if expected is not None and _presence(prediction) == expected else 0.0)
        return rewards


class CognitiveBBoxReward(ORM):
    """Present 样本使用 IoU；Absent 样本奖励不输出 bbox。"""

    def __call__(self, completions: Sequence[Any], solution: Any = None, **kwargs: Any) -> list[float]:
        rewards = []
        batch_size = len(completions)
        for index, completion in enumerate(completions):
            prediction, _ = _parse_payload(completion)
            reference = _reference_payload(index, batch_size, solution, kwargs)
            expected_presence = _presence(reference)
            pred_presence = _presence(prediction)
            pred_bbox, pred_format = _bbox_value(prediction)
            ref_bbox, ref_format = _bbox_value(reference)
            if expected_presence == "absent":
                rewards.append(1.0 if pred_presence == "absent" and pred_bbox is None else 0.0)
            elif expected_presence == "present" and pred_presence == "present":
                rewards.append(_iou(pred_bbox, pred_format, ref_bbox, ref_format))
            else:
                rewards.append(0.0)
        return rewards


class CognitiveConsistencyReward(ORM):
    """奖励 target_status 与 bbox 内部自洽，抑制投机输出。"""

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return [
            1.0 if _state_consistent(_parse_payload(completion)[0]) else 0.0
            for completion in completions
        ]


# external_plugins 导入本文件后，ms-swift 会从全局 orms 注册表解析名称。
orms.update(
    {
        "cogtrack_format": CognitiveFormatReward,
        "cogtrack_presence": CognitivePresenceReward,
        "cogtrack_bbox": CognitiveBBoxReward,
        "cogtrack_consistency": CognitiveConsistencyReward,
    }
)


__all__ = [
    "CognitiveBBoxReward",
    "CognitiveConsistencyReward",
    "CognitiveFormatReward",
    "CognitivePresenceReward",
]
