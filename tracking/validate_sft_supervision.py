#!/usr/bin/env python3
"""训练启动前审计 JSONL 的 SFT loss 档位与字段级 mask。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, MutableMapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.loss_mask import (  # noqa: E402
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_HARD_NULL,
    MEMORY_STATE_VERIFIED_UPDATE,
    SFT_SUPERVISION_FULL,
    SFT_SUPERVISION_MIXED_SFT,
    SFT_SUPERVISION_PROFILES,
    SFT_SUPERVISION_STATE_UPDATE_SFT,
    SFT_SUPERVISION_TRACKING_SFT,
    assistant_loss_scale_for_state,
    split_tracking_sft_response,
    validate_memory_supervision_state,
    validate_sft_supervision_profile,
)
from cogtrack.training.swift_dataset import read_jsonl  # noqa: E402


def _assistant_message(row: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("样本缺少 messages 列表")
    answers = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if len(answers) != 1 or not isinstance(answers[0].get("content"), str):
        raise ValueError("SFT 样本必须恰好包含一条字符串 assistant response")
    return answers[0]


def _assistant_text(row: Mapping[str, Any]) -> str:
    return str(_assistant_message(row)["content"])


def _check_memory_state(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    source: str,
) -> str:
    """校验三态记忆监督，并确认 metadata 与真正驱动 loss 的 loss_scale 一致。

    ms-swift 会丢弃 row 顶层的 ``metadata``；真正决定 loss 的只有
    ``messages[assistant].loss_scale``。两者必须成对，否则审计说一套、训练做
    另一套，而且这种不一致在训练日志里不可见。
    """

    raw_state = metadata.get("memory_supervision_state")
    if raw_state is None:
        raise ValueError(
            f"{source}: 缺少 memory_supervision_state；三态监督数据必须显式声明"
        )
    state = validate_memory_supervision_state(str(raw_state))

    assistant = _assistant_message(row)
    actual_scale = assistant.get("loss_scale")
    expected_scale = assistant_loss_scale_for_state(state)
    if actual_scale != expected_scale:
        raise ValueError(
            f"{source}: memory_supervision_state={state} 期望 "
            f"loss_scale={expected_scale!r}，实际为 {actual_scale!r}。"
            "metadata 不驱动 loss，两者必须一致。"
        )
    if actual_scale is not None and float(actual_scale) not in (0.0, 1.0):
        raise ValueError(
            f"{source}: LossScale.is_binary=True，loss_scale 只能是 0.0 或 1.0"
        )

    masked_flag = bool(metadata.get("memory_loss_masked", False))
    if masked_flag != (state == MEMORY_STATE_MASKED_UNKNOWN):
        raise ValueError(
            f"{source}: memory_loss_masked={masked_flag} 与 state={state} 矛盾"
        )

    response = _assistant_text(row)
    parts, weights = split_tracking_sft_response(response, require_memory_field=True)
    if "".join(parts) != response:
        raise ValueError(f"{source}: loss mask 分段无法无损重组 response")
    memory_value = str(parts[1]).strip()

    declared_verified_null = bool(metadata.get("memory_verified_null", False))
    if declared_verified_null and state != MEMORY_STATE_VERIFIED_HARD_NULL:
        raise ValueError(
            f"{source}: memory_verified_null=true 但 state={state}；已声明的负标签"
            "必须参与 loss，不能落回 masked_unknown"
        )

    if state == MEMORY_STATE_MASKED_UNKNOWN:
        if weights != [1.0, 0.0, 1.0]:
            raise ValueError(f"{source}: masked_unknown 的 memory 值必须权重 0")
        # Phase-1 不提供伪语义 teacher；null 只是保持三字段协议的占位符。
        if memory_value != "null":
            raise ValueError(
                f"{source}: masked_unknown 的占位值必须为 null，实际为 {memory_value!r}"
            )
    elif state == MEMORY_STATE_VERIFIED_HARD_NULL:
        if memory_value != "null":
            raise ValueError(f"{source}: verified_hard_null 的值必须为 null")
        if str(metadata.get("temporal_case")) != "absent" and not bool(
            metadata.get("memory_verified_null", False)
        ):
            raise ValueError(
                f"{source}: verified_hard_null 只允许 absent 行，或显式声明了"
                " memory_verified_null=true 的 present 行；否则 present 行缺少"
                "“确定不该更新”的证据，必须保持 masked_unknown"
            )
    else:  # MEMORY_STATE_VERIFIED_UPDATE
        if memory_value in {"null", ""}:
            raise ValueError(f"{source}: verified_update 必须携带非空状态文本")
        if str(metadata.get("temporal_case")) not in {"present", "absent"}:
            raise ValueError(f"{source}: verified_update 必须带 present/absent temporal_case")
    return state


def validate_dataset(
    path: str | Path,
    *,
    profile: str,
    states: MutableMapping[str, int] | None = None,
) -> int:
    """验证单个 JSONL；返回非空样本数，并按记忆监督状态累加计数。"""

    expected = validate_sft_supervision_profile(profile)
    if states is None:
        states = Counter()
    count = 0
    for row in read_jsonl(path):
        count += 1
        source = str(row.get("_source", f"{path}:?"))
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{source}: 缺少 metadata")
        actual = validate_sft_supervision_profile(
            str(metadata.get("sft_supervision_profile", SFT_SUPERVISION_FULL))
        )
        allowed_actual = (
            {SFT_SUPERVISION_TRACKING_SFT, SFT_SUPERVISION_STATE_UPDATE_SFT}
            if expected == SFT_SUPERVISION_MIXED_SFT
            else {expected}
        )
        if actual not in allowed_actual:
            raise ValueError(
                f"{source}: 数据监督档位为 {actual!r}，训练请求为 {expected!r}"
            )
        if metadata.get("memory_supervision") == "feasibility_null":
            raise ValueError(f"{source}: feasibility_null 只能做管线验证，禁止用于 SFT")
        masked_flag = bool(metadata.get("memory_loss_masked", False))
        if expected in {
            SFT_SUPERVISION_TRACKING_SFT,
            SFT_SUPERVISION_STATE_UPDATE_SFT,
            SFT_SUPERVISION_MIXED_SFT,
        }:
            if metadata.get("prompt_profile") != "vlt_v6":
                raise ValueError(
                    f"{source}: {expected} 当前只允许 vlt_v6 数据"
                )
            state = _check_memory_state(row, metadata, source=source)
            if actual == SFT_SUPERVISION_STATE_UPDATE_SFT and state == MEMORY_STATE_MASKED_UNKNOWN:
                raise ValueError(
                    f"{source}: state_update_sft 不能包含 masked_unknown；"
                    "未验证的行应留在 tracking_sft"
                )
            states[state] += 1
        elif masked_flag:
            raise ValueError(f"{source}: full 档位不能混入 memory_loss_masked=true 样本")
    if count == 0:
        raise ValueError(f"训练数据为空：{path}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="待审计 JSONL；可重复。")
    parser.add_argument(
        "--profile",
        required=True,
        choices=tuple(sorted(SFT_SUPERVISION_PROFILES)),
    )
    args = parser.parse_args()
    states: Counter[str] = Counter()
    try:
        counts = {
            str(Path(path).expanduser()): validate_dataset(
                path, profile=args.profile, states=states
            )
            for path in args.dataset
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print(
        "[CognitiveTrack] SFT loss 档位检查通过："
        f"profile={args.profile} samples={sum(counts.values())} files={len(counts)}"
    )
    if states:
        total = sum(states.values())
        print("[CognitiveTrack] 记忆监督状态分布：")
        for state in (
            MEMORY_STATE_MASKED_UNKNOWN,
            MEMORY_STATE_VERIFIED_HARD_NULL,
            MEMORY_STATE_VERIFIED_UPDATE,
        ):
            value = states.get(state, 0)
            share = (value / total * 100) if total else 0.0
            loss = "0（被 mask）" if state == MEMORY_STATE_MASKED_UNKNOWN else "1（全监督）"
            print(f"  {state:20s} {value:7d}  {share:5.1f}%  memory 值 loss={loss}")
        if not states.get(MEMORY_STATE_VERIFIED_UPDATE):
            print(
                "  [提示] verified_update 为 0：第三字段只有 null 方向的监督信号，"
                "训练后必须测量 present 帧的非空提议率，确认未塌缩成恒定 null。"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
