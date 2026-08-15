#!/usr/bin/env python3
"""训练启动前审计 JSONL 的 SFT loss 档位与字段级 mask。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training.loss_mask import (  # noqa: E402
    SFT_SUPERVISION_FULL,
    SFT_SUPERVISION_PROFILES,
    SFT_SUPERVISION_TRACKING_CORE,
    split_tracking_core_response,
    validate_sft_supervision_profile,
)
from cogtrack.training.swift_dataset import read_jsonl  # noqa: E402


def _assistant_text(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("样本缺少 messages 列表")
    answers = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if len(answers) != 1 or not isinstance(answers[0], str):
        raise ValueError("SFT 样本必须恰好包含一条字符串 assistant response")
    return answers[0]


def validate_dataset(path: str | Path, *, profile: str) -> int:
    """验证单个 JSONL；返回非空样本数。"""

    expected = validate_sft_supervision_profile(profile)
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
        if actual != expected:
            raise ValueError(
                f"{source}: 数据监督档位为 {actual!r}，训练请求为 {expected!r}"
            )
        if metadata.get("memory_supervision") == "feasibility_null":
            raise ValueError(f"{source}: feasibility_null 只能做管线验证，禁止用于 SFT")
        masked_flag = bool(metadata.get("memory_loss_masked", False))
        if expected == SFT_SUPERVISION_TRACKING_CORE:
            if metadata.get("prompt_profile") != "vlt_v6":
                raise ValueError(f"{source}: tracking_core 当前只允许 vlt_v6 数据")
            if metadata.get("memory_supervision") != "masked_null":
                raise ValueError(f"{source}: tracking_core 数据必须使用 memory_supervision=masked_null")
            if not masked_flag:
                raise ValueError(f"{source}: tracking_core 样本必须声明 memory_loss_masked=true")
            response = _assistant_text(row)
            parts, weights = split_tracking_core_response(
                response,
                require_memory_field=True,
            )
            if "".join(parts) != response or weights != [1.0, 0.0, 1.0]:
                raise ValueError(f"{source}: memory_update loss 分段不符合预期")
            # Phase-1 不提供伪语义 teacher；null 只是保持最终三字段协议的占位符。
            if str(parts[1]).strip() != "null":
                raise ValueError(f"{source}: tracking_core 的 memory_update 占位值必须为 null")
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
    try:
        counts = {
            str(Path(path).expanduser()): validate_dataset(path, profile=args.profile)
            for path in args.dataset
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print(
        "[CognitiveTrack] SFT loss 档位检查通过："
        f"profile={args.profile} samples={sum(counts.values())} files={len(counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
