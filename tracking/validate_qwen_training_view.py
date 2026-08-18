#!/usr/bin/env python3
"""训练启动前验证 Qwen 模型代际与坐标数据视图严格匹配。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.model_family import validate_model_dataset_family  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="本地模型或 adapter 目录。")
    parser.add_argument("--dataset", action="append", required=True, help="待核对 JSONL；可重复。")
    parser.add_argument("--expected-family", choices=("qwen2_5_vl", "qwen3_vl"))
    args = parser.parse_args()
    try:
        families = {
            validate_model_dataset_family(
                args.model,
                dataset,
                expected_family=args.expected_family,
            )
            for dataset in args.dataset
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    family = families.pop()
    print(f"[CognitiveTrack] 模型/训练视图匹配：{family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
