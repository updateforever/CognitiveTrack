#!/usr/bin/env python3
"""将已合成数据打成 Hugging Face/ModelScope 友好的图片 TAR 分卷。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training import package_dataset_release  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="包含 images/ 与 ms_swift/ 的合成数据根目录。")
    parser.add_argument("--release-dir", required=True, help="待上传的数据发布目录；必须位于源目录之外。")
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=2.0,
        help="单个图片 TAR 的目标上限；不会拆分单个序列，默认 2GB。",
    )
    parser.add_argument("--force", action="store_true", help="覆盖同名文件；不会删除额外文件。")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.max_shard_gb <= 0:
            raise ValueError("--max-shard-gb 必须大于 0")
        report = package_dataset_release(
            args.source_dir,
            args.release_dir,
            max_shard_bytes=int(args.max_shard_gb * 1024**3),
            overwrite=args.force,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print("[CognitiveTrack] 可迁移数据发布包构建完成")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"[上传目录] {Path(args.release_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
