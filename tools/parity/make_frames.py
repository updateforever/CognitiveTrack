#!/usr/bin/env python3
"""生成 parity 用的固定帧清单与初始框，两个实现共用同一份输入。

两侧脚本都从这份 JSON 读取帧路径和初始框，绕开各自的数据集加载器，
使对比只反映网络与预处理的差异。

用法：

    PARITY_SEQ_DIR=/data2/DATASETS_PUBLIC/MGIT/data/val/029/frame_029 \
    PARITY_GT=/data2/DATASETS_PUBLIC/MGIT/attribute/groundtruth/029.txt \
    python tools/parity/make_frames.py 60
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SEQ_DIR = Path(
    os.environ.get("PARITY_SEQ_DIR", "/data2/DATASETS_PUBLIC/MGIT/data/val/029/frame_029")
)
GT = Path(os.environ.get("PARITY_GT", "/data2/DATASETS_PUBLIC/MGIT/attribute/groundtruth/029.txt"))
OUT = Path(os.environ.get("PARITY_FRAMES", "/tmp/parity/frames.json"))
SEQUENCE = os.environ.get("PARITY_SEQUENCE", SEQ_DIR.parent.name)
# 原版按 dataset_name 大写后查 cfg.TEST.*，需要与被对比的数据集口径一致。
DATASET_NAME = os.environ.get("PARITY_DATASET", "videocube")


def main() -> int:
    if not SEQ_DIR.is_dir():
        print(f"缺少序列目录: {SEQ_DIR}", file=sys.stderr)
        return 2
    if not GT.is_file():
        print(f"缺少 groundtruth: {GT}", file=sys.stderr)
        return 2

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    frames = sorted(SEQ_DIR.glob("*.jpg"))[:limit]
    if not frames:
        print(f"序列目录没有 jpg: {SEQ_DIR}", file=sys.stderr)
        return 2

    first = GT.read_text(encoding="utf-8").splitlines()[0].strip()
    sep = "," if "," in first else None
    init_bbox = [float(v) for v in (first.split(sep) if sep else first.split())]
    if len(init_bbox) != 4:
        print(f"初始框解析失败: {first!r}", file=sys.stderr)
        return 2

    payload = {
        "sequence": SEQUENCE,
        "dataset_name": DATASET_NAME,
        "init_bbox": init_bbox,
        "frames": [str(p) for p in frames],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"写出 {OUT}: {len(frames)} 帧, init_bbox={init_bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
