#!/usr/bin/env python3
"""离线校验 CognitiveBench v1 标注结构并输出统计摘要。

该工具只读取约 35MB 的 benchmark 标注，不访问 LaSOT/TNL2K/MGIT 图片。它用于 Git
clone 后快速确认 995 条序列、逐帧状态、关键帧范围和 meta 来源映射均完整。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pytracking.utils.io import load_numeric_table, read_index_file

EXPECTED = {
    "sequence_count": 995,
    "frame_count": 1_408_438,
    "keyframe_count": 343_616,
    "present_count": 1_274_185,
    "absent_count": 134_253,
    "invalid_present_bbox_count": 2,
    "sequence_counts_by_source": {"lasot": 280, "mgit": 15, "tnl2k": 700},
    "annotation_sha256": "fc10d30be2042b9e227c608c19b77772de375b14eddc1134a50e5acfb7fa5a0e",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须为 object：{path}")
    return value


def verify_cognitivebench(root: str | Path, *, split: str = "test") -> dict[str, Any]:
    benchmark_root = Path(root).expanduser().resolve()
    benchmark_meta = _read_json(benchmark_root / "benchmark_meta.json")
    if benchmark_meta.get("version") != "v1":
        raise ValueError(f"只支持 CognitiveBench v1，收到 {benchmark_meta.get('version')!r}")
    if benchmark_meta.get("bbox_format") != "xywh" or benchmark_meta.get("frame_index_base") != 0:
        raise ValueError("benchmark_meta 必须声明 xywh 和 0-based 帧索引")

    split_root = benchmark_root / split
    sequence_dirs = sorted(
        (path for path in split_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.encode("utf-8"),
    )
    source_counts: Counter[str] = Counter()
    frames = present = absent = keyframes = invalid_present_boxes = 0
    aggregate = hashlib.sha256()

    for sequence_dir in sequence_dirs:
        required = (
            sequence_dir / "groundtruth.txt",
            sequence_dir / "target_status.txt",
            sequence_dir / "keyframes.txt",
            sequence_dir / "meta.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{sequence_dir.name} 缺少标注文件：{missing}")

        meta = _read_json(required[3])
        source = str(meta.get("source_dataset", "")).lower()
        if source not in {"lasot", "tnl2k", "mgit"}:
            raise ValueError(f"{sequence_dir.name}: 非法 source_dataset={source!r}")
        if int(meta.get("frame_index_base", -1)) != 0 or meta.get("bbox_format") != "xywh":
            raise ValueError(f"{sequence_dir.name}: meta 坐标/帧索引协议错误")

        gt = load_numeric_table(required[0], columns=4)
        status = load_numeric_table(required[1], dtype=np.int64).reshape(-1)
        indices = read_index_file(required[2])
        if len(gt) != len(status) or int(meta.get("num_frames", -1)) != len(gt):
            raise ValueError(f"{sequence_dir.name}: GT/status/meta 帧数不一致")
        if not len(status) or int(status[0]) != 1:
            raise ValueError(f"{sequence_dir.name}: 首帧必须 present")
        if set(status.tolist()) - {0, 1}:
            raise ValueError(f"{sequence_dir.name}: target_status 只能为 0/1")
        if len(indices) != len(set(indices)):
            raise ValueError(f"{sequence_dir.name}: keyframes 存在重复索引")
        if any(index < 0 or index >= len(gt) for index in indices):
            raise ValueError(f"{sequence_dir.name}: keyframes 越界")

        valid_bbox = np.all(np.isfinite(gt), axis=1) & np.all(gt[:, 2:] > 0, axis=1)
        invalid_present_boxes += int(np.count_nonzero((status == 1) & ~valid_bbox))
        source_counts[source] += 1
        frames += len(gt)
        present += int(np.count_nonzero(status == 1))
        absent += int(np.count_nonzero(status == 0))
        keyframes += len(indices)

        # 指纹基于文件相对路径和原始字节，能发现 Git/LFS/传输造成的任意标注变化。
        for path in required:
            relative = path.relative_to(benchmark_root).as_posix()
            file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            aggregate.update(f"{file_digest}  {relative}\n".encode("utf-8"))

    report = {
        "schema_version": "cogtrack.cognitivebench_verification.v1",
        "benchmark_version": benchmark_meta["version"],
        "split": split,
        "sequence_count": len(sequence_dirs),
        "frame_count": frames,
        "keyframe_count": keyframes,
        "present_count": present,
        "absent_count": absent,
        "absent_ratio": absent / frames,
        "invalid_present_bbox_count": invalid_present_boxes,
        "sequence_counts_by_source": dict(sorted(source_counts.items())),
        "annotation_sha256": aggregate.hexdigest(),
    }
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in EXPECTED.items()
        if report.get(key) != value
    }
    report["expected"] = EXPECTED
    report["mismatches"] = mismatches
    report["ok"] = not mismatches
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="benchmarks/cognitivebench/v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", help="可选：写出 JSON 校验报告")
    args = parser.parse_args()

    try:
        report = verify_cognitivebench(args.root, split=args.split)
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
