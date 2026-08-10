"""debug 截断结果必须默认排除在指标之外。

``--debug-frames N`` 只跑前 N 帧，但主指标按序列宏平均，截断序列会和完整
序列等权。实测一条 20 帧的 029 混进 3 条完整序列时，AUC 从 62.06 被抬到
69.52，因此这里锁住"默认跳过"的行为。
"""

from __future__ import annotations

import json
from pathlib import Path

from cogtrack.evaluation import is_debug_limited, partition_debug_limited


def _write_result(directory: Path, name: str, *, debug_limited: bool | None) -> Path:
    """写出一份最小的 frames.jsonl + manifest 组合。"""

    directory.mkdir(parents=True, exist_ok=True)
    frames_path = directory / f"{name}_frames.jsonl"
    record = {
        "sequence_name": name,
        "frame_num": 0,
        "ground_truth_rect": [1, 2, 10, 10],
        "target_bbox": [1, 2, 10, 10],
    }
    frames_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    if debug_limited is not None:
        manifest = {
            "sequence": name,
            "expected_frames": 1,
            "extra": {"debug_limited": debug_limited, "full_sequence_frames": 100},
        }
        (directory / f"{name}_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return frames_path


def test_debug_limited_manifest_is_detected(tmp_path: Path):
    truncated = _write_result(tmp_path, "truncated", debug_limited=True)
    complete = _write_result(tmp_path, "complete", debug_limited=False)

    assert is_debug_limited(truncated) is True
    assert is_debug_limited(complete) is False


def test_missing_manifest_is_treated_as_complete(tmp_path: Path):
    # manifest 缺失时不能误删正常结果，只能保守当成完整序列。
    orphan = _write_result(tmp_path, "orphan", debug_limited=None)
    assert is_debug_limited(orphan) is False


def test_corrupt_manifest_is_treated_as_complete(tmp_path: Path):
    frames_path = _write_result(tmp_path, "corrupt", debug_limited=None)
    (tmp_path / "corrupt_manifest.json").write_text("{ not json", encoding="utf-8")
    assert is_debug_limited(frames_path) is False


def test_partition_splits_debug_from_complete(tmp_path: Path):
    truncated = _write_result(tmp_path, "truncated", debug_limited=True)
    complete = _write_result(tmp_path, "complete", debug_limited=False)
    orphan = _write_result(tmp_path, "orphan", debug_limited=None)

    full, debug = partition_debug_limited([truncated, complete, orphan])

    assert debug == [truncated]
    assert sorted(full) == sorted([complete, orphan])
