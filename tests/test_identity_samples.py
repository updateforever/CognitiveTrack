import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogtrack.protocol import ModelOutputParseError
from cogtrack.training.identity_samples import IdentitySampleConfig, build_identity_samples
from cogtrack.training.swift_dataset import (
    split_records_by_sequence,
    to_ms_swift_record,
    validate_ms_swift_record,
)
from cogtrack.vlm import parse_tracking_output
from pytracking.evaluation.data import Sequence


def _make_sequence(
    root: Path,
    *,
    name: str,
    object_class: str | None,
    dataset: str = "synthetic",
    metadata: dict | None = None,
    visible: list[bool] | None = None,
) -> Sequence:
    sequence_root = root / dataset / name
    sequence_root.mkdir(parents=True, exist_ok=True)
    frames = []
    boxes = []
    for frame_id in range(4):
        image = np.full((50, 100, 3), 30 + 10 * frame_id, dtype=np.uint8)
        image[:, :5, :] = (frame_id * 20, 0, 0)
        frame_path = sequence_root / f"{frame_id:04d}.png"
        assert cv2.imwrite(str(frame_path), image)
        frames.append(str(frame_path))
        boxes.append([10 + frame_id * 5, 10, 20, 10])
    return Sequence(
        name=name,
        frames=frames,
        dataset=dataset,
        ground_truth_rect=boxes,
        target_visible=visible or [True] * len(frames),
        object_class=object_class,
        language_query=f"the initialized {object_class}" if object_class else None,
        keyframe_indices=(1, 3),
        metadata=metadata,
    )


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_traceable_same_class_cross_sequence_negatives(tmp_path: Path) -> None:
    first = _make_sequence(tmp_path / "source", name="car-a", object_class="Car")
    second = _make_sequence(tmp_path / "source", name="car-b", object_class="car")
    other_class = _make_sequence(tmp_path / "source", name="dog-a", object_class="dog")
    missing_class = _make_sequence(tmp_path / "source", name="unknown", object_class=None)

    output = tmp_path / "identity"
    report = build_identity_samples(
        [first, second, other_class, missing_class],
        output,
        config=IdentitySampleConfig(max_candidate_frames=2, seed=7),
    )
    rows = _read_rows(output / "identity_source_samples.jsonl")

    assert report.sample_count == 4
    assert report.pair_group_count == 1
    assert report.missing_class_count == 1
    assert report.unpaired_instance_count == 1
    assert {row["metadata"]["split_group"] for row in rows} == {
        rows[0]["metadata"]["split_group"]
    }

    for row in rows:
        answer = row["assistant"]
        metadata = row["metadata"]
        assert answer["target_presence"] == "present"
        assert answer["identity_match"] == "different"
        assert answer["localizability"] == "localizable"
        assert answer["bbox_norm1000_xyxy"] is not None
        assert metadata["label_source"] == "cross_sequence_same_class"
        assert metadata["reference_sequence"] != metadata["candidate_sequence"]
        assert metadata["reference_instance_group"] != metadata["candidate_instance_group"]
        assert metadata["source_sequence"] == metadata["split_group"]
        assert len(row["images"]) == 2
        # identity 辅助数据与 v3 presence 主协议严格隔离，不能误混入当前训练集。
        with pytest.raises(ModelOutputParseError):
            parse_tracking_output(json.dumps(answer), image_width=100, image_height=50)

        record = to_ms_swift_record(row)
        issues = validate_ms_swift_record(record, image_root=output)
        assert any(issue.code == "answer.protocol" for issue in issues)

    # 第二张图的候选框为红色；框只提供候选位置，不代表 same 标签。
    candidate_image = cv2.imread(str(output / rows[0]["images"][1]))
    assert candidate_image is not None
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(candidate_image))
    assert np.any((red > 180) & (red > green + 60) & (red > blue + 60))


def test_pairing_and_instance_groups_are_deterministic_and_split_safe(tmp_path: Path) -> None:
    sequences = [
        _make_sequence(tmp_path / "source", name=f"car-{index}", object_class="car")
        for index in range(4)
    ]
    config = IdentitySampleConfig(max_candidate_frames=1, seed=123)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    build_identity_samples(sequences, first_output, config=config)
    build_identity_samples(sequences, second_output, config=config)
    first_rows = _read_rows(first_output / "identity_source_samples.jsonl")
    second_rows = _read_rows(second_output / "identity_source_samples.jsonl")

    assert first_rows == second_rows
    instance_groups: dict[str, set[str]] = {}
    for row in first_rows:
        metadata = row["metadata"]
        split_group = metadata["split_group"]
        for key in ("reference_instance_group", "candidate_instance_group"):
            instance_groups.setdefault(metadata[key], set()).add(split_group)
    assert len(instance_groups) == 4
    assert all(len(groups) == 1 for groups in instance_groups.values())

    records = [to_ms_swift_record(row) for row in first_rows]
    splits = split_records_by_sequence(records, val_ratio=0.5, seed=9)
    instance_splits: dict[str, set[str]] = {}
    for split, split_rows in splits.items():
        for row in split_rows:
            metadata = row["metadata"]
            for key in ("reference_instance_group", "candidate_instance_group"):
                instance_splits.setdefault(metadata[key], set()).add(split)
    assert all(len(values) == 1 for values in instance_splits.values())


def test_missing_classes_are_not_guessed(tmp_path: Path) -> None:
    missing = _make_sequence(tmp_path / "source", name="unknown", object_class=None)
    lone = _make_sequence(tmp_path / "source", name="car-a", object_class="car")

    with pytest.raises(ValueError, match="missing_class=1"):
        build_identity_samples([missing, lone], tmp_path / "output")
    assert not (tmp_path / "output/identity_source_samples.jsonl").exists()


def test_same_physical_source_aliases_are_never_labeled_different(tmp_path: Path) -> None:
    alias = _make_sequence(
        tmp_path / "source",
        name="wrapped-a",
        object_class="car",
        dataset="benchmark",
        metadata={"source_dataset": "raw", "source_sequence": "car-a"},
    )
    original = _make_sequence(
        tmp_path / "source",
        name="car-a",
        object_class="car",
        dataset="raw",
    )
    different = _make_sequence(
        tmp_path / "source",
        name="car-b",
        object_class="car",
        dataset="raw",
    )

    output = tmp_path / "identity"
    report = build_identity_samples(
        [alias, original, different],
        output,
        config=IdentitySampleConfig(max_candidate_frames=1),
    )
    rows = _read_rows(output / "identity_source_samples.jsonl")

    assert report.duplicate_alias_count == 1
    assert report.eligible_instance_count == 2
    assert rows
    assert all(
        row["metadata"]["reference_instance_group"]
        != row["metadata"]["candidate_instance_group"]
        for row in rows
    )
