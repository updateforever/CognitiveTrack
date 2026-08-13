import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogtrack.context import REFERENCE_MODE_VISUAL_BOX, VISUAL_MARKER_VERSION
from cogtrack.training.swift_dataset import to_ms_swift_record, validate_ms_swift_record
from cogtrack.training.tracking_samples import (
    MEMORY_SUPERVISION_EXPLICIT,
    MEMORY_SUPERVISION_FEASIBILITY_NULL,
    MemoryUpdateLabel,
    TrackingSampleConfig,
    build_tracking_samples,
    load_memory_labels_jsonl,
)
from cogtrack.vlm import parse_tracking_output
from pytracking.evaluation.data import Sequence


def _write_frame(path: Path, value: int, *, width: int = 100, height: int = 50) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _make_sequence(
    root: Path,
    boxes: list[list[float]],
    visible: list[bool] | None,
    *,
    name: str = "sequence-1",
) -> Sequence:
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_id in range(len(boxes)):
        path = root / f"{frame_id:04d}.png"
        _write_frame(path, 30 + frame_id * 10)
        frames.append(str(path))
    return Sequence(
        name=name,
        frames=frames,
        dataset="synthetic",
        ground_truth_rect=boxes,
        target_visible=visible,
        language_query="small gray target",
        keyframe_indices=range(1, len(boxes)),
    )


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pair_samples_handle_present_absent_and_invalid_bbox(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [
            [10, 10, 20, 10],
            [20, 10, 20, 10],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [True, True, False, True],
    )
    output = tmp_path / "pair"
    report = build_tracking_samples([sequence], output)

    assert report.sample_count == 2
    assert report.present_count == 1
    assert report.absent_count == 1
    assert report.skipped_invalid_bbox == 1
    rows = _read_rows(output / "source_samples.jsonl")

    present, absent = rows
    assert present["metadata"]["frame_id"] == 1
    assert present["assistant"]["target_status"] == "present"
    assert present["assistant"]["bbox_norm1000_xyxy"] == pytest.approx([200, 200, 400, 400])
    parse_tracking_output(
        json.dumps(present["assistant"]),
        image_width=100,
        image_height=50,
        require_memory_update=False,
    )

    assert absent["metadata"]["frame_id"] == 2
    assert absent["assistant"]["target_status"] == "absent"
    assert absent["assistant"]["bbox_norm1000_xyxy"] is None
    parse_tracking_output(
        json.dumps(absent["assistant"]),
        image_width=100,
        image_height=50,
        require_memory_update=False,
    )

    # 初始化帧保持原始像素，不通过绿色框或局部裁剪修改视觉内容。
    original = cv2.imread(sequence.frames[0])
    assert original is not None and np.all(original == 30)
    reference = cv2.imread(str(output / present["images"][0]))
    assert reference is not None
    assert np.max(np.abs(reference.astype(np.int16) - original.astype(np.int16))) <= 1
    assert "unmodified full earlier reference frame" in present["user_prompt"]
    assert "normalized 0-to-1000 xyxy coordinates" in present["user_prompt"]
    assert present["metadata"]["reference_bbox_norm1000_xyxy"] == pytest.approx(
        [100, 200, 300, 400]
    )

    for row in rows:
        record = to_ms_swift_record(row)
        assert not validate_ms_swift_record(record, image_root=output)


def test_seeded_cap_is_reproducible(tmp_path: Path) -> None:
    boxes = [[5 + index, 5, 20, 10] for index in range(10)]
    sequence = _make_sequence(tmp_path / "source", boxes, [True] * len(boxes))
    config = TrackingSampleConfig(max_samples_per_sequence=3, seed=17)

    build_tracking_samples([sequence], tmp_path / "first", config=config)
    build_tracking_samples([sequence], tmp_path / "second", config=config)
    first = _read_rows(tmp_path / "first/source_samples.jsonl")
    second = _read_rows(tmp_path / "second/source_samples.jsonl")

    assert len(first) == 3
    assert first == second
    assert [row["metadata"]["frame_id"] for row in first] == sorted(
        row["metadata"]["frame_id"] for row in first
    )


def test_presence_balancing_prefers_equal_states_and_is_reproducible(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[5, 5, 20, 10]] * 10,
        [True] * 8 + [False] * 2,
    )
    config = TrackingSampleConfig(
        max_samples_per_sequence=4,
        balance_presence=True,
        seed=23,
    )

    first_report = build_tracking_samples([sequence], tmp_path / "first", config=config)
    build_tracking_samples([sequence], tmp_path / "second", config=config)

    assert first_report.present_count == 2
    assert first_report.absent_count == 2
    assert first_report.balance_presence is True
    assert _read_rows(tmp_path / "first/source_samples.jsonl") == _read_rows(
        tmp_path / "second/source_samples.jsonl"
    )


def test_mosaic_uses_only_past_positive_history_and_falls_back_to_pair(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10]] * 4,
        [True, True, False, True],
    )
    output = tmp_path / "mosaic"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(mode="mosaic", history_size=3),
    )
    rows = _read_rows(output / "source_samples.jsonl")

    assert report.pair_count == 1
    assert report.mosaic_count == 2
    assert rows[0]["metadata"]["effective_mode"] == "pair"
    assert rows[0]["metadata"]["history_frame_ids"] == []
    assert len(rows[0]["images"]) == 2

    # 帧 3 的历史只能来自过去的正帧 1；缺席帧 2 不能进入 mosaic。
    last = rows[-1]
    assert last["metadata"]["requested_mode"] == "mosaic"
    assert last["metadata"]["effective_mode"] == "mosaic"
    assert last["metadata"]["history_frame_ids"] == [1]
    assert len(last["images"]) == 3
    assert (output / last["images"][1]).is_file()
    history = cv2.imread(str(output / last["images"][1]))
    assert history is not None
    # 视觉输入不添加包含绝对 frame_id 的 30px header。
    assert history.shape[:2] == (240, 480)


def test_mosaic_corruption_is_explicit_and_keeps_current_answer(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10]] * 5,
        [True] * 5,
    )
    output = tmp_path / "corrupt"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="mosaic",
            history_size=3,
            history_corruption_ratio=1.0,
        ),
    )
    rows = _read_rows(output / "source_samples.jsonl")
    corrupt = [row for row in rows if row["metadata"].get("history_corruption")]
    assert report.corrupted_mosaic_count == len(corrupt)
    assert corrupt
    assert all(row["assistant"]["target_status"] == "present" for row in corrupt)
    assert all("::" + row["metadata"]["history_corruption"] in row["id"] for row in corrupt)


def test_stage1_both_is_present_only_visual_and_reuses_current_assets(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [
            [10, 10, 20, 10],
            [12, 10, 20, 10],
            [0, 0, 0, 0],
            [16, 10, 20, 10],
        ],
        [True, True, False, True],
    )
    output = tmp_path / "stage1"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="both",
            present_only=True,
            use_language_description=False,
            max_image_side=40,
        ),
    )
    rows = _read_rows(output / "source_samples.jsonl")

    # 帧 1 尚无历史，只生成 pair；帧 3 同时生成 pair 和 mosaic。
    assert report.sample_count == 3
    assert report.present_count == 3
    assert report.absent_count == 0
    assert report.pair_count == 2
    assert report.mosaic_count == 1
    assert {row["metadata"]["frame_id"] for row in rows} == {1, 3}
    assert all(row["target_status"] == "present" for row in rows)
    assert all(row["metadata"]["used_language_description"] is False for row in rows)
    assert all("small gray target" not in row["user_prompt"] for row in rows)

    current_assets = sorted((output / "images/synthetic/sequence-1").glob("current_*.jpg"))
    assert len(current_assets) == 2
    for path in current_assets:
        image = cv2.imread(str(path))
        assert image is not None
        assert max(image.shape[:2]) <= 40


def test_planned_late_initialization_anchor_is_recorded_and_used(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [10, 10, 20, 10],
            [12, 10, 20, 10],
            [0, 0, 0, 0],
        ],
        [False, False, True, True, False],
        name="late-init",
    )
    output = tmp_path / "planned"
    key = "synthetic::late-init"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(use_language_description=False),
        frame_ids_by_sequence={key: (3, 4)},
        anchor_frame_ids_by_sequence={key: 2},
    )
    rows = _read_rows(output / "source_samples.jsonl")

    assert report.sample_count == 2
    assert [row["metadata"]["reference_frame_id"] for row in rows] == [2, 2]
    assert rows[0]["images"][0].endswith("reference_00000002.jpg")
    assert [row["target_status"] for row in rows] == ["present", "absent"]


def test_same_current_with_distinct_references_builds_unique_pairs(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [11, 10, 20, 10], [12, 10, 20, 10], [13, 10, 20, 10]],
        [True, True, True, True],
        name="multi-reference",
    )
    key = "synthetic::multi-reference"
    output = tmp_path / "multi-reference"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(use_language_description=False),
        frame_ids_by_sequence={key: (3, 3)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (0, 1)},
    )
    rows = _read_rows(output / "source_samples.jsonl")

    assert report.sample_count == 2
    assert len({row["id"] for row in rows}) == 2
    assert [row["metadata"]["frame_id"] for row in rows] == [3, 3]
    assert [row["metadata"]["reference_frame_id"] for row in rows] == [0, 1]
    assert len(list((output / "images/synthetic/multi-reference").glob("current_*.jpg"))) == 1


def test_v5_visual_reference_and_three_field_feasibility_data(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [20, 10, 20, 10], [0, 0, 0, 0]],
        [True, True, False],
    )
    output = tmp_path / "visual-v5"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="pair",
            use_language_description=False,
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_FEASIBILITY_NULL,
        ),
    )
    rows = _read_rows(output / "source_samples.jsonl")

    assert report.reference_mode == REFERENCE_MODE_VISUAL_BOX
    assert report.visual_marker_version == VISUAL_MARKER_VERSION
    assert report.memory_null_count == 2
    assert report.memory_non_null_count == 0
    assert all(set(row["assistant"]) == {"target_status", "bbox_norm1000_xyxy", "memory_update"} for row in rows)
    assert all(row["assistant"]["memory_update"] is None for row in rows)
    assert all("normalized 0-to-1000 xyxy coordinates" not in row["user_prompt"] for row in rows)
    assert all(row["metadata"]["reference_mode"] == REFERENCE_MODE_VISUAL_BOX for row in rows)
    assert all("not_for_formal_training" in row["metadata"]["memory_label_source"] for row in rows)

    reference = cv2.imread(str(output / rows[0]["images"][0]))
    current = cv2.imread(str(output / rows[0]["images"][-1]))
    assert reference is not None and current is not None
    # cv2 读取为 BGR：视觉指代红框应使 R 通道显著高于 B/G；current 仍是灰度图。
    assert int(reference[:, :, 2].max()) > int(reference[:, :, :2].max()) + 80
    assert int(np.max(np.ptp(current.astype(np.int16), axis=2))) <= 2


def test_visual_reference_rejects_two_field_training_protocol() -> None:
    with pytest.raises(ValueError, match="必须启用三字段"):
        TrackingSampleConfig(reference_mode=REFERENCE_MODE_VISUAL_BOX)


def test_explicit_memory_labels_are_required_and_audited(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [20, 10, 20, 10]],
        [True, True],
    )
    config = TrackingSampleConfig(
        reference_mode=REFERENCE_MODE_VISUAL_BOX,
        memory_supervision=MEMORY_SUPERVISION_EXPLICIT,
    )
    with pytest.raises(ValueError, match="必须提供 memory_labels_by_sequence"):
        build_tracking_samples([sequence], tmp_path / "missing", config=config)

    labels = {
        "synthetic::sequence-1": {
            1: MemoryUpdateLabel(
                "Rear view reveals two stable white stripes.",
                source="manual_double_review_v1",
                reviewed=True,
            )
        }
    }
    output = tmp_path / "explicit"
    report = build_tracking_samples(
        [sequence],
        output,
        config=config,
        memory_labels_by_sequence=labels,
    )
    row = _read_rows(output / "source_samples.jsonl")[0]

    assert report.memory_non_null_count == 1
    assert row["assistant"]["memory_update"] == "Rear view reveals two stable white stripes."
    assert row["metadata"]["memory_label_source"] == "manual_double_review_v1"
    assert row["metadata"]["memory_label_reviewed"] is True


def test_memory_label_jsonl_loader_rejects_duplicates_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "memory.jsonl"
    label_path.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "sequence": "sequence-1",
                "frame_id": 7,
                "memory_update": "A stable white stripe is now visible.",
                "source": "manual_review_v1",
                "reviewed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = load_memory_labels_jsonl(label_path)

    label = labels["synthetic::sequence-1"][7]
    assert label.value == "A stable white stripe is now visible."
    assert label.source == "manual_review_v1"
    assert label.reviewed is True

    with label_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "dataset": "synthetic",
                    "sequence": "sequence-1",
                    "frame_id": 7,
                    "memory_update": None,
                    "source": "hard_null_v1",
                }
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="重复 memory label"):
        load_memory_labels_jsonl(label_path)
