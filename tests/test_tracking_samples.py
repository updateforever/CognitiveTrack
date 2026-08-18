import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogtrack.context import (
    HISTORY_LAYOUT_RECENT_STRIP_3_V2,
    PROMPT_PROFILE_VLT_V6,
    REFERENCE_MODE_VISUAL_BOX,
    VISUAL_MARKER_VERSION,
)
from cogtrack.training.swift_dataset import to_ms_swift_record, validate_ms_swift_record
from cogtrack.training.tracking_samples import (
    HISTORY_COMPLETENESS_H0,
    HISTORY_COMPLETENESS_H1,
    HISTORY_COMPLETENESS_H2,
    HISTORY_COMPLETENESS_H3,
    HISTORY_QUALITY_JITTER,
    HISTORY_QUALITY_STALE,
    MEMORY_SUPERVISION_EXPLICIT,
    MEMORY_SUPERVISION_FEASIBILITY_NULL,
    MEMORY_SUPERVISION_MASKED_NULL,
    MEMORY_SUPERVISION_THREE_STATE,
    VALID_VISUAL_COMBINATIONS,
    MemoryUpdateLabel,
    TrackingSampleConfig,
    _corrupt_history_panels,
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
    assert present["assistant"]["status"] == "present"
    assert present["assistant"]["bbox_2d"] == pytest.approx([200, 200, 400, 400])
    parse_tracking_output(
        json.dumps(present["assistant"]),
        image_width=100,
        image_height=50,
        require_memory_update=False,
    )

    assert absent["metadata"]["frame_id"] == 2
    assert absent["assistant"]["status"] == "absent"
    assert absent["assistant"]["bbox_2d"] is None
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
    assert all(row["assistant"]["status"] == "present" for row in corrupt)
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
    assert all(
        set(row["assistant"]) == {"bbox_2d", "status", "memory_update"} for row in rows
    )
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


def test_vlt_v6_tracking_sft_uses_three_images_and_masked_unknown_updates(
    tmp_path: Path,
) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [20, 10, 20, 10], [0, 0, 0, 0]],
        [True, True, False],
    )
    output = tmp_path / "vlt-v6"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="mosaic",
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            prompt_profile=PROMPT_PROFILE_VLT_V6,
            force_history_image=True,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
            use_language_description=True,
        ),
    )
    rows = _read_rows(output / "source_samples.jsonl")

    assert report.schema_version == "cogtrack.training.source.v7"
    assert report.prompt_profile == PROMPT_PROFILE_VLT_V6
    assert report.force_history_image is True
    assert report.history_layout_version == HISTORY_LAYOUT_RECENT_STRIP_3_V2
    assert report.memory_null_count == 2
    assert report.memory_non_null_count == 0
    assert report.semantic_memory_input_count == 0
    assert all(len(row["images"]) == 3 for row in rows)
    assert all(row["metadata"]["effective_mode"] == "mosaic" for row in rows)
    assert rows[0]["metadata"]["history_anchor_fallback"] is True
    assert rows[1]["metadata"]["history_anchor_fallback"] is False
    assert rows[0]["metadata"]["history_frame_ids"] == [0, 0, 0]
    assert rows[1]["metadata"]["history_frame_ids"] == [1, 1, 1]
    assert all(
        row["metadata"]["history_layout_version"]
        == HISTORY_LAYOUT_RECENT_STRIP_3_V2
        for row in rows
    )
    history_images = [cv2.imread(str(output / row["images"][1])) for row in rows]
    assert all(image is not None and image.shape[0] == 240 for image in history_images)
    assert all(image is not None and image.shape[1] == 1454 for image in history_images)
    assert all(row["assistant"]["memory_update"] is None for row in rows)
    # tracking_sft 不提供状态语义：present/absent 的 null 都是 masked placeholder。
    present_rows = [r for r in rows if r["target_status"] == "present"]
    absent_rows = [r for r in rows if r["target_status"] == "absent"]
    assert all(r["metadata"]["memory_supervision_state"] == "masked_unknown" for r in present_rows)
    assert all(r["metadata"]["memory_loss_masked"] is True for r in present_rows)
    assert all(r["metadata"]["memory_supervision_state"] == "masked_unknown" for r in absent_rows)
    assert all(r["metadata"]["memory_loss_masked"] is True for r in absent_rows)
    assert all(
        row["metadata"]["sft_supervision_profile"] == "tracking_sft" for row in rows
    )
    assert all("small gray target" in row["user_prompt"] for row in rows)
    assert all("Initial target identity: small gray target" in row["user_prompt"] for row in rows)
    assert all(
        "Current maintained target state: small gray target" in row["user_prompt"]
        for row in rows
    )
    assert all(
        row["metadata"]["current_target_state"] == "small gray target" for row in rows
    )
    assert all("Decision order" not in row["user_prompt"] for row in rows)


def test_tracking_taxonomy_classifies_events_and_h0_to_h3(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10 + frame_id, 10, 20, 10] for frame_id in range(6)],
        [True, True, False, True, True, True],
        name="taxonomy",
    )
    key = "synthetic::taxonomy"
    output = tmp_path / "taxonomy"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="mosaic",
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            prompt_profile=PROMPT_PROFILE_VLT_V6,
            force_history_image=True,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
            history_corruption_ratio=0.0,
        ),
        frame_ids_by_sequence={key: (1, 2, 3, 4, 5)},
        anchor_frame_ids_by_sequence={key: 0},
    )
    rows = _read_rows(output / "source_samples.jsonl")
    by_frame = {row["metadata"]["frame_id"]: row["metadata"] for row in rows}

    assert by_frame[1]["history_completeness"] == HISTORY_COMPLETENESS_H0
    assert by_frame[2]["history_completeness"] == HISTORY_COMPLETENESS_H1
    assert by_frame[3]["history_completeness"] == HISTORY_COMPLETENESS_H1
    assert by_frame[4]["history_completeness"] == HISTORY_COMPLETENESS_H2
    assert by_frame[5]["history_completeness"] == HISTORY_COMPLETENESS_H3
    assert by_frame[1]["temporal_event"] == "continuous_present"
    assert by_frame[2]["temporal_event"] == "absent"
    assert by_frame[2]["absent_phase"] == "single"
    assert by_frame[3]["temporal_event"] == "reappearance"
    # Image 2 已经包含 frame 3 的重现观测，frame 4 不应继续被算作 reappearance。
    assert by_frame[4]["temporal_event"] == "continuous_present"

    assert report.temporal_event_counts == {
        "absent": 1,
        "continuous_present": 3,
        "reappearance": 1,
    }
    assert report.history_completeness_counts == {
        HISTORY_COMPLETENESS_H0: 1,
        HISTORY_COMPLETENESS_H1: 2,
        HISTORY_COMPLETENESS_H2: 1,
        HISTORY_COMPLETENESS_H3: 1,
    }
    assert set(report.visual_combination_counts) == set(VALID_VISUAL_COMBINATIONS)
    assert len(report.visual_combination_counts) == 27
    assert sum(report.visual_combination_counts.values()) == report.sample_count


def test_absent_run_phase_is_start_middle_end(tmp_path: Path) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10]] * 5,
        [True, False, False, False, True],
        name="absent-run",
    )
    key = "synthetic::absent-run"
    output = tmp_path / "absent-run"
    report = build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="mosaic",
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            prompt_profile=PROMPT_PROFILE_VLT_V6,
            force_history_image=True,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
        ),
        frame_ids_by_sequence={key: (1, 2, 3, 4)},
        anchor_frame_ids_by_sequence={key: 0},
    )
    rows = _read_rows(output / "source_samples.jsonl")
    phases = {
        row["metadata"]["frame_id"]: row["metadata"]["absent_phase"] for row in rows
    }

    assert phases == {1: "start", 2: "middle", 3: "end", 4: None}
    assert report.absent_phase_counts == {
        "end": 1,
        "middle": 1,
        "single": 0,
        "start": 1,
    }


def test_history_corruption_obeys_completeness_constraints() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    padded_h1 = [(1, image, (10.0, 10.0, 20.0, 20.0))] * 3
    corrupted_h1, h1_mode = _corrupt_history_panels(padded_h1, seed_key="h1")

    assert h1_mode == HISTORY_QUALITY_JITTER
    assert (
        sum(
            before[2] != after[2]
            for before, after in zip(padded_h1, corrupted_h1, strict=True)
        )
        == 1
    )

    h3 = [
        (1, image, (5.0, 5.0, 20.0, 20.0)),
        (2, image, (35.0, 25.0, 18.0, 22.0)),
        (3, image, (60.0, 55.0, 15.0, 16.0)),
    ]
    stale = None
    for index in range(100):
        candidate, mode = _corrupt_history_panels(h3, seed_key=f"stale-{index}")
        if mode == HISTORY_QUALITY_STALE:
            stale = candidate
            break
    assert stale is not None
    assert (
        sum(before[2] != after[2] for before, after in zip(h3, stale, strict=True))
        == 1
    )


def test_h0_padding_uses_the_sampled_reference_not_sequence_anchor(
    tmp_path: Path,
) -> None:
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10]] * 4,
        [True] * 4,
        name="sampled-reference-h0",
    )
    key = "synthetic::sampled-reference-h0"
    output = tmp_path / "sampled-reference-h0"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            mode="mosaic",
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            prompt_profile=PROMPT_PROFILE_VLT_V6,
            force_history_image=True,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
        ),
        frame_ids_by_sequence={key: (3,)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (2,)},
    )
    metadata = _read_rows(output / "source_samples.jsonl")[0]["metadata"]

    assert metadata["history_frame_ids"] == [2, 2, 2]
    assert metadata["history_completeness"] == HISTORY_COMPLETENESS_H0
    assert metadata["reference_source"] == "sampled_prior_present"
    assert metadata["history_anchor_fallback"] is False
    assert metadata["history_reference_fallback"] is True


def test_three_state_mode_tolerates_partial_labels_and_absent_updates(tmp_path: Path) -> None:
    """three_state 的三条核心规则，一次性锁住。

    1. present + 标签 -> verified_update；
    2. present + 无标签 -> masked_unknown（占位 null，不参与 loss）；
    3. absent + 消失文本 -> verified_update。
    """
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [20, 10, 20, 10], [30, 10, 20, 10], [40, 10, 20, 10]],
        [True, True, True, False],
    )
    update_text = "The target now shows a bright red stripe."
    labels = {
        "synthetic::sequence-1": {
            # 帧 1 是 present，应成为 verified_update。
            1: MemoryUpdateLabel(update_text, source="mgit_action_segment_v1"),
            # 帧 3 不可见，显式消失描述应当完整监督。
            3: MemoryUpdateLabel(
                "The target has disappeared and is currently not visible.",
                source="dataset_gt_disappearance_transition_v1",
            ),
        }
    }
    output = tmp_path / "three_state"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_THREE_STATE,
        ),
        memory_labels_by_sequence=labels,
    )
    rows = _read_rows(output / "source_samples.jsonl")
    by_frame = {row["metadata"]["frame_id"]: row for row in rows}

    assert by_frame[1]["metadata"]["memory_supervision_state"] == "verified_update"
    assert by_frame[1]["assistant"]["memory_update"] == update_text
    assert by_frame[1]["metadata"]["memory_loss_masked"] is False

    # 帧 2 没有标签：不报错，退化为 masked_unknown。
    assert by_frame[2]["metadata"]["memory_supervision_state"] == "masked_unknown"
    assert by_frame[2]["assistant"]["memory_update"] is None
    assert by_frame[2]["metadata"]["memory_loss_masked"] is True
    assert by_frame[2]["metadata"]["memory_label_source"] == "three_state_unlabelled_frame_v1"

    absent_row = by_frame[3]
    assert absent_row["target_status"] == "absent"
    assert absent_row["metadata"]["memory_supervision_state"] == "verified_update"
    assert absent_row["assistant"]["memory_update"].startswith("The target has disappeared")
    assert absent_row["metadata"]["memory_loss_masked"] is False

    # 防泄漏：任何一行的输入侧都不得出现它自己的目标更新文本。
    for row in rows:
        target = row["assistant"].get("memory_update")
        if not target:
            continue
        metadata = row["metadata"]
        prompt_side = " || ".join(
            str(metadata.get(key) or "")
            for key in (
                "recent_semantic_memory",
                "current_target_state",
                "initial_identity_description",
            )
        )
        assert target not in prompt_side


def test_three_state_rejects_labels_without_the_matching_mode(tmp_path: Path) -> None:
    sequence = _make_sequence(tmp_path / "source", [[10, 10, 20, 10], [20, 10, 20, 10]], [True, True])
    with pytest.raises(ValueError, match="不接受 memory_labels_by_sequence"):
        build_tracking_samples(
            [sequence],
            tmp_path / "bad",
            config=TrackingSampleConfig(
                reference_mode=REFERENCE_MODE_VISUAL_BOX,
                memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
            ),
            memory_labels_by_sequence={"synthetic::sequence-1": {1: MemoryUpdateLabel(None, "x")}},
        )
    with pytest.raises(ValueError, match="必须提供 memory_labels_by_sequence"):
        build_tracking_samples(
            [sequence],
            tmp_path / "bad2",
            config=TrackingSampleConfig(
                reference_mode=REFERENCE_MODE_VISUAL_BOX,
                memory_supervision=MEMORY_SUPERVISION_THREE_STATE,
            ),
        )


def test_three_state_present_verified_null_becomes_supervised_hard_null(
    tmp_path: Path,
) -> None:
    """present + verified_null 标签 -> hard-null，且该声明写入 metadata。

    没有这一态，被监督的 present 行只有 verified_update，hard-null 与
    target_status=absent 完全共线，模型可以用“present 就输出文本”通过训练。
    """
    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [20, 10, 20, 10], [30, 10, 20, 10]],
        [True, True, True],
    )
    labels = {
        "synthetic::sequence-1": {
            1: MemoryUpdateLabel(
                None, source="mgit_action_segment_stable_v1", verified_null=True
            ),
        }
    }
    output = tmp_path / "verified_null"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_THREE_STATE,
        ),
        memory_labels_by_sequence=labels,
    )
    by_frame = {
        row["metadata"]["frame_id"]: row
        for row in _read_rows(output / "source_samples.jsonl")
    }

    claimed = by_frame[1]
    assert claimed["target_status"] == "present"
    assert claimed["assistant"]["memory_update"] is None
    assert claimed["metadata"]["memory_supervision_state"] == "verified_hard_null"
    assert claimed["metadata"]["memory_loss_masked"] is False
    assert claimed["metadata"]["memory_verified_null"] is True

    # 未声明的 present 帧不受影响，仍是占位 null。
    plain = by_frame[2]
    assert plain["metadata"]["memory_supervision_state"] == "masked_unknown"
    assert plain["metadata"]["memory_loss_masked"] is True
    assert plain["metadata"]["memory_verified_null"] is False


def test_memory_label_rejects_verified_null_with_text() -> None:
    with pytest.raises(ValueError, match="verified_null=True 的标签不能同时带更新文本"):
        MemoryUpdateLabel("some text", source="x", verified_null=True)


def _mgit_sequence(root: Path, *, frame_count: int, description_path: Path) -> Sequence:
    """带 action 标注路径的 MGIT 序列；语言域是 first_action_description。"""

    root.mkdir(parents=True, exist_ok=True)
    frames = []
    boxes = []
    for frame_id in range(frame_count):
        path = root / f"{frame_id:04d}.png"
        _write_frame(path, 30 + frame_id * 5)
        frames.append(str(path))
        boxes.append([10 + frame_id, 10, 20, 10])
    return Sequence(
        name="mgit-seq",
        frames=frames,
        dataset="mgit",
        ground_truth_rect=boxes,
        target_visible=[True] * frame_count,
        language_query="the man in a white shirt standing still",
        metadata={
            "split": "train",
            "language_scope": "first_action_description",
            "description_path": str(description_path),
        },
    )


def test_mgit_identity_text_reanchors_to_the_template_frame_segment(tmp_path: Path) -> None:
    """模板帧离开锚点后，身份文本必须来自覆盖该帧的 action 段。

    MGIT 的 language_query 是**序列开头**那段动作的文本。逐 case 随机模板会把 Image 1
    挪到中段，此时沿用开头文本等于给模型一张与描述不符的模板图——凭空造出的图文矛盾。
    """

    description_path = tmp_path / "mgit_desc.json"
    description_path.write_text(
        json.dumps(
            {
                "action": {
                    "1": {
                        "start_frame": 0,
                        "end_frame": 2,
                        "description": "the man in a white shirt standing still",
                    },
                    "2": {
                        "start_frame": 3,
                        "end_frame": 7,
                        "description": "The man in a white shirt riding a red bicycle",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    sequence = _mgit_sequence(
        tmp_path / "source", frame_count=8, description_path=description_path
    )
    key = "mgit::mgit-seq"
    output = tmp_path / "reanchored"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
        ),
        frame_ids_by_sequence={key: (7, 7)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (0, 4)},
    )
    rows = _read_rows(output / "source_samples.jsonl")
    by_reference = {row["metadata"]["reference_frame_id"]: row for row in rows}

    anchored = by_reference[0]["metadata"]
    assert anchored["initial_identity_description"] == (
        "the man in a white shirt standing still"
    )
    assert anchored["initial_identity_description_source"] == "dataset_initial_language"

    moved = by_reference[4]["metadata"]
    assert moved["initial_identity_description"] == (
        "The man in a white shirt riding a red bicycle"
    )
    assert moved["initial_identity_description_source"] == "mgit_reference_action_description"
    assert moved["used_language_description"] is False


def test_mgit_identity_text_falls_back_when_no_segment_covers_template(
    tmp_path: Path,
) -> None:
    """没有段覆盖模板帧时退回序列级文本，绝不编造描述。"""

    description_path = tmp_path / "gap_desc.json"
    description_path.write_text(
        json.dumps(
            {
                "action": {
                    "1": {
                        "start_frame": 0,
                        "end_frame": 1,
                        "description": "the man in a white shirt standing still",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sequence = _mgit_sequence(
        tmp_path / "source", frame_count=8, description_path=description_path
    )
    key = "mgit::mgit-seq"
    output = tmp_path / "gap"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
        ),
        frame_ids_by_sequence={key: (7,)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (5,)},
    )
    metadata = _read_rows(output / "source_samples.jsonl")[0]["metadata"]

    assert metadata["initial_identity_description"] == (
        "the man in a white shirt standing still"
    )
    assert metadata["initial_identity_description_source"] == "dataset_initial_language"


def test_non_mgit_identity_text_is_never_reanchored(tmp_path: Path) -> None:
    """lasot/tnl2k 的 initial_target 是静态物体短语，任何模板帧都成立。"""

    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10], [11, 10, 20, 10], [12, 10, 20, 10], [13, 10, 20, 10]],
        [True, True, True, True],
    )
    key = "synthetic::sequence-1"
    output = tmp_path / "static"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
        ),
        frame_ids_by_sequence={key: (3,)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (2,)},
    )
    metadata = _read_rows(output / "source_samples.jsonl")[0]["metadata"]

    assert metadata["initial_identity_description"] == "small gray target"
    assert metadata["initial_identity_description_source"] == "dataset_initial_language"


def test_memory_replay_is_bounded_below_by_the_template_frame(tmp_path: Path) -> None:
    """模板帧之前的更新在这条伪 episode 里从未被观测，不能回放进 Prompt。

    每个 case 都是"从模板帧开始跟踪"。回放更早的记忆等于告诉模型一段它无从得知的
    历史，推理时不可能复现。
    """

    sequence = _make_sequence(
        tmp_path / "source",
        [[10, 10, 20, 10]] * 6,
        [True] * 6,
    )
    key = "synthetic::sequence-1"
    labels = {
        key: {
            1: MemoryUpdateLabel("target now holds a blue umbrella", source="review"),
            4: MemoryUpdateLabel("target now wears a yellow helmet", source="review"),
        }
    }
    output = tmp_path / "bounded"
    build_tracking_samples(
        [sequence],
        output,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_THREE_STATE,
        ),
        frame_ids_by_sequence={key: (5, 5)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (0, 3)},
        memory_labels_by_sequence=labels,
    )
    rows = _read_rows(output / "source_samples.jsonl")
    by_reference = {row["metadata"]["reference_frame_id"]: row for row in rows}

    # 模板=0：帧 1 和 4 的更新都可见，取最近的一条。
    assert by_reference[0]["metadata"]["current_target_state"] == (
        "target now wears a yellow helmet"
    )
    # 模板=3：帧 1 早于模板，必须不可见；帧 4 仍在区间内。
    assert by_reference[3]["metadata"]["current_target_state"] == (
        "target now wears a yellow helmet"
    )

    # 模板晚于所有更新时，输入记忆必须回落到身份描述而非任何更新文本。
    late = tmp_path / "late"
    build_tracking_samples(
        [sequence],
        late,
        config=TrackingSampleConfig(
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            memory_supervision=MEMORY_SUPERVISION_THREE_STATE,
        ),
        frame_ids_by_sequence={key: (5,)},
        anchor_frame_ids_by_sequence={key: 0},
        reference_frame_ids_by_sequence={key: (4,)},
        memory_labels_by_sequence={
            key: {1: MemoryUpdateLabel("target now holds a blue umbrella", source="review")}
        },
    )
    late_metadata = _read_rows(late / "source_samples.jsonl")[0]["metadata"]
    assert late_metadata["current_target_state"] == "small gray target"
