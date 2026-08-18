import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogtrack.context import (
    PROMPT_PROFILE_VLT_V6,
    REFERENCE_MODE_VISUAL_BOX,
)
from cogtrack.training.tracking_samples import (
    MEMORY_SUPERVISION_MASKED_NULL,
    TrackingSampleConfig,
    build_tracking_samples,
)
from pytracking.evaluation.data import Sequence
from tracking.validate_tracking_sft_taxonomy import validate_tracking_sft_dataset


def _sequence(root: Path) -> Sequence:
    root.mkdir(parents=True)
    frames = []
    for frame_id in range(5):
        path = root / f"{frame_id:04d}.png"
        assert cv2.imwrite(
            str(path), np.full((50, 100, 3), 30 + frame_id * 10, dtype=np.uint8)
        )
        frames.append(str(path))
    return Sequence(
        name="taxonomy-preflight",
        frames=frames,
        dataset="synthetic",
        ground_truth_rect=[[10 + frame_id, 10, 20, 10] for frame_id in range(5)],
        target_visible=[True, True, False, True, True],
        language_query="gray target",
    )


def _build(tmp_path: Path) -> Path:
    output = tmp_path / "release"
    key = "synthetic::taxonomy-preflight"
    build_tracking_samples(
        [_sequence(tmp_path / "source")],
        output,
        config=TrackingSampleConfig(
            mode="mosaic",
            reference_mode=REFERENCE_MODE_VISUAL_BOX,
            prompt_profile=PROMPT_PROFILE_VLT_V6,
            force_history_image=True,
            memory_supervision=MEMORY_SUPERVISION_MASKED_NULL,
            history_corruption_ratio=1.0,
        ),
        frame_ids_by_sequence={key: (1, 2, 3, 4)},
        anchor_frame_ids_by_sequence={key: 0},
    )
    return output


def test_taxonomy_preflight_recomputes_build_report(tmp_path: Path) -> None:
    output = _build(tmp_path)
    report = validate_tracking_sft_dataset(
        output / "source_samples.jsonl",
        build_report=output / "build_report.json",
    )

    assert report["sample_count"] > 4
    assert len(report["tracking_scenario_counts"]) == 9
    assert len(report["visual_combination_counts"]) == 27
    assert report["temporal_event_counts"]["reappearance"] > 0


def test_taxonomy_preflight_rejects_stale_h1(tmp_path: Path) -> None:
    output = _build(tmp_path)
    source = output / "source_samples.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    row = next(
        item
        for item in rows
        if item["metadata"]["history_completeness"] == "h1_one_observation"
    )
    row["metadata"]["history_quality"] = "stale_box"
    row["metadata"]["history_corruption"] = "stale_box"
    row["metadata"]["tracking_scenario"] = (
        f"{row['metadata']['temporal_event']}__stale_box"
    )
    row["metadata"]["visual_combination"] = (
        f"{row['metadata']['tracking_scenario']}__h1_one_observation"
    )
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="不允许 history_quality=stale_box"):
        validate_tracking_sft_dataset(tampered)
