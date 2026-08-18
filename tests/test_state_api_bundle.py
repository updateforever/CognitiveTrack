from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tracking.prepare_state_update_api_bundle import build_bundle


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((80, 120, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_bundle_is_self_contained_and_keeps_absent_transition(tmp_path: Path) -> None:
    release = tmp_path / "rendered"
    assets = release / "images" / "lasot" / "seq"
    reference = assets / "reference.jpg"
    history_present = assets / "history_present.jpg"
    current_present = assets / "current_present.jpg"
    history_absent = assets / "history_absent.jpg"
    current_absent = assets / "current_absent.jpg"
    for index, path in enumerate(
        (reference, history_present, current_present, history_absent, current_absent), start=1
    ):
        _write_image(path, 20 * index)
    rows = []
    for frame_id, status, history, current in (
        (100, "present", history_present, current_present),
        (200, "absent", history_absent, current_absent),
    ):
        rows.append(
            {
                "id": f"case-{frame_id}",
                "target_status": status,
                "bbox_norm1000_xyxy": [100, 100, 500, 600] if status == "present" else None,
                "images": [
                    reference.relative_to(release).as_posix(),
                    history.relative_to(release).as_posix(),
                    current.relative_to(release).as_posix(),
                ],
                "metadata": {
                    "source_dataset": "lasot",
                    "source_sequence": "seq",
                    "source_split": "train",
                    "frame_id": frame_id,
                    "reference_frame_id": 0,
                    "history_frame_ids": [10, 20, 30],
                    "history_quality": "clean",
                    "initial_identity_description": "a white airplane",
                    "temporal_event": "absent" if status == "absent" else "continuous_present",
                },
            }
        )
    with (release / "source_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n")
    output = tmp_path / "bundle"
    info = build_bundle(release, output, sampling_plan=plan)
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert info["cases"] == 2
    assert info["absent_cases"] == 1
    assert info["absent_supervision_policy"] == {
        "first_observed_absence": "dataset_gt_disappearance_transition_v1",
        "continued_absence": "dataset_gt_continued_absence_v1",
    }
    assert [row["target_status"] for row in manifest] == ["present", "absent"]
    assert manifest[0]["images"][2].endswith("current_boxed_00000100.jpg")
    assert manifest[1]["images"][2].endswith("current_absent_00000200.jpg")
    assert (output / "tools" / "annotate_state_update_openai_api.py").is_file()
    assert (output / "SHA256SUMS").is_file()
