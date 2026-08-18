from __future__ import annotations

from tracking.annotate_state_update_openai_api import (
    CONTINUED_ABSENCE_SOURCE,
    DISAPPEARANCE_SOURCE,
    HARD_NULL_SOURCE,
    IMAGE_ROLE_LABELS,
    LABEL_SOURCE,
    _build_multimodal_content,
    _diverse_cap,
    _label_source,
)


def test_label_source_preserves_dataset_gt_absence_provenance() -> None:
    assert _label_source(target_status="absent", memory_update="gone") == DISAPPEARANCE_SOURCE
    assert _label_source(target_status="absent", memory_update=None) == CONTINUED_ABSENCE_SOURCE
    assert _label_source(target_status="present", memory_update="new state") == LABEL_SOURCE
    assert _label_source(target_status="present", memory_update=None) == HARD_NULL_SOURCE


def test_diverse_cap_keeps_only_temporal_prefixes() -> None:
    rows = [
        {"dataset": "lasot", "sequence": sequence, "frame_id": frame_id}
        for sequence in ("a", "b", "c")
        for frame_id in (10, 20, 30)
    ]
    selected = _diverse_cap(rows, limit=5, seed=7)
    assert len(selected) == 5
    for sequence in ("a", "b", "c"):
        frame_ids = sorted(
            row["frame_id"] for row in selected if row["sequence"] == sequence
        )
        assert frame_ids in ([10], [10, 20])


def test_multimodal_content_labels_each_image_immediately_before_it(tmp_path) -> None:
    images = []
    for index in range(3):
        path = tmp_path / f"image-{index}.jpg"
        path.write_bytes(f"image-{index}".encode())
        images.append(path)

    content = _build_multimodal_content("teacher prompt", images)

    assert content[0] == {"type": "text", "text": "teacher prompt"}
    assert [item["type"] for item in content] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert [content[index]["text"] for index in (1, 3, 5)] == list(IMAGE_ROLE_LABELS)
    assert all(
        content[index]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for index in (2, 4, 6)
    )
