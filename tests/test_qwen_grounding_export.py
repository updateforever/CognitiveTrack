import json
from pathlib import Path

from PIL import Image

from cogtrack.training.qwen_grounding import (
    answer_with_materialized_bbox,
    to_qwen_grounding_record,
)
from cogtrack.training.swift_dataset import validate_ms_swift_record


def _image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path)


def _row(*, status: str = "present") -> dict:
    bbox = [500, 400, 800, 800] if status == "present" else None
    return {
        "id": "unit::seq::1::pair::pair",
        "images": ["images/ref.jpg", "images/current.jpg"],
        "target_status": status,
        "bbox_norm1000_xyxy": bbox,
        "metadata": {
            "source_dataset": "unit",
            "source_sequence": "seq",
            "frame_id": 1,
            "reference_frame_id": 0,
            "reference_bbox_norm1000_xyxy": [100, 200, 400, 600],
            "effective_mode": "pair",
            "history_frame_ids": [],
        },
    }


def test_qwen_families_use_different_official_coordinate_protocols(tmp_path: Path) -> None:
    _image(tmp_path / "images/ref.jpg", (200, 100))
    _image(tmp_path / "images/current.jpg", (400, 200))

    qwen25 = to_qwen_grounding_record(
        _row(), image_root=tmp_path, model_family="qwen2_5_vl"
    )
    qwen3 = to_qwen_grounding_record(_row(), image_root=tmp_path, model_family="qwen3_vl")

    assert qwen25["objects"] == {
        "bbox": [[20.0, 20.0, 80.0, 60.0], [200.0, 80.0, 320.0, 160.0]],
        "bbox_type": "real",
        "image_id": [0, 1],
    }
    assert qwen3["objects"] == qwen25["objects"]

    qwen25_user = qwen25["messages"][1]["content"]
    qwen3_user = qwen3["messages"][1]["content"]
    assert "processor-resized pixel grid" in qwen25_user
    assert "normalized 0-to-1000" in qwen3_user
    assert qwen25["messages"][2]["content"] == (
        '{"target_status":"present","bbox_pixel_xyxy":<bbox>}'
    )
    assert qwen3["messages"][2]["content"] == (
        '{"target_status":"present","bbox_norm1000_xyxy":<bbox>}'
    )
    assert not validate_ms_swift_record(qwen25, image_root=tmp_path)
    assert not validate_ms_swift_record(qwen3, image_root=tmp_path)


def test_absent_record_has_only_reference_bbox_placeholder(tmp_path: Path) -> None:
    _image(tmp_path / "images/ref.jpg", (200, 100))
    _image(tmp_path / "images/current.jpg", (400, 200))
    record = to_qwen_grounding_record(
        _row(status="absent"), image_root=tmp_path, model_family="qwen2_5_vl"
    )

    assert record["objects"]["image_id"] == [0]
    assert len(record["objects"]["bbox"]) == 1
    assistant = record["messages"][2]["content"]
    assert assistant == '{"target_status":"absent","bbox_pixel_xyxy":null}'
    assert sum(message["content"].count("<bbox>") for message in record["messages"]) == 1
    assert not validate_ms_swift_record(record, image_root=tmp_path)


def test_materialized_assistant_remains_strict_json() -> None:
    payload = answer_with_materialized_bbox(
        '{"target_status":"present","bbox_pixel_xyxy":<bbox>}',
        [10, 20, 30, 40],
    )
    assert payload == {
        "target_status": "present",
        "bbox_pixel_xyxy": [10, 20, 30, 40],
    }
    assert json.loads(json.dumps(payload)) == payload


def _visual_row(*, status: str) -> dict:
    row = _row(status=status)
    row["metadata"]["reference_mode"] = "visual_box"
    row["metadata"]["visual_marker_version"] = "red_box_v1"
    row["assistant"] = {
        "target_status": status,
        "bbox_norm1000_xyxy": row["bbox_norm1000_xyxy"],
        "memory_update": None,
    }
    return row


def test_visual_box_export_removes_input_bbox_object_and_keeps_current_output_binding(
    tmp_path: Path,
) -> None:
    _image(tmp_path / "images/ref.jpg", (200, 100))
    _image(tmp_path / "images/current.jpg", (400, 200))

    present = to_qwen_grounding_record(
        _visual_row(status="present"), image_root=tmp_path, model_family="qwen3_vl"
    )
    absent = to_qwen_grounding_record(
        _visual_row(status="absent"), image_root=tmp_path, model_family="qwen3_vl"
    )

    assert present["objects"] == {
        "bbox": [[200.0, 80.0, 320.0, 160.0]],
        "bbox_type": "real",
        "image_id": [1],
    }
    assert "objects" not in absent
    assert present["messages"][2]["content"] == (
        '{"target_status":"present","bbox_norm1000_xyxy":<bbox>,"memory_update":null}'
    )
    assert absent["messages"][2]["content"] == (
        '{"target_status":"absent","bbox_norm1000_xyxy":null,"memory_update":null}'
    )
    assert present["messages"][1]["content"].count("<bbox>") == 0
    assert "reference bbox" not in present["messages"][1]["content"].lower()
    assert not validate_ms_swift_record(present, image_root=tmp_path)
    assert not validate_ms_swift_record(absent, image_root=tmp_path)


def test_vlt_v6_export_preserves_dynamic_text_and_fixed_three_images(tmp_path: Path) -> None:
    _image(tmp_path / "images/ref.jpg", (200, 100))
    _image(tmp_path / "images/history.jpg", (200, 100))
    _image(tmp_path / "images/current.jpg", (400, 200))
    row = _visual_row(status="present")
    row["images"] = [
        "images/ref.jpg",
        "images/history.jpg",
        "images/current.jpg",
    ]
    row["metadata"].update(
        effective_mode="mosaic",
        history_frame_ids=[0],
        prompt_profile="vlt_v6",
        initial_identity_description="a small gray vehicle",
        current_target_state="rear view exposes two white stripes",
        memory_loss_masked=True,
        sft_supervision_profile="tracking_core",
    )

    record = to_qwen_grounding_record(
        row,
        image_root=tmp_path,
        model_family="qwen3_vl",
    )

    assert record["messages"][0]["content"].startswith(
        "You are a long-term vision-language single-object tracker."
    )
    user = record["messages"][1]["content"]
    assert user.count("<image>") == 3
    assert "a small gray vehicle" in user
    assert "rear view exposes two white stripes" in user
    assert "Initial target identity:" in user
    assert "Current maintained target state:" in user
    assert record["metadata"]["prompt_name"] == "cognitive_vlt_mosaic"
    assert record["metadata"]["prompt_version"] == "6.3.0"
    assert record["metadata"]["sft_supervision_profile"] == "tracking_core"
    assert record["objects"]["image_id"] == [2]
    assert not validate_ms_swift_record(record, image_root=tmp_path)
