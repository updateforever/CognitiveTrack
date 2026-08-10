import json

import pytest

from cogtrack.protocol import ModelOutputParseError, bbox_iou_xywh
from cogtrack.vlm import parse_tracking_output


def _valid_payload():
    return {
        "target_status": "present",
        "bbox_norm1000_xyxy": [100, 100, 300, 400],
        "memory_update": None,
    }


def test_tracking_parser_converts_explicit_norm1000_bbox():
    parsed = parse_tracking_output(json.dumps(_valid_payload()), image_width=200, image_height=100)
    assert parsed.prediction.bbox_xywh == pytest.approx((20.0, 10.0, 40.0, 30.0))
    assert parsed.prediction.target_presence.value == "present"


def test_tracking_parser_rejects_unknown_field():
    payload = _valid_payload()
    payload["unexpected"] = True
    with pytest.raises(ModelOutputParseError):
        parse_tracking_output(json.dumps(payload), image_width=200, image_height=100)


def test_tracking_parser_rejects_legacy_self_reported_confidence():
    payload = _valid_payload()
    payload["presence_confidence"] = 0.99
    with pytest.raises(ModelOutputParseError, match="未知字段"):
        parse_tracking_output(json.dumps(payload), image_width=200, image_height=100)


def test_tracking_parser_rejects_unsupervised_semantic_fields():
    payload = _valid_payload()
    payload["identity_match"] = "same"
    with pytest.raises(ModelOutputParseError, match="未知字段"):
        parse_tracking_output(json.dumps(payload), image_width=200, image_height=100)


def test_tracking_parser_never_converts_invalid_json_to_absent():
    with pytest.raises(ModelOutputParseError):
        parse_tracking_output("not json", image_width=200, image_height=100)


def test_tracking_parser_accepts_a_short_memory_delta():
    payload = _valid_payload()
    payload["memory_update"] = "The target is now viewed from the rear, revealing two white stripes."
    parsed = parse_tracking_output(json.dumps(payload), image_width=200, image_height=100)
    assert parsed.cognition.memory_update_proposal == payload["memory_update"]


@pytest.mark.parametrize("memory_update", ["", 1, [], {}])
def test_tracking_parser_rejects_invalid_memory_update(memory_update):
    payload = _valid_payload()
    payload["memory_update"] = memory_update
    with pytest.raises(ModelOutputParseError, match="memory_update"):
        parse_tracking_output(json.dumps(payload), image_width=200, image_height=100)


def test_absent_prediction_cannot_update_memory():
    payload = _valid_payload()
    payload.update(
        target_status="absent",
        bbox_norm1000_xyxy=None,
        memory_update="The target changed appearance.",
    )
    with pytest.raises(ModelOutputParseError, match="absent.*memory_update"):
        parse_tracking_output(json.dumps(payload), image_width=200, image_height=100)


def test_bbox_iou():
    assert bbox_iou_xywh((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(25 / 175)
