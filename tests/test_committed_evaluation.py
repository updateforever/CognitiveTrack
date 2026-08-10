from cogtrack.evaluation.io import canonicalize_record


def test_rejected_candidate_is_not_used_as_benchmark_bbox():
    record = {
        "sequence": "identity-case",
        "frame_id": 1,
        # runner 始终把已提交框提升到顶层；显式 null 表示拒绝候选。
        "target_bbox": None,
        "execution": {"status": "ok"},
        "tracker_output": {
            "target_bbox": None,
            "candidate_bbox": [20, 10, 40, 30],
            "committed_target_presence": "uncertain",
            "commit_decision": {"accepted": False},
            "prediction": {
                "target_presence": "present",
                "identity_match": "different",
                "bbox_xywh": [20, 10, 40, 30],
            },
        },
        "ground_truth": {"target_presence": "present", "bbox_xywh": [20, 10, 40, 30]},
    }

    frame = canonicalize_record(record, source_line=1, default_sequence="fallback")

    assert frame.pred_bbox is None
    assert frame.pred_presence == "uncertain"
    assert frame.pred_identity == "different"


def test_committed_candidate_remains_standard_prediction():
    record = {
        "sequence": "identity-case",
        "frame_id": 2,
        "target_bbox": [20, 10, 40, 30],
        "execution": {"status": "ok"},
        "tracker_output": {
            "committed_target_presence": "present",
            "prediction": {
                "target_presence": "present",
                "identity_match": "same",
                "bbox_xywh": [20, 10, 40, 30],
            },
        },
        "ground_truth": {"target_presence": "present", "bbox_xywh": [20, 10, 40, 30]},
    }

    frame = canonicalize_record(record, source_line=2, default_sequence="fallback")

    assert frame.pred_bbox == (20.0, 10.0, 40.0, 30.0)
    assert frame.pred_presence == "present"
    assert frame.pred_identity == "same"
