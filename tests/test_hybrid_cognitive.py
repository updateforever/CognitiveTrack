from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import pytracking.trackers.hybrid_cognitive as hybrid_module
from pytracking.trackers.hybrid_cognitive import HybridCognitiveTracker, get_tracker_class


def _prediction(
    *,
    presence="present",
    identity="same",
):
    return {
        "target_presence": presence,
        "identity_match": identity,
        "localizability": "localizable" if presence == "present" else "not_applicable",
        "bbox_xywh": None,
    }


def _vlm_committed(bbox):
    prediction = _prediction()
    prediction["bbox_xywh"] = list(bbox)
    return {
        "target_bbox": list(bbox),
        "candidate_bbox": list(bbox),
        "commit_decision": {"accepted": True, "reason": "trusted"},
        "execution": {"status": "ok"},
        "prediction": prediction,
        "cognition": {"reasoning": "same target"},
        "schema_version": "cogtrack.v2",
    }


def _vlm_uncommitted(**prediction_values):
    return {
        "target_bbox": None,
        "candidate_bbox": None,
        "commit_decision": {"accepted": False, "reason": "not trusted"},
        "execution": {"status": "ok"},
        "prediction": _prediction(**prediction_values),
    }


def _vlm_skipped():
    return {
        "target_bbox": None,
        "execution": {"status": "skipped"},
        "prediction": None,
        "commit_decision": {"accepted": False, "reason": "non observation"},
    }


def _vlm_error():
    return {
        "target_bbox": None,
        "execution": {
            "status": "model_error",
            "error_type": "RuntimeError",
            "error_message": "fake qwen failure",
        },
        "prediction": None,
    }


def _install_fakes(monkeypatch, vlm_outputs, sutrack_outputs=None):
    sutrack_outputs = list(sutrack_outputs or [])

    class FakeSUTrack:
        instances = []

        def __init__(self, params):
            self.params = params
            self.track_infos = []
            self.corrections = []
            self.closed = False
            self.outputs = deepcopy(sutrack_outputs)
            type(self).instances.append(self)

        def initialize(self, image, info):
            del image
            return {"target_bbox": list(info["init_bbox"]), "execution": {"status": "ok"}}

        def track(self, image, info):
            del image
            self.track_infos.append(dict(info))
            if self.outputs:
                return self.outputs.pop(0)
            frame_id = int(info["frame_num"])
            return {"target_bbox": [10 + frame_id, 20, 30, 40], "execution": {"status": "ok"}}

        def correct(self, image, bbox_xywh, info):
            self.corrections.append((image.copy(), list(bbox_xywh), dict(info)))
            return {"supported": True, "applied": True, "reason": "fake corrected"}

        def close(self):
            self.closed = True

    class FakeVLM:
        instances = []

        def __init__(self, params):
            self.params = params
            self.track_infos = []
            self.closed = False
            self.outputs = deepcopy(list(vlm_outputs))
            type(self).instances.append(self)

        def initialize(self, image, info):
            del image
            return _vlm_committed(info["init_bbox"])

        def track(self, image, info):
            del image
            self.track_infos.append(dict(info))
            return self.outputs.pop(0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(hybrid_module, "SUTrackAdapter", FakeSUTrack)
    monkeypatch.setattr(hybrid_module, "CognitiveVLMTracker", FakeVLM)
    return FakeSUTrack, FakeVLM


def _params(tmp_path, fusion=None):
    values = {
        "_config_path": str(tmp_path / "configs" / "trackers" / "hybrid.yaml"),
        "runtime": {"dataset_name": "demo", "model_root": str(tmp_path / "models")},
        "sutrack": {
            "implementation": {"module": "fake_sutrack.runtime"},
            "runtime": {"device": "cuda:0"},
        },
        "vlm": {
            "model_config": "../models/qwen.yaml",
            "runtime": {"device": "cuda:1"},
        },
    }
    if fusion is not None:
        values["fusion"] = fusion
    return values


def _initialize(tracker):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    output = tracker.initialize(
        image,
        {
            "init_bbox": [5, 6, 20, 30],
            "frame_num": 0,
            "is_observation_frame": True,
            "sequence_name": "demo",
        },
    )
    return image, output


def test_non_keyframe_runs_both_branches_and_uses_dense_sutrack_bbox(monkeypatch, tmp_path):
    fake_sutrack, fake_vlm = _install_fakes(monkeypatch, [_vlm_skipped()])
    tracker = HybridCognitiveTracker(_params(tmp_path))
    image, initialized = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": False})

    assert initialized["committed_target_presence"] == "present"
    assert output["target_bbox"] == [11.0, 20.0, 30.0, 40.0]
    assert output["committed_target_presence"] == "present"
    assert output["diagnostics"]["selected_source"] == "sutrack"
    assert output["diagnostics"]["baseline_assumed_present"] is True
    assert output["diagnostics"]["vlm_execution"]["status"] == "skipped"
    assert len(fake_sutrack.instances[0].track_infos) == 1
    assert len(fake_vlm.instances[0].track_infos) == 1
    assert fake_vlm.instances[0].track_infos[0]["is_observation_frame"] is False

    # 子配置必须继承父 YAML 归属和 runtime，且允许子项覆盖 device。
    for child in (fake_sutrack.instances[0], fake_vlm.instances[0]):
        assert child.params._config_path == _params(tmp_path)["_config_path"]
        assert child.params.runtime.dataset_name == "demo"
        assert child.params.runtime.model_root == str(tmp_path / "models")
    assert fake_sutrack.instances[0].params.runtime.device == "cuda:0"
    assert fake_vlm.instances[0].params.runtime.device == "cuda:1"

    tracker.close()
    assert fake_sutrack.instances[0].closed is True
    assert fake_vlm.instances[0].closed is True


def test_observation_prefers_committed_vlm_bbox_and_corrects_sutrack(monkeypatch, tmp_path):
    fake_sutrack, _ = _install_fakes(monkeypatch, [_vlm_committed([50, 30, 12, 14])])
    tracker = HybridCognitiveTracker(_params(tmp_path))
    image, _ = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": True})

    assert output["target_bbox"] == [50.0, 30.0, 12.0, 14.0]
    assert output["committed_target_presence"] == "present"
    assert output["diagnostics"]["selected_source"] == "vlm"
    assert output["diagnostics"]["sutrack_correction"]["applied"] is True
    assert fake_sutrack.instances[0].corrections[0][1] == [50.0, 30.0, 12.0, 14.0]


@pytest.mark.parametrize(
    ("prediction_values", "expected_presence"),
    [
        ({"presence": "absent", "identity": "not_applicable"}, "absent"),
        ({"presence": "present", "identity": "different"}, "uncertain"),
        ({"presence": "uncertain", "identity": "uncertain"}, "uncertain"),
    ],
)
def test_discrete_semantic_rejection_is_identity_safe_by_default(
    monkeypatch,
    tmp_path,
    prediction_values,
    expected_presence,
):
    _install_fakes(monkeypatch, [_vlm_uncommitted(**prediction_values)])
    tracker = HybridCognitiveTracker(_params(tmp_path))
    image, _ = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": True})

    assert output["target_bbox"] is None
    assert output["committed_target_presence"] == expected_presence
    assert output["execution"]["status"] == "ok"
    assert output["diagnostics"]["selected_source"] == "suppressed"


def test_semantic_rejection_can_be_configured_to_fallback_without_rewriting_absent(monkeypatch, tmp_path):
    _install_fakes(
        monkeypatch,
        [_vlm_uncommitted(presence="absent", identity="not_applicable")],
    )
    tracker = HybridCognitiveTracker(
        _params(tmp_path, {"semantic_rejection_action": "fallback"})
    )
    image, _ = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": True})

    assert output["target_bbox"] == [11.0, 20.0, 30.0, 40.0]
    assert output["committed_target_presence"] == "uncertain"
    assert output["diagnostics"]["selected_source"] == "sutrack"
    assert output["diagnostics"]["baseline_assumed_present"] is False


def test_vlm_engineering_error_falls_back_and_keeps_error_diagnostics(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, [_vlm_error()])
    tracker = HybridCognitiveTracker(_params(tmp_path))
    image, _ = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": True})

    assert output["target_bbox"] == [11.0, 20.0, 30.0, 40.0]
    assert output["execution"]["status"] == "ok"
    assert output["committed_target_presence"] == "uncertain"
    assert output["diagnostics"]["selected_source"] == "sutrack"
    assert output["diagnostics"]["vlm_execution"] == _vlm_error()["execution"]


def test_unlocalizable_prediction_uses_configured_general_fallback(monkeypatch, tmp_path):
    _install_fakes(
        monkeypatch,
        [_vlm_uncommitted()],
    )
    tracker = HybridCognitiveTracker(_params(tmp_path))
    image, _ = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": True})

    assert output["target_bbox"] == [11.0, 20.0, 30.0, 40.0]
    assert output["committed_target_presence"] == "uncertain"
    assert output["diagnostics"]["selected_source"] == "sutrack"


def test_committed_vlm_survives_sutrack_branch_error(monkeypatch, tmp_path):
    broken_sutrack = {
        "target_bbox": None,
        "execution": {
            "status": "model_error",
            "error_type": "RuntimeError",
            "error_message": "fake sutrack failure",
        },
    }
    _install_fakes(monkeypatch, [_vlm_committed([30, 25, 20, 20])], [broken_sutrack])
    tracker = HybridCognitiveTracker(_params(tmp_path))
    image, _ = _initialize(tracker)
    output = tracker.track(image, {"frame_num": 1, "is_observation_frame": True})

    assert output["target_bbox"] == [30.0, 25.0, 20.0, 20.0]
    assert output["diagnostics"]["selected_source"] == "vlm"
    assert output["diagnostics"]["sutrack_execution"]["status"] == "model_error"
    assert output["diagnostics"]["sutrack_correction"]["applied"] is True


def test_dynamic_factory_returns_hybrid_class():
    assert get_tracker_class() is HybridCognitiveTracker
