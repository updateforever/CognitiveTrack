import types

import numpy as np
import pytest

from cogtrack.models.sutrack import (
    SUTrackConfigurationError,
    SUTrackOutputError,
    SUTrackPluginLoadError,
)
from pytracking.trackers.sutrack_adapter import SUTrackAdapter, get_tracker_class


class FakeRuntime:
    def __init__(self):
        self.closed = False
        self.init_info = None

    def initialize(self, image, info):
        assert image.shape == (80, 120, 3)
        self.init_info = info
        # 官方 pytracking tracker 常在 initialize 中只修改状态而不返回结果。
        return None

    def track(self, image, info):
        del image, info
        return {
            "box_xyxy": np.asarray([10.0, 20.0, 50.0, 70.0]),
            "run": {"status": "ok", "latency_ms": 2},
            "score": np.float32(0.75),
        }

    def close(self):
        self.closed = True

    def correct(self, image, bbox_xywh, info):
        self.corrected = (image.shape, bbox_xywh, info)
        return {"applied": True, "reason": "state reset"}


def _params(**overrides):
    values = {
        "implementation": {
            "module": "test_sutrack_plugin.runtime",
            "factory": "make_runtime",
            "kwargs": {"checkpoint": "checkpoints/model.pth"},
        },
        "output": {
            "bbox_key": "box_xyxy",
            "bbox_format": "xyxy",
            "execution_key": "run",
            "score_key": "score",
        },
    }
    values.update(overrides)
    return values


def test_adapter_lazy_loads_factory_and_normalizes_output(monkeypatch):
    runtime = FakeRuntime()
    captured = {}

    def make_runtime(*, params, checkpoint):
        captured["params"] = params
        captured["checkpoint"] = checkpoint
        return runtime

    module = types.SimpleNamespace(make_runtime=make_runtime)
    imports = []

    def fake_import(name):
        imports.append(name)
        return module

    monkeypatch.setattr("pytracking.trackers.sutrack_adapter.importlib.import_module", fake_import)
    tracker = SUTrackAdapter(_params())
    assert imports == []
    assert tracker.runtime_loaded is False

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    initialized = tracker.initialize(image, {"init_bbox": [5, 6, 20, 30], "sequence_name": "demo"})
    assert imports == ["test_sutrack_plugin.runtime"]
    assert initialized["target_bbox"] == [5.0, 6.0, 20.0, 30.0]
    assert initialized["execution"] == {"status": "ok"}
    assert captured["checkpoint"] == "checkpoints/model.pth"
    assert captured["params"].implementation.module == "test_sutrack_plugin.runtime"

    output = tracker.track(image, {"frame_num": 1})
    assert output["target_bbox"] == [10.0, 20.0, 40.0, 50.0]
    assert output["execution"] == {"status": "ok", "latency_ms": 2.0}
    assert output["confidence"] == pytest.approx(0.75)
    assert output["diagnostics"]["backend"] == "sutrack_plugin"

    correction = tracker.correct(image, [30, 20, 10, 12], {"frame_num": 1})
    assert correction == {"supported": True, "applied": True, "reason": "state reset"}
    assert runtime.corrected == ((80, 120, 3), [30.0, 20.0, 10.0, 12.0], {"frame_num": 1})

    tracker.close()
    assert runtime.closed is True
    assert tracker.runtime_loaded is False


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "implementation mapping"),
        (_params(implementation={"module": "../legacy/runtime"}), "绝对 Python 模块名"),
        (_params(implementation={"module": "plugin", "kwargs": []}), "kwargs 必须是 mapping"),
        (
            _params(implementation={"module": "plugin", "kwargs": {"params": {}}}),
            "保留参数",
        ),
        (_params(output={"bbox_format": "auto"}), "只允许 xywh 或 xyxy"),
    ],
)
def test_adapter_rejects_invalid_configuration(params, message):
    with pytest.raises(SUTrackConfigurationError, match=message):
        SUTrackAdapter(params)


def test_adapter_reports_missing_factory_in_chinese(monkeypatch):
    monkeypatch.setattr(
        "pytracking.trackers.sutrack_adapter.importlib.import_module",
        lambda name: types.SimpleNamespace(),
    )
    tracker = SUTrackAdapter(_params())
    with pytest.raises(SUTrackPluginLoadError, match="不存在可调用工厂"):
        tracker.initialize(np.zeros((4, 4, 3), dtype=np.uint8), {"init_bbox": [0, 0, 2, 2]})


def test_adapter_does_not_turn_unknown_semantic_status_into_execution_status(monkeypatch):
    class LostRuntime(FakeRuntime):
        def track(self, image, info):
            del image, info
            return {"run": "lost"}

    module = types.SimpleNamespace(make_runtime=lambda **kwargs: LostRuntime())
    monkeypatch.setattr("pytracking.trackers.sutrack_adapter.importlib.import_module", lambda name: module)
    tracker = SUTrackAdapter(_params())
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    tracker.initialize(image, {"init_bbox": [5, 6, 20, 30]})
    with pytest.raises(SUTrackOutputError, match="未知 SUTrack execution.status='lost'"):
        tracker.track(image, {"frame_num": 1})


def test_adapter_allows_failure_without_fabricated_bbox(monkeypatch):
    class FailedRuntime(FakeRuntime):
        def track(self, image, info):
            del image, info
            return {
                "run": {
                    "status": "model_error",
                    "error_type": "RuntimeError",
                    "error_message": "backend failed",
                }
            }

    module = types.SimpleNamespace(make_runtime=lambda **kwargs: FailedRuntime())
    monkeypatch.setattr("pytracking.trackers.sutrack_adapter.importlib.import_module", lambda name: module)
    tracker = SUTrackAdapter(_params())
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    tracker.initialize(image, {"init_bbox": [5, 6, 20, 30]})
    output = tracker.track(image, {"frame_num": 1})
    assert output["target_bbox"] is None
    assert output["execution"]["status"] == "model_error"


def test_dynamic_factory_returns_adapter_class():
    assert get_tracker_class() is SUTrackAdapter


def test_adapter_reports_unsupported_optional_correction(monkeypatch):
    class RuntimeWithoutCorrection:
        def initialize(self, image, info):
            del image, info

        def track(self, image, info):
            del image, info
            return {"box_xyxy": [1, 1, 3, 3]}

    module = types.SimpleNamespace(make_runtime=lambda **kwargs: RuntimeWithoutCorrection())
    monkeypatch.setattr("pytracking.trackers.sutrack_adapter.importlib.import_module", lambda name: module)
    tracker = SUTrackAdapter(_params())
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    tracker.initialize(image, {"init_bbox": [5, 6, 20, 30]})
    result = tracker.correct(image, [10, 10, 20, 20], {"frame_num": 1})
    assert result["supported"] is False
    assert result["applied"] is False
