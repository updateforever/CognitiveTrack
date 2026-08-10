from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from tracking.smoke_test_qwen import _build_tracker_params, _legacy_bbox_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment():
    return SimpleNamespace(
        model_root=Path("/models"),
        project_root=PROJECT_ROOT,
    )


def test_legacy_model_config_detects_qwen3_native_bbox_protocol():
    model_config = PROJECT_ROOT / "configs/models/qwen3vl_4b.yaml"

    assert _legacy_bbox_protocol(model_config) == "norm1000"

    params = _build_tracker_params(
        Namespace(tracker_config=None, model_config=str(model_config)),
        _environment(),
    )
    assert params.bbox_protocol == "norm1000"


def test_tracker_config_preserves_explicit_bbox_protocol():
    tracker_config = PROJECT_ROOT / "configs/trackers/qwen3vl_4b_pair.yaml"
    params = _build_tracker_params(
        Namespace(tracker_config=str(tracker_config), model_config=None),
        _environment(),
    )

    assert params.bbox_protocol == "norm1000"
    assert params.model_config == "../models/qwen3vl_4b.yaml"
    assert params.runtime.model_root == "/models"
