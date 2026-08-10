from pathlib import Path

import numpy as np

from cogtrack.evaluation import evaluate_jsonl_files
from pytracking.evaluation.data import Sequence
from pytracking.evaluation.result_writer import SequenceResultWriter
from pytracking.evaluation.runner import SequenceRunner
from pytracking.trackers.dummy import DummyTracker


def test_dummy_runner_to_evaluator_closed_loop(tmp_path: Path):
    sequence = Sequence(
        name="synthetic",
        frames=["0.jpg", "1.jpg", "2.jpg"],
        dataset="synthetic",
        ground_truth_rect=np.asarray([[1, 2, 10, 10], [1, 2, 10, 10], [0, 0, 0, 0]], dtype=float),
        target_visible=np.asarray([True, True, False]),
    )
    writer = SequenceResultWriter(
        tmp_path,
        sequence=sequence.name,
        dataset=sequence.dataset,
        tracker_name="dummy",
        parameter_name="default",
        expected_frames=3,
    )
    runner = SequenceRunner(image_loader=lambda _: np.zeros((20, 20, 3), dtype=np.uint8))
    result = runner.run(DummyTracker(), sequence, writer=writer)
    assert len(result.records) == 3

    summary = evaluate_jsonl_files([tmp_path / "synthetic_frames.jsonl"])
    assert summary["num_frames"] == 3
    # v3 起认知诊断下沉到 cognitive_diagnostics，主指标是 pytracking 口径。
    diagnostics = summary["cognitive_diagnostics"]
    assert diagnostics["presence"]["fp"] == 1
    assert diagnostics["execution"]["error_frames"] == 0
    # 闭环还要保证主指标真的算出来了，而不是只有诊断指标。
    assert summary["pytracking"]["num_valid_sequences"] == 1


def test_image_loading_failure_has_dedicated_execution_status():
    sequence = Sequence(
        name="broken-image",
        frames=["missing.jpg"],
        dataset="synthetic",
        ground_truth_rect=np.asarray([[1, 2, 10, 10]], dtype=float),
    )
    runner = SequenceRunner(
        image_loader=lambda _: (_ for _ in ()).throw(OSError("decode failed")),
        fail_fast=False,
    )

    result = runner.run(DummyTracker(), sequence)

    assert result.errors == 1
    assert result.records[0].execution["status"] == "image_error"
    assert result.records[0].execution["error_stage"] == "image_loading"


def test_optional_identity_ground_truth_is_logged_only_after_tracking():
    sequence = Sequence(
        name="identity-labeled",
        frames=["0.jpg", "1.jpg"],
        dataset="synthetic",
        ground_truth_rect=np.asarray([[1, 2, 10, 10], [1, 2, 10, 10]], dtype=float),
        target_identity=("same", "different"),
    )
    seen_infos = []

    class InspectTracker(DummyTracker):
        def initialize(self, image, info):
            seen_infos.append(dict(info))
            return super().initialize(image, info)

        def track(self, image, info):
            seen_infos.append(dict(info))
            return super().track(image, info)

    runner = SequenceRunner(image_loader=lambda _: np.zeros((20, 20, 3), dtype=np.uint8))
    result = runner.run(InspectTracker(), sequence)

    assert all("target_identity" not in info and "identity_match" not in info for info in seen_infos)
    assert result.records[1].ground_truth["identity_match"] == "different"
