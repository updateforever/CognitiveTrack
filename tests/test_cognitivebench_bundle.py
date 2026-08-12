import json
from pathlib import Path

import numpy as np

from pytracking.datasets.cognitivebench import CognitiveBenchDataset
from pytracking.evaluation.environment import EnvironmentSettings, load_environment
from tools.verify_cognitivebench import verify_cognitivebench

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_sequence(root: Path) -> None:
    sequence = root / "test/example"
    sequence.mkdir(parents=True)
    (root / "benchmark_meta.json").write_text(
        json.dumps(
            {
                "version": "v1",
                "bbox_format": "xywh",
                "frame_index_base": 0,
                "sequence_counts": {"lasot": 1},
            }
        ),
        encoding="utf-8",
    )
    (sequence / "meta.json").write_text(
        json.dumps(
            {
                "sequence": "example",
                "source_dataset": "lasot",
                "source_split": "test",
                "num_frames": 2,
                "bbox_format": "xywh",
                "frame_index_base": 0,
            }
        ),
        encoding="utf-8",
    )
    (sequence / "groundtruth.txt").write_text("1,2,10,8\n0,0,0,0\n", encoding="utf-8")
    (sequence / "target_status.txt").write_text("1\n0\n", encoding="utf-8")
    (sequence / "keyframes.txt").write_text("0\n1\n", encoding="utf-8")


def test_environment_defaults_to_bundled_cognitivebench(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("COGTRACK_COGNITIVEBENCH_ROOT", raising=False)
    config = tmp_path / "env.yaml"
    config.write_text(
        "project_root: " + str(PROJECT_ROOT) + "\n"
        "datasets:\n"
        f"  lasot: {tmp_path / 'lasot'}\n"
        f"  tnl2k: {tmp_path / 'tnl2k'}\n"
        f"  mgit: {tmp_path / 'mgit'}\n",
        encoding="utf-8",
    )
    environment = load_environment(
        config,
        overrides={
            "lasot": tmp_path / "lasot",
            "tnl2k": tmp_path / "tnl2k",
            "mgit": tmp_path / "mgit",
        }
    )
    assert environment.dataset_root("cognitivebench") == (
        PROJECT_ROOT / "benchmarks/cognitivebench/v1"
    ).resolve()


def test_loader_checks_benchmark_meta_before_constructing_sequences(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    _write_sequence(benchmark)
    environment = EnvironmentSettings(
        project_root=tmp_path,
        results_path=tmp_path / "outputs",
        dataset_roots={"cognitivebench": benchmark, "lasot": tmp_path / "lasot"},
    )
    dataset = CognitiveBenchDataset(environment)

    assert len(dataset) == 1
    assert dataset.benchmark_meta["version"] == "v1"


def test_bundled_cognitivebench_statistics_are_frozen() -> None:
    report = verify_cognitivebench(PROJECT_ROOT / "benchmarks/cognitivebench/v1")

    assert report["ok"] is True
    assert report["sequence_count"] == 995
    assert report["frame_count"] == 1_408_438
    assert report["keyframe_count"] == 343_616
    assert report["sequence_counts_by_source"] == {"lasot": 280, "mgit": 15, "tnl2k": 700}
    assert np.isclose(report["absent_ratio"], 0.09532048979081792)
