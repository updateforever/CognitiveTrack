from pathlib import Path

from pytracking.datasets.subset import read_sequence_subset, sequence_subset_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_PATH = PROJECT_ROOT / "benchmarks/cognitivebench/v1/subsets/tiny24.txt"


def test_tiny24_manifest_is_frozen_and_references_complete_sequences() -> None:
    names = read_sequence_subset(TINY_PATH)

    assert len(names) == 24
    assert len(set(names)) == 24
    assert names[0] == "273"
    assert names[-1] == "test_002_BF5_FireGUN_video_05"
    for name in names:
        sequence_dir = PROJECT_ROOT / "benchmarks/cognitivebench/v1/test" / name
        assert sequence_dir.is_dir()
        assert (sequence_dir / "groundtruth.txt").is_file()
        assert (sequence_dir / "target_status.txt").is_file()
        assert (sequence_dir / "keyframes.txt").is_file()
        assert (sequence_dir / "meta.json").is_file()


def test_dataset_config_resolves_subset_relative_to_yaml() -> None:
    config_path = PROJECT_ROOT / "configs/datasets/cognitivebench_tiny.yaml"
    payload = {
        "sequences_file": "../../benchmarks/cognitivebench/v1/subsets/tiny24.txt"
    }

    assert sequence_subset_from_config(payload, config_path) == read_sequence_subset(TINY_PATH)
