import json
from pathlib import Path

import pytest

from pytracking.datasets.mgit import MGITDataset


def _description_only_dataset(dataset_root: Path) -> MGITDataset:
    dataset = object.__new__(MGITDataset)
    dataset.base_path = dataset_root
    return dataset


def test_mgit_description_sorts_mixed_numeric_start_frames(tmp_path: Path) -> None:
    description_root = tmp_path / "attribute" / "description"
    description_root.mkdir(parents=True)
    (description_root / "mixed.json").write_text(
        json.dumps(
            {
                "action": {
                    "action_3": {"start_frame": 20, "description": "third action"},
                    "action_1": {"start_frame": "0\u00a0", "description": "first action"},
                    "action_2": {"start_frame": "9", "description": "second action"},
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = _description_only_dataset(tmp_path)

    assert dataset._load_description("mixed") == "First action"


def test_mgit_description_rejects_malformed_start_frame(tmp_path: Path) -> None:
    description_root = tmp_path / "attribute" / "description"
    description_root.mkdir(parents=True)
    path = description_root / "broken.json"
    path.write_text(
        json.dumps(
            {
                "action": {
                    "action_bad": {
                        "start_frame": "unknown",
                        "description": "bad action",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = _description_only_dataset(tmp_path)

    with pytest.raises(ValueError, match=r"broken\.json.*action_bad.*unknown"):
        dataset._load_description("broken")
