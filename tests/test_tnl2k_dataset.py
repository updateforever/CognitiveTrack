from pathlib import Path

import cv2
import numpy as np

from pytracking.datasets.tnl2k import TNL2KDataset
from pytracking.evaluation.environment import EnvironmentSettings


def test_tnl2k_zero_box_is_explicit_absent(tmp_path: Path) -> None:
    dataset_root = tmp_path / "TNL2K"
    sequence_root = dataset_root / "TNL2K_train_subset" / "sequence-1"
    frames_root = sequence_root / "imgs"
    frames_root.mkdir(parents=True)
    for frame_id in range(3):
        assert cv2.imwrite(
            str(frames_root / f"{frame_id + 1:04d}.jpg"),
            np.full((20, 30, 3), frame_id, dtype=np.uint8),
        )
    (sequence_root / "groundtruth.txt").write_text(
        "1,2,10,8\n0,0,0,0\n3,4,9,7\n",
        encoding="utf-8",
    )
    environment = EnvironmentSettings(
        project_root=tmp_path,
        results_path=tmp_path / "results",
        dataset_roots={"tnl2k": dataset_root},
    )

    sequence = TNL2KDataset(environment, split="train").get_sequence("sequence-1")

    assert sequence.target_visible is not None
    assert sequence.target_visible.tolist() == [True, False, True]
