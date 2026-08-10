"""LaSOT 官方训练集与 Protocol-II 测试集加载器。"""

from __future__ import annotations

import numpy as np

from pytracking.evaluation.data import BaseDataset, Sequence, SequenceList
from pytracking.evaluation.environment import EnvironmentSettings
from pytracking.utils.io import ensure_unique, list_image_files, load_numeric_table, read_text


class LaSOTDataset(BaseDataset):
    """按官方 split 文件加载 LaSOT 训练集或 Protocol-II 测试集。

    不在代码中复制序列名：``training_set.txt`` / ``testing_set.txt`` 更易核验，
    也能兼容数据集发布方修订后的镜像。若 split 文件缺失会明确报错，绝不通过
    扫描目录猜测划分，防止训练和测试序列混用。
    """

    def __init__(
        self,
        environment: EnvironmentSettings | None = None,
        *,
        split: str = "test",
    ) -> None:
        super().__init__(environment)
        normalized = split.lower()
        aliases = {
            "train": "train",
            "training": "train",
            "test": "test",
            "testing": "test",
            "protocol_ii": "test",
        }
        if normalized not in aliases:
            raise ValueError("LaSOT split 仅支持 train/test（test 即官方 Protocol-II）")
        self.split = aliases[normalized]
        self.base_path = self.environment.dataset_root("lasot")
        split_file = self.base_path / ("training_set.txt" if self.split == "train" else "testing_set.txt")
        names = read_text(split_file, required=True)
        assert names is not None
        self.sequence_names = ensure_unique(
            (line.strip() for line in names.splitlines() if line.strip()),
            label=str(split_file),
        )
        self.sequence_list = self.sequence_names

    def __len__(self) -> int:
        return len(self.sequence_names)

    def get_sequence_list(self) -> SequenceList:
        return SequenceList(self._construct_sequence(name) for name in self.sequence_names)

    def _construct_sequence(self, name: str) -> Sequence:
        class_name = name.rsplit("-", 1)[0]
        sequence_dir = self.base_path / class_name / name
        ground_truth = load_numeric_table(sequence_dir / "groundtruth.txt", columns=4)
        frames = list_image_files(sequence_dir / "img")

        full_occlusion = load_numeric_table(sequence_dir / "full_occlusion.txt").reshape(-1)
        out_of_view = load_numeric_table(sequence_dir / "out_of_view.txt").reshape(-1)
        if len(full_occlusion) != len(frames) or len(out_of_view) != len(frames):
            raise ValueError(f"序列 {name}: 遮挡状态与帧数不一致")
        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        return Sequence(
            name=name,
            frames=frames,
            dataset="lasot",
            ground_truth_rect=ground_truth,
            object_class=class_name,
            target_visible=target_visible,
            language_query=read_text(sequence_dir / "nlp.txt"),
            metadata={"split": self.split, "bbox_format": "xywh"},
        )
