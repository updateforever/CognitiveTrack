"""TNL2K 语言跟踪数据集加载器。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pytracking.evaluation.data import BaseDataset, Sequence, SequenceList
from pytracking.evaluation.environment import EnvironmentSettings
from pytracking.utils.io import list_image_files, load_numeric_table, read_text


def _resolve_split_root(configured_root: Path, split: str) -> Path:
    """兼容配置指向 TNL2K 总目录或具体 subset 目录的两种情况。"""

    normalized = split.lower()
    if normalized not in {"test", "train"}:
        raise ValueError(f"TNL2K split 仅支持 test/train，收到: {split}")
    root_name = configured_root.name.lower()
    if root_name in {"tnl2k_test_subset", "tnl2k_train_subset"}:
        configured_split = "test" if "_test_" in root_name else "train"
        if configured_split == normalized:
            return configured_root
        sibling = configured_root.parent / f"TNL2K_{normalized}_subset"
        if sibling.is_dir():
            return sibling
        raise FileNotFoundError(
            f"配置指向 TNL2K {configured_split} subset，但未找到请求的 {normalized} subset: {sibling}"
        )
    candidate = configured_root / f"TNL2K_{normalized}_subset"
    return candidate if candidate.is_dir() else configured_root


class TNL2KDataset(BaseDataset):
    """从序列目录动态发现 TNL2K，不依赖本地生成的列表文件。"""

    def __init__(
        self,
        environment: EnvironmentSettings | None = None,
        *,
        split: str = "test",
    ) -> None:
        super().__init__(environment)
        self.split = split.lower()
        self.base_path = _resolve_split_root(self.environment.dataset_root("tnl2k"), self.split)
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"TNL2K split 目录不存在: {self.base_path}")
        self.sequence_names = sorted(
            path.name
            for path in self.base_path.iterdir()
            if path.is_dir() and (path / "groundtruth.txt").is_file() and (path / "imgs").is_dir()
        )
        if not self.sequence_names:
            raise FileNotFoundError(f"TNL2K 中未发现有效序列: {self.base_path}")
        self.sequence_list = self.sequence_names

    def __len__(self) -> int:
        return len(self.sequence_names)

    def get_sequence_list(self) -> SequenceList:
        return SequenceList(self._construct_sequence(name) for name in self.sequence_names)

    def _construct_sequence(self, name: str) -> Sequence:
        sequence_dir = self.base_path / name
        ground_truth = load_numeric_table(sequence_dir / "groundtruth.txt", columns=4)
        # TNL2K 用 [0,0,0,0] 明确标记目标不在画面中。这里将有限且宽高为正的
        # bbox 视为 present，其余视为 absent，避免训练构造器把真实消失帧当成
        # “未知标注”跳过。首帧仍由 Sequence/构造器检查为有效初始化框。
        target_visible = np.logical_and(
            np.all(np.isfinite(ground_truth), axis=1),
            np.logical_and(ground_truth[:, 2] > 0, ground_truth[:, 3] > 0),
        )
        return Sequence(
            name=name,
            frames=list_image_files(sequence_dir / "imgs"),
            dataset="tnl2k",
            ground_truth_rect=ground_truth,
            target_visible=target_visible,
            language_query=read_text(sequence_dir / "language.txt"),
            metadata={"split": self.split, "bbox_format": "xywh"},
        )
