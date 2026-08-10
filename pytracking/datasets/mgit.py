"""MGIT（原 VideoCube）长时跟踪数据集加载器。

移植自 lib/test/evaluation/videocubedataset.py，保留官方 videocube.json
定义的 tiny / full 版本与 train / val / test split，因此
``videocube_val_tiny`` 等既有实验命令可以继续使用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pytracking.evaluation.data import BaseDataset, Sequence, SequenceList
from pytracking.evaluation.environment import EnvironmentSettings
from pytracking.utils.io import ensure_unique, list_image_files, load_numeric_table

_SPLIT_DEFINITION = Path(__file__).with_name("videocube.json")


def load_split_definition(version: str, split: str) -> list[str]:
    """从随包分发的 videocube.json 读取序列名列表。

    与旧实现一致：序列名来自官方 split 定义，而不是扫描数据目录。这样即使
    本地镜像多出或缺少某些序列，实验范围仍然可复现。
    """

    if not _SPLIT_DEFINITION.is_file():
        raise FileNotFoundError(f"缺少 MGIT split 定义文件: {_SPLIT_DEFINITION}")
    with _SPLIT_DEFINITION.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    if version not in payload:
        raise ValueError(f"MGIT version 仅支持 {sorted(payload)}，收到: {version}")
    splits = payload[version]
    if split not in splits:
        raise ValueError(f"MGIT {version} 仅有 split {sorted(splits)}，收到: {split}")
    return [str(name) for name in splits[split]]


class MGITDataset(BaseDataset):
    """加载 MGIT 的 tiny/full 版本与 train/val/test split。

    ``attribute/absent`` 中 1 表示目标不可见。若某个镜像没有该文件，loader
    不会从 bbox 人为推断可见性，而是令 ``target_visible=None``。
    """

    def __init__(
        self,
        environment: EnvironmentSettings | None = None,
        *,
        split: str = "test",
        version: str = "full",
    ) -> None:
        super().__init__(environment)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"MGIT split 仅支持 train/val/test，收到: {split}")
        self.version = version.lower()
        self.base_path = self.environment.dataset_root("mgit")
        self.frames_root = self.base_path / "data" / self.split
        if not self.frames_root.is_dir():
            raise FileNotFoundError(f"MGIT split 目录不存在: {self.frames_root}")
        self.sequence_names = ensure_unique(
            load_split_definition(self.version, self.split),
            label=f"videocube.json[{self.version}][{self.split}]",
        )
        self.sequence_list = self.sequence_names

    def __len__(self) -> int:
        return len(self.sequence_names)

    def get_sequence_list(self) -> SequenceList:
        return SequenceList(self._construct_sequence(name) for name in self.sequence_names)

    def _construct_sequence(self, name: str) -> Sequence:
        frames_dir = self.frames_root / name / f"frame_{name}"
        frames = list_image_files(frames_dir)
        ground_truth = load_numeric_table(
            self.base_path / "attribute" / "groundtruth" / f"{name}.txt",
            columns=4,
        )

        absent_path = self.base_path / "attribute" / "absent" / f"{name}.txt"
        target_visible: np.ndarray | None = None
        if absent_path.is_file():
            absent = load_numeric_table(absent_path).reshape(-1)
            if len(absent) != len(frames):
                raise ValueError(f"MGIT 序列 {name}: absent 长度 {len(absent)} != 帧数 {len(frames)}")
            target_visible = absent == 0

        return Sequence(
            name=name,
            frames=frames,
            dataset="mgit",
            ground_truth_rect=ground_truth,
            target_visible=target_visible,
            language_query=self._load_description(name),
            metadata={
                "split": self.split,
                "version": self.version,
                "bbox_format": "xywh",
            },
        )

    def _load_description(self, name: str) -> str | None:
        path = self.base_path / "attribute" / "description" / f"{name}.json"
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload: dict[str, Any] = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"MGIT 描述文件损坏: {path}") from exc

        story = payload.get("story", {})
        if isinstance(story, dict):
            # 优先使用官方 story_1；缺失时按键名顺序选择首个非空描述。
            candidates = ["story_1", *sorted(key for key in story if key != "story_1")]
            for key in candidates:
                item = story.get(key)
                if isinstance(item, dict) and item.get("description"):
                    text = str(item["description"]).strip()
                    if text:
                        # 与旧实现一致：首字母大写。
                        return text[0].upper() + text[1:]
        return None
