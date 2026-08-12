"""CognitiveBench v1 标注集加载器。

CognitiveBench 本身只保存认知跟踪标注，图像由 ``meta.json`` 指向 LaSOT、
TNL2K 或 MGIT。该设计避免复制大规模视频帧，同时保持逐帧评测的一致性。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pytracking.evaluation.data import BaseDataset, Sequence, SequenceList
from pytracking.evaluation.environment import EnvironmentSettings
from pytracking.utils.io import (
    list_image_files,
    load_numeric_table,
    read_index_file,
    read_text,
)

from .tnl2k import _resolve_split_root as _resolve_tnl2k_split_root


class CognitiveBenchDataset(BaseDataset):
    """加载 present/absent 标注与 0-based 关键帧索引。"""

    _SUPPORTED_SOURCES = frozenset({"lasot", "tnl2k", "mgit"})

    def __init__(
        self,
        environment: EnvironmentSettings | None = None,
        *,
        split: str = "test",
    ) -> None:
        super().__init__(environment)
        self.split = split
        self.base_path = self.environment.dataset_root("cognitivebench")
        benchmark_meta_path = self.base_path / "benchmark_meta.json"
        self.benchmark_meta = self._read_json(benchmark_meta_path)
        if str(self.benchmark_meta.get("version")) != "v1":
            raise ValueError(
                f"CognitiveBench 仅支持冻结的 v1 标注，收到："
                f"{self.benchmark_meta.get('version')!r}"
            )
        if str(self.benchmark_meta.get("bbox_format", "")).lower() != "xywh":
            raise ValueError("CognitiveBench benchmark_meta bbox_format 必须为 xywh")
        if int(self.benchmark_meta.get("frame_index_base", -1)) != 0:
            raise ValueError("CognitiveBench benchmark_meta frame_index_base 必须为 0")
        self.annotation_root = self.base_path / split
        if not self.annotation_root.is_dir():
            raise FileNotFoundError(f"CognitiveBench split 目录不存在: {self.annotation_root}")
        self.sequence_names = sorted(
            path.name for path in self.annotation_root.iterdir() if path.is_dir() and (path / "meta.json").is_file()
        )
        if not self.sequence_names:
            raise FileNotFoundError(f"CognitiveBench 中未发现序列: {self.annotation_root}")
        expected_counts = self.benchmark_meta.get("sequence_counts") or {}
        expected_total = sum(int(value) for value in expected_counts.values())
        if expected_total and expected_total != len(self.sequence_names):
            raise ValueError(
                f"CognitiveBench 序列数与 benchmark_meta 不一致："
                f"expected={expected_total} actual={len(self.sequence_names)}"
            )
        self.sequence_list = self.sequence_names

    def __len__(self) -> int:
        return len(self.sequence_names)

    def get_sequence_list(self) -> SequenceList:
        return SequenceList(self._construct_sequence(name) for name in self.sequence_names)

    def _construct_sequence(self, name: str) -> Sequence:
        sequence_dir = self.annotation_root / name
        meta = self._read_json(sequence_dir / "meta.json")
        source_dataset = str(meta.get("source_dataset", "")).lower()
        if source_dataset not in self._SUPPORTED_SOURCES:
            raise ValueError(f"CognitiveBench 序列 {name}: 不支持来源 {source_dataset!r}")
        source_sequence = str(meta.get("sequence", name))
        source_split = str(meta.get("source_split", meta.get("split", self.split)))

        if meta.get("bbox_format", "xywh").lower() != "xywh":
            raise ValueError(f"CognitiveBench 序列 {name}: v1 只接受 xywh GT")
        if int(meta.get("frame_index_base", 0)) != 0:
            raise ValueError(f"CognitiveBench 序列 {name}: 关键帧索引必须为 0-based")

        ground_truth = load_numeric_table(sequence_dir / "groundtruth.txt", columns=4)
        target_status = load_numeric_table(sequence_dir / "target_status.txt", dtype=np.int64).reshape(-1)
        invalid_status = sorted(set(target_status.tolist()) - {0, 1})
        if invalid_status:
            raise ValueError(f"CognitiveBench 序列 {name}: target_status 只能为 0/1，收到 {invalid_status}")
        if len(target_status) == 0 or target_status[0] != 1:
            raise ValueError(f"CognitiveBench 序列 {name}: 首帧必须 present，才能按标准 SOT 初始化")

        declared_frames = meta.get("num_frames")
        if declared_frames is not None and int(declared_frames) != len(ground_truth):
            raise ValueError(f"CognitiveBench 序列 {name}: meta.num_frames={declared_frames} != GT={len(ground_truth)}")
        valid_boxes = np.all(np.isfinite(ground_truth), axis=1) & np.all(ground_truth[:, 2:] > 0, axis=1)
        invalid_present_indices = np.flatnonzero((target_status == 1) & ~valid_boxes).tolist()
        if invalid_present_indices:
            warnings.warn(
                f"CognitiveBench 序列 {name}: {len(invalid_present_indices)} 个 present 帧的 bbox 无效；"
                "保留 presence 标签，但定位评测应跳过这些框",
                RuntimeWarning,
                stacklevel=2,
            )

        frames = self._resolve_frames(source_dataset, source_sequence, source_split)
        if len(frames) != len(ground_truth) or len(target_status) != len(ground_truth):
            raise ValueError(
                f"CognitiveBench 序列 {name}: frames={len(frames)}, GT={len(ground_truth)}, "
                f"status={len(target_status)}，三者必须一致"
            )

        keyframe_path = sequence_dir / "keyframes.txt"
        keyframes = read_index_file(keyframe_path) if keyframe_path.is_file() else []
        if len(keyframes) != len(set(keyframes)):
            raise ValueError(f"CognitiveBench 序列 {name}: keyframes.txt 存在重复索引")
        return Sequence(
            name=name,
            frames=frames,
            dataset="cognitivebench",
            ground_truth_rect=ground_truth,
            target_visible=target_status == 1,
            language_query=self._resolve_language(source_dataset, source_sequence),
            keyframe_indices=keyframes,
            metadata={
                "split": self.split,
                "bbox_format": "xywh",
                "target_status": target_status,
                "invalid_present_bbox_indices": tuple(invalid_present_indices),
                "source_dataset": source_dataset,
                "source_split": source_split,
                "source_sequence": source_sequence,
                "benchmark_meta": meta,
            },
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON 文件无法读取: {path}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"JSON 顶层必须是 object: {path}")
        return value

    def _resolve_frames(self, source: str, sequence: str, split: str) -> list[str]:
        if source == "lasot":
            class_name = sequence.rsplit("-", 1)[0]
            return list_image_files(self.environment.dataset_root("lasot") / class_name / sequence / "img")
        if source == "tnl2k":
            root = _resolve_tnl2k_split_root(self.environment.dataset_root("tnl2k"), split)
            return list_image_files(root / sequence / "imgs")
        if source == "mgit":
            root = self.environment.dataset_root("mgit")
            return list_image_files(root / "data" / split / sequence / f"frame_{sequence}")
        raise AssertionError(f"未处理的数据集来源: {source}")

    def _resolve_language(self, source: str, sequence: str) -> str | None:
        if source == "lasot":
            class_name = sequence.rsplit("-", 1)[0]
            return read_text(self.environment.dataset_root("lasot") / class_name / sequence / "nlp.txt")
        if source == "tnl2k":
            # CognitiveBench 的 meta 带 source_split，但语言文件在各 subset 中位置相同。
            for split in ("test", "train"):
                root = _resolve_tnl2k_split_root(self.environment.dataset_root("tnl2k"), split)
                text = read_text(root / sequence / "language.txt")
                if text:
                    return text
            return None
        if source == "mgit":
            path = self.environment.dataset_root("mgit") / "attribute" / "description" / f"{sequence}.json"
            if path.is_file():
                payload = self._read_json(path)
                story = payload.get("story", {})
                item = story.get("story_1", {}) if isinstance(story, dict) else {}
                if isinstance(item, dict) and item.get("description"):
                    return str(item["description"]).strip() or None
        return None
