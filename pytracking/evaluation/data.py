"""与 pytracking 习惯兼容的单目标序列数据表示。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

import numpy as np

from .environment import EnvironmentSettings, load_environment


class BaseDataset(ABC):
    """所有评测数据集的基类。

    loader 只负责描述序列，不在初始化时读取图像像素，因此上千序列的数据集也
    能快速完成索引。环境配置通过构造参数注入，便于测试与后续开源部署。
    """

    def __init__(self, environment: EnvironmentSettings | None = None) -> None:
        self.environment = environment or load_environment()

    @abstractmethod
    def get_sequence_list(self) -> "SequenceList":
        """返回此数据集的完整序列列表。"""

    def get_sequence(self, name: str) -> "Sequence":
        """按名称构建单个序列。

        首批 loader 均维护轻量的 ``sequence_names`` 并实现
        ``_construct_sequence``。单序列调试走此接口，可避免为 CognitiveBench
        预先展开上百万个帧路径。
        """

        sequence_names = getattr(self, "sequence_names", None)
        constructor = getattr(self, "_construct_sequence", None)
        if sequence_names is not None and callable(constructor):
            if name not in sequence_names:
                raise KeyError(f"数据集 {self.__class__.__name__} 中不存在序列: {name}")
            return constructor(name)
        return self.get_sequence_list()[name]

    def __iter__(self) -> Iterator["Sequence"]:
        return iter(self.get_sequence_list())


class Sequence:
    """单个单目标跟踪序列。

    ``ground_truth_rect`` 始终使用像素级 ``xywh``。runner 的 ``frame_info``
    不包含当前帧 GT，避免推理阶段误读未来标注；完整标注只供 evaluator 使用。
    """

    def __init__(
        self,
        name: str,
        frames: SequenceABC[str | Path],
        dataset: str,
        ground_truth_rect: np.ndarray | SequenceABC[SequenceABC[float]] | None,
        *,
        init_data: dict[int, dict[str, Any]] | None = None,
        object_class: str | None = None,
        target_visible: np.ndarray | SequenceABC[bool] | None = None,
        target_identity: SequenceABC[str] | None = None,
        language_query: str | None = None,
        keyframe_indices: Iterable[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name or Path(name).name != name:
            raise ValueError(f"序列名必须是非空安全文件名: {name!r}")
        if not frames:
            raise ValueError(f"序列 {name} 不包含任何帧")

        self.name = name
        self.frames = [str(Path(frame)) for frame in frames]
        self.dataset = dataset
        self.object_class = object_class
        self.language_query = language_query.strip() if language_query else None
        self.metadata = dict(metadata or {})
        # 保留 pytracking 常见的单目标属性名，方便迁移 SUTrack tracker；v1 暂不
        # 实现多目标和分割，但显式 None/False 比运行时缺属性更易诊断。
        self.ground_truth_seg = None
        self.object_ids = None
        self.multiobj_mode = False

        if ground_truth_rect is None:
            self.ground_truth_rect = None
        else:
            ground_truth = np.asarray(ground_truth_rect, dtype=np.float64).reshape(-1, 4)
            if len(ground_truth) != len(self.frames):
                raise ValueError(f"序列 {name}: GT 长度 {len(ground_truth)} != 帧数 {len(self.frames)}")
            self.ground_truth_rect = ground_truth

        if target_visible is None:
            self.target_visible = None
        else:
            visible = np.asarray(target_visible, dtype=bool).reshape(-1)
            if len(visible) != len(self.frames):
                raise ValueError(f"序列 {name}: target_visible 长度 {len(visible)} != 帧数 {len(self.frames)}")
            self.target_visible = visible

        # 普通 SOT 数据集通常没有逐帧身份真值，因此默认 None。未来的候选级
        # 身份 benchmark 可显式传入；runner 只会落盘，不会把它暴露给 tracker。
        if target_identity is None:
            self.target_identity = None
        else:
            identities = tuple(str(value).strip().lower() for value in target_identity)
            if len(identities) != len(self.frames):
                raise ValueError(
                    f"序列 {name}: target_identity 长度 {len(identities)} != 帧数 {len(self.frames)}"
                )
            allowed_identities = {"same", "different", "uncertain", "not_applicable"}
            invalid_identities = sorted(set(identities) - allowed_identities)
            if invalid_identities:
                raise ValueError(
                    f"序列 {name}: target_identity 含非法值 {invalid_identities}；"
                    f"允许 {sorted(allowed_identities)}"
                )
            self.target_identity = identities

        indices = frozenset(int(index) for index in (keyframe_indices or ()))
        invalid_indices = sorted(index for index in indices if index < 0 or index >= len(self.frames))
        if invalid_indices:
            preview = invalid_indices[:8]
            raise ValueError(f"序列 {name}: 关键帧索引越界: {preview}")
        self.keyframe_indices = indices

        self.init_data = self._build_init_data(init_data)

    def _build_init_data(self, init_data: dict[int, dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
        if init_data is not None:
            normalized = deepcopy(init_data)
            for frame_id, values in normalized.items():
                if frame_id < 0 or frame_id >= len(self.frames):
                    raise ValueError(f"序列 {self.name}: 初始化帧 {frame_id} 越界")
                if "bbox" in values and values["bbox"] is not None:
                    bbox = np.asarray(values["bbox"], dtype=np.float64).reshape(-1)
                    if bbox.size != 4:
                        raise ValueError(f"序列 {self.name}: 初始化 bbox 必须包含 4 个数")
                    values["bbox"] = bbox.tolist()
            return normalized

        values: dict[str, Any] = {}
        if self.ground_truth_rect is not None:
            values["bbox"] = self.ground_truth_rect[0].tolist()
        if self.language_query:
            values["nlp"] = self.language_query
        if self.object_class:
            values["object_class"] = self.object_class
        return {0: values}

    def __len__(self) -> int:
        return len(self.frames)

    def init_info(self) -> dict[str, Any]:
        """返回首帧初始化信息；这是 runner 唯一默认暴露 GT 的位置。"""

        return self.frame_info(0)

    def frame_info(self, frame_num: int) -> dict[str, Any]:
        """生成 tracker 输入上下文，不泄露当前帧评测标注。"""

        if frame_num < 0 or frame_num >= len(self.frames):
            raise IndexError(f"序列 {self.name}: 帧索引越界 {frame_num}")
        info: dict[str, Any] = {
            "frame_num": frame_num,
            "sequence_name": self.name,
            "dataset_name": self.dataset,
        }
        for key, value in self.init_data.get(frame_num, {}).items():
            info[f"init_{key}"] = deepcopy(value)
        return info

    def init_bbox(self, frame_num: int = 0) -> list[float] | None:
        value = self.init_data.get(frame_num, {}).get("bbox")
        return list(value) if value is not None else None

    def init_nlp(self, frame_num: int = 0) -> str | None:
        value = self.init_data.get(frame_num, {}).get("nlp")
        return str(value) if value is not None else None

    def init_mask(self, frame_num: int = 0) -> None:
        """v1 是 bbox-only 单目标框架，保留接口以兼容 pytracking 调用方。"""

        del frame_num
        return None

    def target_class(self, frame_num: int | None = None) -> str | None:
        del frame_num
        return self.object_class

    def get(self, name: str, frame_num: int | None = None) -> Any:
        """兼容 pytracking 的 ``sequence.get('target_class')`` 访问方式。"""

        member = getattr(self, name)
        return member(frame_num) if callable(member) else member

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, dataset={self.dataset!r}, frames={len(self)})"


_SequenceT = TypeVar("_SequenceT", bound=Sequence)


class SequenceList(list[_SequenceT], Generic[_SequenceT]):
    """支持按名称、索引、切片和索引数组访问的序列列表。"""

    @overload
    def __getitem__(self, item: int | str) -> _SequenceT: ...

    @overload
    def __getitem__(self, item: slice | list[int] | tuple[int, ...]) -> "SequenceList[_SequenceT]": ...

    def __getitem__(self, item: int | str | slice | list[int] | tuple[int, ...]):
        if isinstance(item, str):
            for sequence in self:
                if sequence.name == item:
                    return sequence
            raise KeyError(f"序列不存在: {item}")
        if isinstance(item, (list, tuple)):
            return SequenceList(super().__getitem__(index) for index in item)
        value = super().__getitem__(item)
        return SequenceList(value) if isinstance(item, slice) else value

    def __add__(self, other: Iterable[_SequenceT]) -> "SequenceList[_SequenceT]":
        return SequenceList([*self, *other])

    def names(self) -> list[str]:
        return [sequence.name for sequence in self]
