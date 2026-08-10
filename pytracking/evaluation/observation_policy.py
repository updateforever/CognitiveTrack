"""稠密/稀疏模型观察策略。

策略只回答“本帧是否应调用昂贵观察模型”，不跳过 runner，也不生成预测框。
因此 hybrid tracker 可在非关键帧继续运行轻量跟踪器，且结果长度始终等于视频帧数。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from pytracking.utils.io import read_index_file

from .data import Sequence


@dataclass(frozen=True)
class ObservationDecision:
    observe: bool
    reason: str


class ObservationPolicy(ABC):
    """无状态观察策略接口；同一实例可安全用于多个序列。"""

    @abstractmethod
    def decide(self, sequence: Sequence, frame_id: int) -> ObservationDecision:
        """返回当前帧的观察决策。首帧通常必须观察。"""


class DenseObservationPolicy(ObservationPolicy):
    """每帧均允许观察的标准跟踪设置。"""

    def decide(self, sequence: Sequence, frame_id: int) -> ObservationDecision:
        del sequence, frame_id
        return ObservationDecision(True, "dense")


class KeyframeObservationPolicy(ObservationPolicy):
    """使用数据集内置、显式索引或外部文件的 0-based 关键帧策略。

    索引优先级为：显式 ``indices`` > ``index_file`` > ``Sequence`` 元数据。
    ``missing`` 控制缺少索引时行为，默认 ``dense`` 以避免静默产生全 NaN 结果。
    """

    def __init__(
        self,
        *,
        indices: Iterable[int] | None = None,
        index_file: str | Path | None = None,
        index_base: int = 0,
        missing: Literal["dense", "none", "error"] = "dense",
        include_first_frame: bool = True,
    ) -> None:
        if index_base not in {0, 1}:
            raise ValueError("index_base 只允许 0 或 1")
        if missing not in {"dense", "none", "error"}:
            raise ValueError("missing 只允许 dense/none/error")
        self._explicit = frozenset(int(value) - index_base for value in indices) if indices is not None else None
        self.index_file = Path(index_file).expanduser().resolve() if index_file else None
        self.index_base = index_base
        self.missing = missing
        self.include_first_frame = include_first_frame
        self._file_indices = self._load_file(self.index_file) if self.index_file else None

    def _load_file(self, path: Path) -> frozenset[int] | dict[str, frozenset[int]]:
        if not path.is_file():
            raise FileNotFoundError(f"关键帧索引文件不存在: {path}")
        if path.suffix.lower() != ".json":
            return frozenset(value - self.index_base for value in read_index_file(path))
        with path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
        if isinstance(payload, list):
            return frozenset(int(value) - self.index_base for value in payload)
        if isinstance(payload, dict):
            result: dict[str, frozenset[int]] = {}
            for name, values in payload.items():
                if isinstance(values, dict):
                    values = values.get("keyframes", values.get("indices"))
                if not isinstance(values, list):
                    raise TypeError(f"关键帧 JSON 中 {name!r} 的值必须是 list")
                result[str(name)] = frozenset(int(value) - self.index_base for value in values)
            return result
        raise TypeError("关键帧 JSON 顶层必须是 list 或 sequence->list mapping")

    def _indices_for(self, sequence: Sequence) -> frozenset[int] | None:
        if self._explicit is not None:
            return self._explicit
        if isinstance(self._file_indices, dict):
            return self._file_indices.get(sequence.name)
        if isinstance(self._file_indices, frozenset):
            return self._file_indices
        return sequence.keyframe_indices or None

    def decide(self, sequence: Sequence, frame_id: int) -> ObservationDecision:
        if frame_id < 0 or frame_id >= len(sequence):
            raise IndexError(f"序列 {sequence.name}: 帧索引越界 {frame_id}")
        if frame_id == 0 and self.include_first_frame:
            return ObservationDecision(True, "initial_frame")

        indices = self._indices_for(sequence)
        if indices is None:
            if self.missing == "error":
                raise RuntimeError(f"序列 {sequence.name} 没有关键帧索引")
            if self.missing == "dense":
                return ObservationDecision(True, "missing_keyframes_fallback_dense")
            return ObservationDecision(False, "missing_keyframes")

        invalid = [index for index in indices if index < 0 or index >= len(sequence)]
        if invalid:
            raise ValueError(f"序列 {sequence.name}: 关键帧索引越界 {sorted(invalid)[:8]}")
        if frame_id in indices:
            return ObservationDecision(True, "keyframe")
        return ObservationDecision(False, "non_keyframe")
