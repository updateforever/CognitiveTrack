"""传统 TXT/time 与扩展 JSONL 的流式、原子结果写入器。"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import IO, Any, Mapping

import numpy as np

FRAME_SCHEMA_VERSION = "cogtrack.frame.v1"
MANIFEST_SCHEMA_VERSION = "cogtrack.run.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_bbox(value: Any) -> list[float] | None:
    """将合法 xywh bbox 规范化为 float list；NaN/Inf/None 统一为不可定位。"""

    if value is None:
        return None
    try:
        bbox = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_bbox 无法转换为数值 xywh: {value!r}") from exc
    if bbox.size != 4:
        raise ValueError(f"target_bbox 必须包含 4 个数，收到 {bbox.size}")
    if not np.all(np.isfinite(bbox)):
        return None
    return bbox.tolist()


def to_json_safe(value: Any) -> Any:
    """递归转换 numpy/Path/dataclass 等常见 tracker 输出，禁止非标准 NaN。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(item) for item in value]
    raise TypeError(f"tracker 输出包含不可 JSON 序列化的类型: {type(value).__name__}")


@dataclass(frozen=True)
class FrameRunRecord:
    """一帧推理的通用运行记录；不解释任何具体认知语义。"""

    sequence: str
    dataset: str
    frame_id: int
    image_path: str
    is_observation_frame: bool
    observation_reason: str
    time: float
    target_bbox: list[float] | None
    execution: Mapping[str, Any]
    tracker_output: Mapping[str, Any]
    ground_truth: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FRAME_SCHEMA_VERSION,
            "sequence": self.sequence,
            "dataset": self.dataset,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "is_observation_frame": self.is_observation_frame,
            "observation_reason": self.observation_reason,
            "time": self.time,
            "target_bbox": self.target_bbox,
            "execution": to_json_safe(self.execution),
            "tracker_output": to_json_safe(self.tracker_output),
            "ground_truth": to_json_safe(self.ground_truth),
        }


@dataclass(frozen=True)
class ResultPaths:
    bbox: Path
    time: Path
    frames: Path
    manifest: Path


class SequenceResultWriter:
    """逐帧写入三种结果，并仅在成功时原子发布最终文件。

    运行失败的记录保留为 ``*.partial.*``，不会被误判为已完成 benchmark。正常
    结束时通过 ``os.replace`` 一次性发布，适合被中断后安全重跑。
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        sequence: str,
        dataset: str,
        tracker_name: str,
        parameter_name: str,
        expected_frames: int,
        overwrite: bool = False,
        manifest_extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.sequence = sequence
        self.dataset = dataset
        self.tracker_name = tracker_name
        self.parameter_name = parameter_name
        self.expected_frames = expected_frames
        self.overwrite = overwrite
        self.manifest_extra = dict(manifest_extra or {})
        self.paths = ResultPaths(
            bbox=self.output_dir / f"{sequence}.txt",
            time=self.output_dir / f"{sequence}_time.txt",
            frames=self.output_dir / f"{sequence}_frames.jsonl",
            manifest=self.output_dir / f"{sequence}_manifest.json",
        )
        token = f"{os.getpid()}.tmp"
        self._temporary = ResultPaths(
            bbox=self.output_dir / f".{sequence}.bbox.{token}",
            time=self.output_dir / f".{sequence}.time.{token}",
            frames=self.output_dir / f".{sequence}.frames.{token}",
            manifest=self.output_dir / f".{sequence}.manifest.{token}",
        )
        self._handles: tuple[IO[str], IO[str], IO[str]] | None = None
        self._frames_written = 0
        self._started_at: str | None = None

    def is_complete(self) -> bool:
        """严格检查 manifest、三种结果文件和帧数。"""

        if not all(
            (
                self.paths.bbox.is_file(),
                self.paths.time.is_file(),
                self.paths.frames.is_file(),
                self.paths.manifest.is_file(),
            )
        ):
            return False
        try:
            with self.paths.manifest.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            manifest.get("completed")
            and manifest.get("frames_written") == self.expected_frames
            and manifest.get("expected_frames") == self.expected_frames
        )

    def has_published_results(self) -> bool:
        """判断是否已有任意正式结果，防止不同帧数的实验静默互相覆盖。"""

        return any(path.is_file() for path in asdict(self.paths).values())

    def __enter__(self) -> "SequenceResultWriter":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        self.close(completed=exc is None, error=exc)
        return False

    def open(self) -> None:
        if self._handles is not None:
            raise RuntimeError("SequenceResultWriter 已打开")
        if self.is_complete() and not self.overwrite:
            raise FileExistsError(f"序列结果已完整存在: {self.paths.manifest}")
        if self.has_published_results() and not self.overwrite:
            raise FileExistsError(
                f"序列已有结果但与本次期望帧数不匹配: {self.output_dir / self.sequence}；"
                "请更换 experiment name，或确认后使用 --force"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles = (
            self._temporary.bbox.open("w", encoding="utf-8", newline="\n"),
            self._temporary.time.open("w", encoding="utf-8", newline="\n"),
            self._temporary.frames.open("w", encoding="utf-8", newline="\n"),
        )
        self._started_at = _utc_now()

    def write(self, record: FrameRunRecord) -> None:
        if self._handles is None:
            raise RuntimeError("请先打开 SequenceResultWriter")
        if record.sequence != self.sequence or record.dataset != self.dataset:
            raise ValueError("FrameRunRecord 与 writer 的序列或数据集不一致")
        if record.frame_id != self._frames_written:
            raise ValueError(f"结果帧必须连续写入；期望 {self._frames_written}，收到 {record.frame_id}")

        bbox_handle, time_handle, frames_handle = self._handles
        bbox = normalize_bbox(record.target_bbox)
        if bbox is None:
            bbox_handle.write("nan\tnan\tnan\tnan\n")
        else:
            bbox_handle.write("\t".join(f"{value:.6f}" for value in bbox) + "\n")
        time_handle.write(f"{float(record.time):.9f}\n")
        frames_handle.write(json.dumps(record.as_dict(), ensure_ascii=False, allow_nan=False) + "\n")
        self._frames_written += 1

    def close(self, *, completed: bool, error: BaseException | None = None) -> None:
        if self._handles is None:
            return
        for handle in self._handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._handles = None

        completed = bool(completed and self._frames_written == self.expected_frames)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tracker": self.tracker_name,
            "parameter": self.parameter_name,
            "dataset": self.dataset,
            "sequence": self.sequence,
            "expected_frames": self.expected_frames,
            "frames_written": self._frames_written,
            "completed": completed,
            "started_at": self._started_at,
            "finished_at": _utc_now(),
            "error": None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error),
            },
            "extra": to_json_safe(self.manifest_extra),
        }
        with self._temporary.manifest.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if completed:
            for temporary, final in zip(asdict(self._temporary).values(), asdict(self.paths).values(), strict=False):
                os.replace(temporary, final)
            return

        partial = ResultPaths(
            bbox=self.output_dir / f"{self.sequence}.partial.txt",
            time=self.output_dir / f"{self.sequence}_time.partial.txt",
            frames=self.output_dir / f"{self.sequence}_frames.partial.jsonl",
            manifest=self.output_dir / f"{self.sequence}_manifest.partial.json",
        )
        for temporary, final in zip(asdict(self._temporary).values(), asdict(partial).values(), strict=False):
            os.replace(temporary, final)
