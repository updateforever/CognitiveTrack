"""遵循标准 initialize/track 生命周期的逐帧运行器。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from typing import Sequence as TypingSequence

import numpy as np

from pytracking.trackers.base import BaseTracker

from .data import Sequence
from .environment import EnvironmentSettings
from .observation_policy import DenseObservationPolicy, ObservationDecision, ObservationPolicy
from .reproducibility import collect_reproducibility_metadata
from .result_writer import FrameRunRecord, SequenceResultWriter, normalize_bbox
from .tracker import TrackerSpec, build_tracker, resolve_tracker_config


class TrackingFrameError(RuntimeError):
    """带序列和帧上下文的推理错误。"""


def read_image_rgb(path: str | Path) -> np.ndarray:
    """读取为 RGB uint8 numpy 数组。

    优先使用 OpenCV 的 ``imdecode``，它比 ``imread`` 更好地处理非 ASCII 路径；
    未安装 OpenCV 时回退 Pillow。返回颜色顺序在框架边界处统一，避免不同 backend
    重复猜测 BGR/RGB。
    """

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    try:
        import cv2

        encoded = np.fromfile(image_path, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV 无法解码图像: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image

        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))


def _normalize_output(output: dict[str, Any] | Mapping[str, Any] | None) -> dict[str, Any]:
    if output is None:
        return {}
    if not isinstance(output, Mapping):
        raise TypeError(f"tracker 输出必须是 mapping 或 None，收到 {type(output).__name__}")
    return dict(output)


def _execution_from_output(output: Mapping[str, Any], decision: ObservationDecision) -> dict[str, Any]:
    raw = output.get("execution")
    if isinstance(raw, Mapping):
        execution = dict(raw)
        execution.setdefault("status", "ok")
        return execution
    if isinstance(raw, str):
        return {"status": raw}
    if raw is not None and hasattr(raw, "status"):
        execution = {"status": str(raw.status)}
        for key in ("error_type", "error_message", "latency_ms"):
            value = getattr(raw, key, None)
            if value is not None:
                execution[key] = value
        return execution
    # 非观察帧若 tracker 仍提供连续跟踪框，说明 hybrid/轻量分支正常执行；只有
    # 完全无输出时才由 runner 标为 skipped。
    status = "skipped" if not decision.observe and output.get("target_bbox") is None else "ok"
    return {"status": status}


def _ground_truth_for_record(sequence: Sequence, frame_id: int) -> dict[str, Any] | None:
    """在 tracker 返回后为离线评测附加 GT，不把标注放入 tracker 的 ``info``。"""

    if sequence.ground_truth_rect is None:
        return None
    bbox = normalize_bbox(sequence.ground_truth_rect[frame_id])
    bbox_is_valid = bbox is not None and bbox[2] > 0 and bbox[3] > 0
    if sequence.target_visible is not None:
        present = bool(sequence.target_visible[frame_id])
    else:
        present = bbox_is_valid
    ground_truth = {
        "target_presence": "present" if present else "absent",
        "bbox_xywh": bbox if present and bbox_is_valid else None,
        "annotation_valid": not present or bbox_is_valid,
    }
    if sequence.target_identity is not None:
        ground_truth["identity_match"] = sequence.target_identity[frame_id]
    return ground_truth


@dataclass
class SequenceRunResult:
    sequence: str
    dataset: str
    records: list[FrameRunRecord] = field(default_factory=list)
    skipped_existing: bool = False

    @property
    def total_time(self) -> float:
        return sum(record.time for record in self.records)

    @property
    def errors(self) -> int:
        return sum(str(record.execution.get("status", "")).endswith("_error") for record in self.records)


class SequenceRunner:
    """执行一个 tracker 在一个序列上的完整生命周期。"""

    def __init__(
        self,
        *,
        observation_policy: ObservationPolicy | None = None,
        image_loader: Callable[[str | Path], np.ndarray] = read_image_rgb,
        fail_fast: bool = True,
        max_frames: int | None = None,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames 必须为正整数")
        self.observation_policy = observation_policy or DenseObservationPolicy()
        self.image_loader = image_loader
        self.fail_fast = fail_fast
        self.max_frames = max_frames

    def run(
        self,
        tracker: BaseTracker,
        sequence: Sequence,
        *,
        writer: SequenceResultWriter | None = None,
    ) -> SequenceRunResult:
        frame_count = min(len(sequence), self.max_frames or len(sequence))
        result = SequenceRunResult(sequence=sequence.name, dataset=sequence.dataset)
        previous_output: dict[str, Any] = {}

        def _execute() -> None:
            nonlocal previous_output
            for frame_id in range(frame_count):
                decision = ObservationDecision(False, "policy_not_evaluated")
                started_at = time.perf_counter()
                raw_output: dict[str, Any] = {}
                execution_stage = "observation_policy"
                try:
                    decision = self.observation_policy.decide(sequence, frame_id)
                    execution_stage = "image_loading"
                    image = self.image_loader(sequence.frames[frame_id])
                    # 与常规 pytracking 一致，time 只统计 tracker 推理，不含磁盘读图。
                    started_at = time.perf_counter()
                    execution_stage = "tracker_initialize" if frame_id == 0 else "tracker_track"
                    # 只有 frame 0 的 init_info 可以包含 init_bbox。后续 frame_info
                    # 只提供在线可得的帧信息；当前/未来 GT 必须留在 runner 外层，
                    # 这是排查数据泄漏时最重要的断点之一。
                    info = sequence.init_info() if frame_id == 0 else sequence.frame_info(frame_id)
                    info.update(
                        {
                            "frame_num": frame_id,
                            "frame_path": sequence.frames[frame_id],
                            "is_observation_frame": decision.observe,
                            "observation_reason": decision.reason,
                            "previous_output": previous_output,
                        }
                    )
                    # initialize 建立永久身份锚点；track 只能消费当前图像、已有状态和
                    # 历史预测。调试 tracker 输入时应在此处检查 info，而不是检查随后
                    # 为评测生成、必然包含 GT 的 FrameRunRecord。
                    output = tracker.initialize(image, info) if frame_id == 0 else tracker.track(image, info)
                    elapsed = time.perf_counter() - started_at
                    raw_output = _normalize_output(output)
                    if frame_id == 0 and "target_bbox" not in raw_output:
                        # initialize 返回 None 是 pytracking 中允许的常见写法；首帧结果
                        # 回退到 init_bbox，但仅限初始化帧，后续绝不读取 GT。
                        raw_output["target_bbox"] = info.get("init_bbox")
                    bbox = normalize_bbox(raw_output.get("target_bbox"))
                    execution = _execution_from_output(raw_output, decision)
                    previous_output = raw_output
                except Exception as exc:
                    elapsed = time.perf_counter() - started_at
                    bbox = None
                    status = "image_error" if execution_stage == "image_loading" else "internal_error"
                    execution = {
                        "status": status,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "error_stage": execution_stage,
                    }

                # GT 在 tracker 返回后才附加，唯一用途是落盘后离线评测。若在
                # tracker.track() 的调用栈内看到 ground_truth_rect/target_visible，
                # 就说明在线边界被破坏。
                record = FrameRunRecord(
                    sequence=sequence.name,
                    dataset=sequence.dataset,
                    frame_id=frame_id,
                    image_path=sequence.frames[frame_id],
                    is_observation_frame=decision.observe,
                    observation_reason=decision.reason,
                    time=elapsed,
                    target_bbox=bbox,
                    execution=execution,
                    tracker_output=raw_output,
                    ground_truth=_ground_truth_for_record(sequence, frame_id),
                )
                result.records.append(record)
                if writer is not None:
                    writer.write(record)

                if str(execution.get("status", "")).endswith("_error") and self.fail_fast:
                    raise TrackingFrameError(
                        f"{sequence.dataset}/{sequence.name} frame {frame_id}: "
                        f"{execution.get('error_type')}: {execution.get('error_message')}"
                    )

        try:
            if writer is None:
                _execute()
            else:
                with writer:
                    _execute()
        finally:
            tracker.close()
        return result


class DatasetRunner:
    """按序列构建 tracker、执行并保存结果的高层调度器。

    大模型通常独占单卡，默认串行最稳妥；外层任务系统可按序列/显卡启动多个
    进程。Qwen backend 在单进程内复用权重并串行生成，tracker 的序列状态
    仍每次重新构建；不跨进程共享 CUDA 对象。
    """

    def __init__(
        self,
        tracker_spec: TrackerSpec,
        environment: EnvironmentSettings,
        *,
        observation_policy: ObservationPolicy | None = None,
        fail_fast: bool = True,
        max_frames: int | None = None,
        overwrite: bool = False,
    ) -> None:
        self.tracker_spec = tracker_spec
        self.environment = environment
        self.observation_policy = observation_policy or DenseObservationPolicy()
        self.fail_fast = fail_fast
        self.max_frames = max_frames
        self.overwrite = overwrite
        tracker_config = resolve_tracker_config(self.tracker_spec, self.environment)
        self.reproducibility = collect_reproducibility_metadata(
            project_root=self.environment.project_root,
            tracker_config=tracker_config,
            environment_config=self.environment.source_file,
        )

    def run(self, sequences: TypingSequence[Sequence]) -> list[SequenceRunResult]:
        results: list[SequenceRunResult] = []
        for sequence in sequences:
            expected_frames = min(len(sequence), self.max_frames or len(sequence))
            output_dir = self.tracker_spec.result_directory(self.environment, sequence.dataset)
            writer = SequenceResultWriter(
                output_dir,
                sequence=sequence.name,
                dataset=sequence.dataset,
                tracker_name=self.tracker_spec.name,
                parameter_name=self.tracker_spec.result_parameter_name,
                expected_frames=expected_frames,
                overwrite=self.overwrite,
                manifest_extra={
                    "observation_policy": type(self.observation_policy).__name__,
                    "debug_limited": expected_frames != len(sequence),
                    "full_sequence_frames": len(sequence),
                    "reproducibility": self.reproducibility,
                },
            )
            if writer.is_complete() and not self.overwrite:
                results.append(
                    SequenceRunResult(
                        sequence=sequence.name,
                        dataset=sequence.dataset,
                        skipped_existing=True,
                    )
                )
                continue

            tracker = build_tracker(
                self.tracker_spec,
                self.environment,
                dataset_name=sequence.dataset,
                runtime={
                    "sequence_name": sequence.name,
                    "results_dir": str(output_dir),
                    # 模型 YAML 只保存可移植的相对目录名；机器相关根目录由
                    # EnvironmentSettings 在运行时注入，避免把本机绝对路径写进实验配置。
                    "model_root": (
                        str(self.environment.model_root) if self.environment.model_root is not None else None
                    ),
                    "project_root": str(self.environment.project_root),
                },
            )
            # tracker 构造完才知道的运行时事实（实际服务地址、生效的坐标系参数
            # 等）补进 manifest。manifest 在 finalize 时才落盘，此处更新有效。
            runtime_description = tracker.describe_runtime()
            if runtime_description:
                writer.manifest_extra["tracker_runtime"] = runtime_description

            runner = SequenceRunner(
                observation_policy=self.observation_policy,
                fail_fast=self.fail_fast,
                max_frames=self.max_frames,
            )
            results.append(runner.run(tracker, sequence, writer=writer))
        return results
