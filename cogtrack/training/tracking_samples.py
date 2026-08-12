"""从 pytracking ``Sequence`` 构建认知跟踪多模态训练样本。

该模块只消费标准序列对象，不感知具体数据集目录结构。输出由一个通用源
``JSONL`` 和自包含图片资产组成，随后可交给 ``tracking/export_swift_dataset.py``
转换为 SFT 或 GRPO 数据。

监督信号严格限制为数据集可直接提供的 ``present/absent`` 与 bbox。本模块不
生成身份、细粒度可见性、推理文本或置信度标签。present 但 bbox 无效的帧会被
跳过，而不会被错误解释为 absent。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from typing import Sequence as TypingSequence

import cv2
import numpy as np

from cogtrack.prompts import PromptSpec, build_mosaic_prompt, build_pair_prompt
from cogtrack.protocol import BoundingBoxError, clip_xywh, pixel_xywh_to_norm1000_xyxy
from pytracking.evaluation.data import Sequence

SOURCE_SCHEMA_VERSION = "cogtrack.training.source.v4"


@dataclass(frozen=True)
class TrackingSampleConfig:
    """跟踪训练样本的确定性构建参数。"""

    mode: str = "pair"
    frame_stride: int = 1
    max_samples_per_sequence: int | None = None
    seed: int = 20260805
    history_size: int = 4
    mosaic_panel_height: int = 240
    keyframes_only: bool = False
    balance_presence: bool = False
    present_only: bool = False
    use_language_description: bool = True
    max_image_side: int | None = None
    jpeg_quality: int = 95
    reuse_existing_assets: bool = False
    history_corruption_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"pair", "mosaic", "both"}:
            raise ValueError("mode 必须为 pair、mosaic 或 both")
        if isinstance(self.frame_stride, bool) or self.frame_stride <= 0:
            raise ValueError("frame_stride 必须是正整数")
        if self.max_samples_per_sequence is not None:
            if isinstance(self.max_samples_per_sequence, bool) or self.max_samples_per_sequence <= 0:
                raise ValueError("max_samples_per_sequence 必须为正整数或 None")
        if isinstance(self.history_size, bool) or self.history_size <= 0:
            raise ValueError("history_size 必须是正整数")
        if isinstance(self.mosaic_panel_height, bool) or self.mosaic_panel_height <= 0:
            raise ValueError("mosaic_panel_height 必须是正整数")
        if isinstance(self.jpeg_quality, bool) or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality 必须位于 [1, 100]")
        if not isinstance(self.balance_presence, bool):
            raise TypeError("balance_presence 必须是 bool")
        if not isinstance(self.present_only, bool):
            raise TypeError("present_only 必须是 bool")
        if not isinstance(self.use_language_description, bool):
            raise TypeError("use_language_description 必须是 bool")
        if not isinstance(self.reuse_existing_assets, bool):
            raise TypeError("reuse_existing_assets 必须是 bool")
        if isinstance(self.history_corruption_ratio, bool) or not 0.0 <= self.history_corruption_ratio <= 1.0:
            raise ValueError("history_corruption_ratio 必须位于 [0, 1]")
        if self.max_image_side is not None:
            if isinstance(self.max_image_side, bool) or self.max_image_side <= 0:
                raise ValueError("max_image_side 必须为正整数或 None")
        if self.present_only and self.balance_presence:
            raise ValueError("present_only 与 balance_presence 不能同时启用")


@dataclass(frozen=True)
class TrackingSampleBuildReport:
    """一次构建的统计摘要；字段可直接序列化为 JSON。"""

    schema_version: str
    requested_mode: str
    balance_presence: bool
    present_only: bool
    use_language_description: bool
    max_image_side: int | None
    sampling_plan_applied: bool
    reference_mode: str
    source_jsonl: str
    image_root: str
    sequence_count: int
    sequences_with_samples: int
    sample_count: int
    present_count: int
    absent_count: int
    pair_count: int
    mosaic_count: int
    skipped_invalid_bbox: int
    skipped_unknown_presence: int
    history_corruption_ratio: float
    corrupted_mosaic_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _MutableStats:
    sequence_count: int = 0
    sequences_with_samples: int = 0
    sample_count: int = 0
    present_count: int = 0
    absent_count: int = 0
    pair_count: int = 0
    mosaic_count: int = 0
    skipped_invalid_bbox: int = 0
    skipped_unknown_presence: int = 0
    corrupted_mosaic_count: int = 0


def _safe_component(value: str) -> str:
    """生成稳定且无目录逃逸风险的资产目录名。"""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unnamed"
    if normalized == value:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"图片无法读取: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_rgb(
    path: Path,
    image: np.ndarray,
    *,
    jpeg_quality: int,
    overwrite: bool,
    reuse_existing: bool = False,
) -> None:
    if path.exists() and not overwrite:
        if reuse_existing:
            return
        raise FileExistsError(f"图片资产已存在；如需覆盖请启用 overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), parameters)
    if not success:
        raise OSError(f"图片资产写入失败: {path}")


def _raw_bbox(sequence: Sequence, frame_id: int) -> tuple[float, float, float, float] | None:
    if sequence.ground_truth_rect is None:
        return None
    values = np.asarray(sequence.ground_truth_rect[frame_id], dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)) or np.any(values[2:] <= 0):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _presence(sequence: Sequence, frame_id: int) -> str | None:
    """返回可验证的二分类真值；没有可见性也没有有效框时返回未知。"""

    if sequence.target_visible is not None:
        return "present" if bool(sequence.target_visible[frame_id]) else "absent"
    return "present" if _raw_bbox(sequence, frame_id) is not None else None


def _clip_bbox_to_image(
    bbox: TypingSequence[float] | None,
    image: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    height, width = image.shape[:2]
    try:
        return clip_xywh(bbox, width, height)
    except (BoundingBoxError, TypeError, ValueError):
        return None


def _scale_bbox_to_image(
    bbox: TypingSequence[float] | None,
    image: np.ndarray,
    source_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """把原始帧 xywh 等比例映射到已导出的缩放图，再执行边界裁剪。"""

    if bbox is None:
        return None
    source_width, source_height = source_size
    height, width = image.shape[:2]
    scale_x = width / float(source_width)
    scale_y = height / float(source_height)
    scaled = (
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    )
    return _clip_bbox_to_image(scaled, image)
def _draw_reference(
    image: np.ndarray,
    bbox_xywh: TypingSequence[float],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """在 RGB 副本上绘制指定颜色的可信框，不修改原图。"""

    output = np.ascontiguousarray(image.copy())
    height, width = output.shape[:2]
    x, y, box_width, box_height = clip_xywh(bbox_xywh, width, height)
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(x1, min(width - 1, int(round(x + box_width))))
    y2 = max(y1, min(height - 1, int(round(y + box_height))))
    thickness = max(2, int(round(min(width, height) / 180.0)))
    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
    return output


def _resize_long_side(image: np.ndarray, max_image_side: int | None) -> np.ndarray:
    """等比例限制长边；norm1000 标签不受整体缩放影响。"""

    if max_image_side is None:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_image_side:
        return image
    scale = max_image_side / float(long_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _build_mosaic(
    panels: list[tuple[int, np.ndarray, tuple[float, float, float, float]]],
    *,
    panel_height: int,
) -> np.ndarray:
    """用过去 GT 模拟在线已接受的可信框，并按时间顺序拼成 RGB mosaic。"""

    rendered: list[np.ndarray] = []
    for _frame_id, image, bbox in panels:
        # RGB 红框与在线 ContextBuilder 的可信正记忆渲染保持一致。GT 只来自过去帧，
        # 用于 teacher-forcing 历史上下文，不包含 current 或未来标签。
        panel = _draw_reference(image, bbox, color=(255, 0, 0))
        height, width = panel.shape[:2]
        resized_width = max(1, int(round(width * panel_height / float(height))))
        interpolation = cv2.INTER_AREA if panel_height < height else cv2.INTER_LINEAR
        panel = cv2.resize(panel, (resized_width, panel_height), interpolation=interpolation)
        # 不把绝对帧号写进视觉输入；时间顺序由 panel 的排列表达，frame_id 只保留在
        # metadata/history_frame_ids 中供审计，避免模型学习数据集位置偏置。
        rendered.append(panel)

    if not rendered:
        raise ValueError("构造 mosaic 至少需要一个有效历史帧")
    columns = 1 if len(rendered) <= 2 else 2
    rows = (len(rendered) + columns - 1) // columns
    cell_width = max(panel.shape[1] for panel in rendered)
    cell_height = panel_height
    canvas = np.full(
        (rows * cell_height, columns * cell_width, 3),
        220,
        dtype=np.uint8,
    )
    for index, panel in enumerate(rendered):
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - panel.shape[1]) // 2
        y = row * cell_height
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def _corrupt_history_panels(
    panels: list[tuple[int, np.ndarray, tuple[float, float, float, float]]],
    *,
    seed_key: str,
) -> tuple[list[tuple[int, np.ndarray, tuple[float, float, float, float]]], str]:
    """Inject one bounded history-box error while preserving the underlying frames.

    The reference and current frame remain clean. Errors are deliberately plausible:
    a shifted/scaled box or a stale box copied from a neighboring history panel. This
    models tracker history noise without inventing a distractor identity label.
    """

    if not panels:
        raise ValueError("corrupted history requires at least one panel")
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    panel_index = digest[0] % len(panels)
    mode = "stale_box" if digest[1] % 3 == 0 and len(panels) > 1 else "jitter_box"
    corrupted = list(panels)
    frame_id, image, bbox = corrupted[panel_index]
    x, y, width, height = bbox
    if mode == "stale_box":
        source_index = (panel_index - 1) if panel_index > 0 else 1
        stale_bbox = corrupted[source_index][2]
        bbox = stale_bbox
    else:
        direction_x = -1.0 if digest[2] & 1 else 1.0
        direction_y = -1.0 if digest[3] & 1 else 1.0
        image_height, image_width = image.shape[:2]
        new_width = min(max(2.0, width * (0.75 if digest[4] & 1 else 1.25)), image_width * 0.8)
        new_height = min(max(2.0, height * (0.75 if digest[5] & 1 else 1.25)), image_height * 0.8)
        shifted_x = x + direction_x * max(8.0, width * 0.55)
        shifted_y = y + direction_y * max(6.0, height * 0.40)
        bbox = (
            min(max(0.0, shifted_x), max(0.0, image_width - new_width)),
            min(max(0.0, shifted_y), max(0.0, image_height - new_height)),
            new_width,
            new_height,
        )
        clipped = _clip_bbox_to_image(bbox, image)
        if clipped is None:
            raise ValueError(f"无法构造有效 corrupted history bbox: frame={frame_id}")
        bbox = clipped
    corrupted[panel_index] = (frame_id, image, bbox)
    return corrupted, mode


def _stable_sample(
    frame_ids: list[int],
    count: int | None,
    *,
    sequence: Sequence,
    seed: int,
    salt: str = "all",
) -> list[int]:
    if count is None or len(frame_ids) <= count:
        return frame_ids
    key = f"{seed}\0{sequence.dataset}\0{sequence.name}\0{salt}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")
    return sorted(random.Random(stable_seed).sample(frame_ids, count))


def _balanced_presence_sample(
    frame_ids: list[int],
    count: int | None,
    *,
    sequence: Sequence,
    seed: int,
) -> list[int]:
    """在单序列内尽量等量抽取 present/absent，并用另一类补足缺口。"""

    if count is None or len(frame_ids) <= count:
        return frame_ids
    present_ids = [frame_id for frame_id in frame_ids if _presence(sequence, frame_id) == "present"]
    absent_ids = [frame_id for frame_id in frame_ids if _presence(sequence, frame_id) == "absent"]
    present_target = count // 2
    absent_target = count - present_target
    selected_present = _stable_sample(
        present_ids,
        min(present_target, len(present_ids)),
        sequence=sequence,
        seed=seed,
        salt="present",
    )
    selected_absent = _stable_sample(
        absent_ids,
        min(absent_target, len(absent_ids)),
        sequence=sequence,
        seed=seed,
        salt="absent",
    )
    selected = set((*selected_present, *selected_absent))
    remaining = count - len(selected)
    if remaining > 0:
        pool = [frame_id for frame_id in frame_ids if frame_id not in selected]
        selected.update(
            _stable_sample(
                pool,
                min(remaining, len(pool)),
                sequence=sequence,
                seed=seed,
                salt="remainder",
            )
        )
    return sorted(selected)


def _candidate_frame_ids(
    sequence: Sequence,
    config: TrackingSampleConfig,
    stats: _MutableStats,
    *,
    planned_frame_ids: TypingSequence[int] | None = None,
) -> list[int]:
    if planned_frame_ids is not None:
        source_ids = sorted(int(frame_id) for frame_id in planned_frame_ids)
        invalid_ids = [frame_id for frame_id in source_ids if frame_id <= 0 or frame_id >= len(sequence)]
        if invalid_ids:
            raise ValueError(f"序列 {sequence.name} 的采样计划包含越界帧: {invalid_ids[:8]}")
    elif config.keyframes_only:
        source_ids = sorted(frame_id for frame_id in sequence.keyframe_indices if frame_id > 0)
        source_ids = source_ids[:: config.frame_stride]
    else:
        # 帧 0 只作为永久身份锚点，不生成“参考图等于当前图”的简单样本。
        source_ids = list(range(1, len(sequence), config.frame_stride))

    eligible: list[int] = []
    for frame_id in source_ids:
        presence = _presence(sequence, frame_id)
        if presence is None:
            if planned_frame_ids is not None:
                raise ValueError(f"序列 {sequence.name} 的计划帧 {frame_id} 缺少可验证 presence")
            stats.skipped_unknown_presence += 1
            continue
        if config.present_only and presence != "present":
            continue
        if presence == "present" and _raw_bbox(sequence, frame_id) is None:
            if planned_frame_ids is not None:
                raise ValueError(f"序列 {sequence.name} 的计划正帧 {frame_id} 缺少有效 bbox")
            stats.skipped_invalid_bbox += 1
            continue
        eligible.append(frame_id)
    if planned_frame_ids is not None:
        return eligible
    sampler = _balanced_presence_sample if config.balance_presence else _stable_sample
    return sampler(
        eligible,
        config.max_samples_per_sequence,
        sequence=sequence,
        seed=config.seed,
    )


def _history_panels(
    sequence: Sequence,
    current_frame_id: int,
    history_size: int,
    *,
    anchor_frame_id: int,
    keyframes_only: bool,
    existing_asset_dir: Path | None = None,
    candidate_frame_ids: TypingSequence[int] | None = None,
    image_cache: dict[int, np.ndarray] | None = None,
    source_size: tuple[int, int] | None = None,
) -> list[tuple[int, np.ndarray, tuple[float, float, float, float]]]:
    """选取最近的有效正历史；只读过去帧，杜绝时序标签泄漏。"""

    panels: list[tuple[int, np.ndarray, tuple[float, float, float, float]]] = []
    if candidate_frame_ids is None:
        frame_ids = range(current_frame_id - 1, anchor_frame_id, -1)
    else:
        frame_ids = sorted(
            (int(frame_id) for frame_id in candidate_frame_ids if anchor_frame_id < frame_id < current_frame_id),
            reverse=True,
        )
    for frame_id in frame_ids:
        if keyframes_only and frame_id not in sequence.keyframe_indices:
            continue
        if _presence(sequence, frame_id) != "present":
            continue
        raw_bbox = _raw_bbox(sequence, frame_id)
        if raw_bbox is None:
            continue
        existing_asset = (
            existing_asset_dir / f"current_{frame_id:08d}.jpg"
            if existing_asset_dir is not None
            else None
        )
        image_path = (
            existing_asset
            if existing_asset is not None and existing_asset.is_file()
            else sequence.frames[frame_id]
        )
        if image_cache is not None and frame_id in image_cache:
            image = image_cache[frame_id]
        else:
            image = _read_rgb(image_path)
            if image_cache is not None:
                image_cache[frame_id] = image
        bbox = (
            _scale_bbox_to_image(raw_bbox, image, source_size)
            if source_size is not None
            else _clip_bbox_to_image(raw_bbox, image)
        )
        if bbox is None:
            continue
        panels.append((frame_id, image, bbox))
        if len(panels) >= history_size:
            break
    panels.reverse()
    return panels


def _answer(
    *,
    presence: str,
    bbox_norm1000_xyxy: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    if presence == "present":
        if bbox_norm1000_xyxy is None:
            raise ValueError("present 训练样本必须包含有效的 norm1000 bbox")
        return {
            "target_status": "present",
            "bbox_norm1000_xyxy": list(bbox_norm1000_xyxy) if bbox_norm1000_xyxy else None,
        }
    return {
        "target_status": "absent",
        "bbox_norm1000_xyxy": None,
    }


def _record(
    *,
    sequence: Sequence,
    frame_id: int,
    reference_frame_id: int,
    reference_bbox_norm1000_xyxy: tuple[float, float, float, float],
    reference_path: str,
    current_path: str,
    history_path: str | None,
    history_frame_ids: list[int],
    prompt: PromptSpec,
    requested_mode: str,
    effective_mode: str,
    presence: str,
    bbox_norm1000_xyxy: tuple[float, float, float, float] | None,
) -> dict[str, Any]:
    images = [reference_path]
    if history_path is not None:
        images.append(history_path)
    images.append(current_path)
    if len(images) != prompt.expected_image_count:
        raise ValueError(
            f"Prompt {prompt.name} 期望 {prompt.expected_image_count} 张图，实际为 {len(images)}"
        )
    answer = _answer(
        presence=presence,
        bbox_norm1000_xyxy=bbox_norm1000_xyxy,
    )
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "id": (
            f"{sequence.dataset}::{sequence.name}::{reference_frame_id:08d}::{frame_id:08d}::"
            f"{requested_mode}::{effective_mode}"
        ),
        "task": f"cognitive_tracking_{effective_mode}",
        "system_prompt": prompt.system_prompt,
        "user_prompt": "<image>" * len(images) + "\n" + prompt.user_prompt,
        "assistant": answer,
        "images": images,
        "target_status": answer["target_status"],
        "bbox_norm1000_xyxy": answer["bbox_norm1000_xyxy"],
        "bbox_format": "norm1000_xyxy",
        "metadata": {
            "dataset": sequence.dataset,
            "source_dataset": sequence.dataset,
            "sequence": sequence.name,
            "source_sequence": sequence.name,
            "frame_id": frame_id,
            "reference_frame_id": reference_frame_id,
            "reference_bbox_norm1000_xyxy": list(reference_bbox_norm1000_xyxy),
            "reference_mode": "full_frame_bbox_text",
            "history_frame_ids": history_frame_ids,
            "requested_mode": requested_mode,
            "effective_mode": effective_mode,
            "prompt_name": prompt.name,
            "prompt_version": prompt.version,
            "bbox_format": "norm1000_xyxy",
        },
    }


def build_tracking_samples(
    sequences: Iterable[Sequence],
    output_dir: str | Path,
    *,
    config: TrackingSampleConfig | None = None,
    overwrite: bool = False,
    frame_ids_by_sequence: Mapping[str, TypingSequence[int]] | None = None,
    anchor_frame_ids_by_sequence: Mapping[str, int] | None = None,
    reference_frame_ids_by_sequence: Mapping[str, TypingSequence[int]] | None = None,
) -> TrackingSampleBuildReport:
    """流式构建源 JSONL 与图片资产，避免百万帧数据集占满内存。

    Args:
        sequences: pytracking 标准 ``Sequence`` 可迭代对象。
        output_dir: 输出根目录；图片路径在 JSONL 中始终相对该目录。
        config: 采样和上下文配置。
        overwrite: 是否覆盖同名 JSONL 与图片。默认拒绝覆盖，避免混合实验版本。
        frame_ids_by_sequence: 可选的 ``dataset::sequence -> frame ids`` 全局采样
            计划。提供后不再执行构造器内部随机采样。
        anchor_frame_ids_by_sequence: 可选的每序列初始化锚点。训练视频以首个
            有效 present 帧初始化时使用；未提供则严格使用标准 frame 0。
        reference_frame_ids_by_sequence: 可选的逐 case reference frame 计划；每个
            reference 必须早于对应的 current frame。未提供时所有 case 使用 anchor。
    """

    options = config or TrackingSampleConfig()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "source_samples.jsonl"
    report_path = root / "build_report.json"
    if not overwrite and (source_path.exists() or report_path.exists()):
        raise FileExistsError(f"输出已存在；如需重建请启用 overwrite: {root}")

    stats = _MutableStats()
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=root,
        prefix=".source_samples.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    written_assets: set[Path] = set()
    try:
        with temporary_handle as handle:
            for sequence in sequences:
                if not isinstance(sequence, Sequence):
                    raise TypeError("sequences 中的元素必须是 pytracking.evaluation.data.Sequence")
                plan_key = f"{sequence.dataset}::{sequence.name}"
                if frame_ids_by_sequence is not None and plan_key not in frame_ids_by_sequence:
                    # 全局规划器会省略没有任何合法候选帧的极短/损坏序列。
                    continue
                stats.sequence_count += 1
                if sequence.ground_truth_rect is None:
                    raise ValueError(f"序列 {sequence.name} 缺少 ground_truth_rect，无法构建监督样本")

                anchor_frame_id = 0
                if anchor_frame_ids_by_sequence is not None:
                    if plan_key not in anchor_frame_ids_by_sequence:
                        raise ValueError(f"初始化锚点计划缺少序列: {plan_key}")
                    anchor_frame_id = int(anchor_frame_ids_by_sequence[plan_key])
                if anchor_frame_id < 0 or anchor_frame_id >= len(sequence):
                    raise ValueError(f"序列 {sequence.name} 的初始化锚点越界: {anchor_frame_id}")
                anchor_image = _read_rgb(sequence.frames[anchor_frame_id])
                source_size = (anchor_image.shape[1], anchor_image.shape[0])
                anchor_raw_bbox = (
                    sequence.init_bbox(0) if anchor_frame_id == 0 else None
                ) or _raw_bbox(sequence, anchor_frame_id)
                anchor_bbox = _clip_bbox_to_image(anchor_raw_bbox, anchor_image)
                if _presence(sequence, anchor_frame_id) != "present" or anchor_bbox is None:
                    raise ValueError(
                        f"序列 {sequence.name} 的锚点帧 {anchor_frame_id} 必须包含有效初始化目标框"
                    )
                anchor_height, anchor_width = anchor_image.shape[:2]
                anchor_bbox_norm = pixel_xywh_to_norm1000_xyxy(
                    anchor_bbox,
                    anchor_width,
                    anchor_height,
                )

                dataset_dir = _safe_component(sequence.dataset)
                sequence_dir = _safe_component(sequence.name)
                asset_dir = root / "images" / dataset_dir / sequence_dir
                image_cache: dict[int, np.ndarray] = {}
                reference_asset = asset_dir / f"reference_{anchor_frame_id:08d}.jpg"
                _write_rgb(
                    reference_asset,
                    _resize_long_side(
                        anchor_image,
                        options.max_image_side,
                    ),
                    jpeg_quality=options.jpeg_quality,
                    overwrite=overwrite,
                    reuse_existing=options.reuse_existing_assets,
                )
                image_cache[anchor_frame_id] = anchor_image
                written_assets.add(reference_asset)
                reference_relative = reference_asset.relative_to(root).as_posix()
                target_text = ""
                if options.use_language_description:
                    target_text = (
                        sequence.language_query or sequence.object_class or "initialized target"
                    ).strip()

                sequence_samples = 0
                planned_frame_ids = None
                if frame_ids_by_sequence is not None:
                    planned_frame_ids = frame_ids_by_sequence[plan_key]
                planned_reference_frame_ids = None
                if reference_frame_ids_by_sequence is not None:
                    if plan_key not in reference_frame_ids_by_sequence:
                        raise ValueError(f"reference frame 计划缺少序列: {plan_key}")
                    planned_reference_frame_ids = tuple(reference_frame_ids_by_sequence[plan_key])
                    if planned_frame_ids is None or len(planned_reference_frame_ids) != len(planned_frame_ids):
                        raise ValueError(f"序列 {sequence.name} 的 reference/current 计划长度不一致")
                planned_index = 0
                for frame_id in _candidate_frame_ids(
                    sequence,
                    options,
                    stats,
                    planned_frame_ids=planned_frame_ids,
                ):
                    reference_frame_id = anchor_frame_id
                    reference_bbox_norm = anchor_bbox_norm
                    if planned_reference_frame_ids is not None:
                        reference_frame_id = int(planned_reference_frame_ids[planned_index])
                        planned_index += 1
                        if reference_frame_id < 0 or reference_frame_id >= frame_id:
                            raise ValueError(
                                f"序列 {sequence.name} 的 reference frame {reference_frame_id} "
                                f"必须严格早于 current frame {frame_id}"
                            )
                        if _presence(sequence, reference_frame_id) != "present":
                            raise ValueError(
                                f"序列 {sequence.name} 的 reference frame {reference_frame_id} 必须是有效 present 帧"
                            )
                        if reference_frame_id != anchor_frame_id:
                            reference_asset_for_case = asset_dir / f"reference_{reference_frame_id:08d}.jpg"
                            reference_source = (
                                reference_asset_for_case
                                if options.reuse_existing_assets and reference_asset_for_case.is_file()
                                else sequence.frames[reference_frame_id]
                            )
                            reference_image = image_cache.get(reference_frame_id)
                            if reference_image is None:
                                reference_image = _read_rgb(reference_source)
                                image_cache[reference_frame_id] = reference_image
                            reference_bbox = _scale_bbox_to_image(
                                _raw_bbox(sequence, reference_frame_id), reference_image, source_size
                            )
                            if reference_bbox is None:
                                raise ValueError(
                                    f"序列 {sequence.name} 的 reference frame {reference_frame_id} 缺少有效 bbox"
                                )
                            ref_height, ref_width = reference_image.shape[:2]
                            reference_bbox_norm = pixel_xywh_to_norm1000_xyxy(
                                reference_bbox, ref_width, ref_height
                            )
                            if reference_asset_for_case not in written_assets:
                                _write_rgb(
                                    reference_asset_for_case,
                                    _resize_long_side(reference_image, options.max_image_side),
                                    jpeg_quality=options.jpeg_quality,
                                    overwrite=overwrite,
                                    reuse_existing=options.reuse_existing_assets,
                                )
                                written_assets.add(reference_asset_for_case)
                            reference_relative = reference_asset_for_case.relative_to(root).as_posix()
                    current_asset = asset_dir / f"current_{frame_id:08d}.jpg"
                    current_source = (
                        current_asset
                        if options.reuse_existing_assets and current_asset.is_file()
                        else sequence.frames[frame_id]
                    )
                    current_image = image_cache.get(frame_id)
                    if current_image is None:
                        current_image = _read_rgb(current_source)
                        image_cache[frame_id] = current_image
                    presence = _presence(sequence, frame_id)
                    assert presence in {"present", "absent"}
                    bbox_norm: tuple[float, float, float, float] | None = None
                    if presence == "present":
                        current_bbox = _scale_bbox_to_image(
                            _raw_bbox(sequence, frame_id), current_image, source_size
                        )
                        if current_bbox is None:
                            # 少数框宽高有效但完全落在图外；同样不伪造语义标签。
                            stats.skipped_invalid_bbox += 1
                            continue
                        height, width = current_image.shape[:2]
                        bbox_norm = pixel_xywh_to_norm1000_xyxy(current_bbox, width, height)

                    if current_asset not in written_assets:
                        _write_rgb(
                            current_asset,
                            _resize_long_side(current_image, options.max_image_side),
                            jpeg_quality=options.jpeg_quality,
                            overwrite=overwrite,
                            reuse_existing=options.reuse_existing_assets,
                        )
                        written_assets.add(current_asset)
                    current_relative = current_asset.relative_to(root).as_posix()

                    contexts: list[tuple[str, str, str | None, list[int], PromptSpec, str | None]] = []
                    if options.mode in {"pair", "both"}:
                        contexts.append(
                            (
                                "pair",
                                "pair",
                                None,
                                [],
                                build_pair_prompt(
                                    target_text=target_text,
                                    reference_has_box=False,
                                    reference_bbox_norm1000_xyxy=reference_bbox_norm,
                                    include_memory_update=False,
                                ),
                                None,
                            )
                        )

                    if options.mode in {"mosaic", "both"}:
                        panels = _history_panels(
                            sequence,
                            frame_id,
                            options.history_size,
                            anchor_frame_id=reference_frame_id,
                            keyframes_only=options.keyframes_only,
                            existing_asset_dir=asset_dir if options.reuse_existing_assets else None,
                            candidate_frame_ids=planned_frame_ids,
                            image_cache=image_cache,
                            source_size=source_size,
                        )
                        if panels:
                            prompt = build_mosaic_prompt(
                                len(panels),
                                target_text=target_text,
                                reference_bbox_norm1000_xyxy=reference_bbox_norm,
                                include_memory_update=False,
                            )
                            variants: list[
                                tuple[
                                    list[
                                        tuple[
                                            int,
                                            np.ndarray,
                                            tuple[float, float, float, float],
                                        ]
                                    ],
                                    str | None,
                                ]
                            ] = [(panels, None)]
                            corruption_key = (
                                f"{sequence.dataset}::{sequence.name}::{reference_frame_id}::{frame_id}"
                            )
                            selector = int.from_bytes(
                                hashlib.sha256(corruption_key.encode("utf-8")).digest()[:8],
                                byteorder="big",
                            ) / float(1 << 64)
                            if selector < options.history_corruption_ratio:
                                corrupted_panels, corruption_mode = _corrupt_history_panels(
                                    panels,
                                    seed_key=corruption_key,
                                )
                                variants.append((corrupted_panels, corruption_mode))
                            for variant_panels, corruption in variants:
                                suffix = f"_corrupt_{corruption}" if corruption is not None else ""
                                history_asset = asset_dir / (
                                    f"history_{reference_frame_id:08d}_before_{frame_id:08d}{suffix}.jpg"
                                )
                                _write_rgb(
                                    history_asset,
                                    _resize_long_side(
                                        _build_mosaic(variant_panels, panel_height=options.mosaic_panel_height),
                                        options.max_image_side,
                                    ),
                                    jpeg_quality=options.jpeg_quality,
                                    overwrite=overwrite,
                                    reuse_existing=options.reuse_existing_assets,
                                )
                                history_relative = history_asset.relative_to(root).as_posix()
                                history_frame_ids = [item[0] for item in variant_panels]
                                contexts.append(
                                    (
                                        "mosaic",
                                        "mosaic",
                                        history_relative,
                                        history_frame_ids,
                                        prompt,
                                        corruption,
                                    )
                                )
                                stats.corrupted_mosaic_count += int(corruption is not None)
                        elif options.mode == "mosaic":
                            # 纯 mosaic 构建需要与在线推理保持一致：尚无可信历史时
                            # 退化为 pair。both 模式已经生成 pair，不再复制同一条样本。
                            contexts.append(
                                (
                                    "mosaic",
                                    "pair",
                                    None,
                                    [],
                                    build_pair_prompt(
                                        target_text=target_text,
                                        reference_has_box=False,
                                        reference_bbox_norm1000_xyxy=reference_bbox_norm,
                                        include_memory_update=False,
                                    ),
                                    None,
                                )
                            )

                    for requested_mode, effective_mode, history_path, history_ids, prompt, corruption in contexts:
                        row = _record(
                            sequence=sequence,
                            frame_id=frame_id,
                            reference_frame_id=reference_frame_id,
                            reference_bbox_norm1000_xyxy=reference_bbox_norm,
                            reference_path=reference_relative,
                            current_path=current_relative,
                            history_path=history_path,
                            history_frame_ids=history_ids,
                            prompt=prompt,
                            requested_mode=requested_mode,
                            effective_mode=effective_mode,
                            presence=presence,
                            bbox_norm1000_xyxy=bbox_norm,
                        )
                        row["metadata"]["source_split"] = sequence.metadata.get("split")
                        row["metadata"]["used_language_description"] = bool(target_text)
                        row["metadata"]["temporal_case"] = presence
                        row["metadata"]["history_corruption"] = corruption
                        if corruption is not None:
                            row["id"] = f"{row['id']}::{corruption}"
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        sequence_samples += 1
                        stats.sample_count += 1
                        stats.present_count += int(presence == "present")
                        stats.absent_count += int(presence == "absent")
                        stats.pair_count += int(effective_mode == "pair")
                        stats.mosaic_count += int(effective_mode == "mosaic")
                stats.sequences_with_samples += int(sequence_samples > 0)

        if stats.sequence_count == 0:
            raise ValueError("sequences 为空")
        if stats.sample_count == 0:
            raise ValueError("没有生成任何合法样本；请检查采样范围、可见性和 bbox 标注")
        os.replace(temporary_path, source_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    report = TrackingSampleBuildReport(
        schema_version=SOURCE_SCHEMA_VERSION,
        requested_mode=options.mode,
        balance_presence=options.balance_presence,
        present_only=options.present_only,
        use_language_description=options.use_language_description,
        max_image_side=options.max_image_side,
        sampling_plan_applied=frame_ids_by_sequence is not None,
        reference_mode="full_frame_bbox_text",
        source_jsonl=source_path.relative_to(root).as_posix(),
        image_root=".",
        sequence_count=stats.sequence_count,
        sequences_with_samples=stats.sequences_with_samples,
        sample_count=stats.sample_count,
        present_count=stats.present_count,
        absent_count=stats.absent_count,
        pair_count=stats.pair_count,
        mosaic_count=stats.mosaic_count,
        skipped_invalid_bbox=stats.skipped_invalid_bbox,
        skipped_unknown_presence=stats.skipped_unknown_presence,
        history_corruption_ratio=options.history_corruption_ratio,
        corrupted_mosaic_count=stats.corrupted_mosaic_count,
    )
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "SOURCE_SCHEMA_VERSION",
    "TrackingSampleBuildReport",
    "TrackingSampleConfig",
    "build_tracking_samples",
]
