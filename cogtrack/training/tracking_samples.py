"""从 pytracking ``Sequence`` 构建认知跟踪多模态训练样本。

该模块只消费标准序列对象，不感知具体数据集目录结构。输出由一个通用源
``JSONL`` 和自包含图片资产组成，随后可交给 ``tracking/export_swift_dataset.py``
转换为 SFT 或 GRPO 数据。

基础监督严格限制为数据集可直接提供的 ``present/absent`` 与 bbox。visual-v5 的
``memory_update`` 必须由调用方提供带来源的显式标签；本模块不会从普通 bbox 伪造
语义文本，也不生成身份、细粒度可见性、推理文本或置信度。present 但 bbox 无效的
帧会被跳过，而不会被错误解释为 absent。
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

from cogtrack.context.builder import (
    PROMPT_PROFILE_VISUAL_V5,
    PROMPT_PROFILE_VLT_V6,
    history_layout_for_prompt_profile,
    validate_prompt_profile,
)
from cogtrack.context.visual import (
    REFERENCE_MODE_BBOX_TEXT,
    REFERENCE_MODE_VISUAL_BOX,
    VISUAL_MARKER_VERSION,
    arrange_history_items,
    build_history_mosaic,
    draw_reference_box,
    validate_reference_mode,
)
from cogtrack.prompts import (
    PromptSpec,
    build_mosaic_prompt,
    build_pair_prompt,
    build_visual_tracking_prompt,
    build_vlt_tracking_prompt,
)
from cogtrack.protocol import BoundingBoxError, clip_xywh, pixel_xywh_to_norm1000_xyxy
from cogtrack.training.loss_mask import (
    SFT_SUPERVISION_FULL,
    SFT_SUPERVISION_TRACKING_CORE,
)
from pytracking.evaluation.data import Sequence

SOURCE_SCHEMA_VERSION = "cogtrack.training.source.v6"
SOURCE_SCHEMA_VERSION_V5 = "cogtrack.training.source.v5"
MEMORY_SUPERVISION_DISABLED = "disabled"
MEMORY_SUPERVISION_FEASIBILITY_NULL = "feasibility_null"
MEMORY_SUPERVISION_EXPLICIT = "explicit"
MEMORY_SUPERVISION_MASKED_NULL = "masked_null"
MEMORY_SUPERVISION_MODES = frozenset(
    {
        MEMORY_SUPERVISION_DISABLED,
        MEMORY_SUPERVISION_FEASIBILITY_NULL,
        MEMORY_SUPERVISION_EXPLICIT,
        MEMORY_SUPERVISION_MASKED_NULL,
    }
)


@dataclass(frozen=True)
class MemoryUpdateLabel:
    """一帧显式语义记忆监督及其可审计来源。"""

    value: str | None
    source: str
    reviewed: bool = False

    def __post_init__(self) -> None:
        if self.value is not None:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("非空 memory_update 标签必须是非空字符串")
            normalized = " ".join(self.value.split())
            if len(normalized) > 256 or len(normalized.split()) > 30:
                raise ValueError("memory_update 标签必须不超过 256 字符且不超过 30 个词")
            object.__setattr__(self, "value", normalized)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("memory label source 必须是非空字符串")
        if not isinstance(self.reviewed, bool):
            raise TypeError("memory label reviewed 必须是 bool")


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
    reference_mode: str = REFERENCE_MODE_BBOX_TEXT
    memory_supervision: str = MEMORY_SUPERVISION_DISABLED
    prompt_profile: str = PROMPT_PROFILE_VISUAL_V5
    force_history_image: bool = False

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
        object.__setattr__(self, "reference_mode", validate_reference_mode(self.reference_mode))
        object.__setattr__(self, "prompt_profile", validate_prompt_profile(self.prompt_profile))
        if (
            self.prompt_profile == PROMPT_PROFILE_VLT_V6
            and self.reference_mode != REFERENCE_MODE_VISUAL_BOX
        ):
            raise ValueError("vlt_v6 训练样本必须使用 reference_mode=visual_box")
        if not isinstance(self.force_history_image, bool):
            raise TypeError("force_history_image 必须是 bool")
        if self.force_history_image and self.mode not in {"mosaic", "both"}:
            raise ValueError("force_history_image 需要 mode=mosaic 或 both")
        if self.force_history_image and self.reference_mode != REFERENCE_MODE_VISUAL_BOX:
            raise ValueError("force_history_image 只支持 visual_box")
        if self.memory_supervision not in MEMORY_SUPERVISION_MODES:
            raise ValueError(
                f"memory_supervision 必须是 {sorted(MEMORY_SUPERVISION_MODES)} 之一"
            )
        if (
            self.reference_mode == REFERENCE_MODE_VISUAL_BOX
            and self.memory_supervision == MEMORY_SUPERVISION_DISABLED
        ):
            raise ValueError("visual_box v5 训练样本必须启用三字段 memory_update 监督")


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
    prompt_profile: str
    force_history_image: bool
    history_layout_version: str | None
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
    memory_supervision: str
    memory_null_count: int
    memory_non_null_count: int
    semantic_memory_input_count: int
    visual_marker_version: str | None

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
    memory_null_count: int = 0
    memory_non_null_count: int = 0
    semantic_memory_input_count: int = 0


def _safe_component(value: str) -> str:
    """生成稳定且无目录逃逸风险的资产目录名。"""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unnamed"
    if normalized == value:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def _initial_target_text(
    sequence: Sequence,
    *,
    enabled: bool,
    prompt_profile: str,
) -> tuple[str, str]:
    """返回初始化时可用的 VLT 文本及来源，不读取整段视频叙事。"""

    if not enabled:
        return "", "disabled"
    scope = str(sequence.metadata.get("language_scope") or "").strip().lower()
    description = str(sequence.language_query or "").strip()
    unsafe_story = scope == "full_video_story" or (
        sequence.dataset == "mgit" and scope != "initial_target"
    )
    if description and not unsafe_story:
        return description, "dataset_initial_language"
    object_class = str(sequence.object_class or "").strip()
    if object_class:
        return object_class, "dataset_object_class"
    if prompt_profile == PROMPT_PROFILE_VLT_V6:
        return "the target marked by the red box in Image 1", "visual_anchor_fallback"
    return "", "unavailable"


def _latest_prior_semantic_memory(
    *,
    memory_supervision: str,
    labels_by_sequence: Mapping[str, Mapping[int, Any]] | None,
    sequence_key: str,
    frame_id: int,
) -> str:
    """从严格早于 current 的已标注更新中选最近一条记忆。

    Phase-1 ``masked_null`` 没有可靠记忆标签，因此输入明确为空；不会把当前标签或
    未来确认帧泄漏进 Prompt。显式记忆阶段才启用这一回放路径。
    """

    if memory_supervision != MEMORY_SUPERVISION_EXPLICIT or labels_by_sequence is None:
        return ""
    frame_labels = labels_by_sequence.get(sequence_key, {})
    candidates: list[tuple[int, MemoryUpdateLabel]] = []
    for candidate_frame, raw_label in frame_labels.items():
        candidate_id = int(candidate_frame)
        if candidate_id >= frame_id:
            continue
        label = _coerce_memory_label(raw_label)
        if label.value:
            candidates.append((candidate_id, label))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1].value or ""


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
    memory_supervision: str,
    memory_label: MemoryUpdateLabel | None,
) -> dict[str, Any]:
    if presence == "present":
        if bbox_norm1000_xyxy is None:
            raise ValueError("present 训练样本必须包含有效的 norm1000 bbox")
        answer: dict[str, Any] = {
            "target_status": "present",
            "bbox_norm1000_xyxy": list(bbox_norm1000_xyxy) if bbox_norm1000_xyxy else None,
        }
    else:
        answer = {
            "target_status": "absent",
            "bbox_norm1000_xyxy": None,
        }

    if memory_supervision == MEMORY_SUPERVISION_DISABLED:
        if memory_label is not None:
            raise ValueError("memory_supervision=disabled 时不能传入 memory label")
        return answer
    if memory_label is None:
        raise ValueError("三字段样本必须有显式 MemoryUpdateLabel")
    if presence == "absent" and memory_label.value is not None:
        raise ValueError("absent 样本的 memory_update 必须为 null")
    answer["memory_update"] = memory_label.value
    return answer


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
    reference_mode: str,
    memory_supervision: str,
    memory_label: MemoryUpdateLabel | None,
    target_text: str,
    target_text_source: str,
    semantic_memory: str,
    prompt_profile: str,
    force_history_image: bool,
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
        memory_supervision=memory_supervision,
        memory_label=memory_label,
    )
    row = {
        "schema_version": (
            SOURCE_SCHEMA_VERSION
            if prompt_profile == PROMPT_PROFILE_VLT_V6
            else SOURCE_SCHEMA_VERSION_V5
        ),
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
            "reference_mode": reference_mode,
            "visual_marker_version": (
                VISUAL_MARKER_VERSION if reference_mode == REFERENCE_MODE_VISUAL_BOX else None
            ),
            "history_frame_ids": history_frame_ids,
            "history_layout_version": (
                history_layout_for_prompt_profile(prompt_profile)
                if effective_mode == "mosaic"
                else None
            ),
            "requested_mode": requested_mode,
            "effective_mode": effective_mode,
            "prompt_name": prompt.name,
            "prompt_version": prompt.version,
            "prompt_profile": prompt_profile,
            "bbox_format": "norm1000_xyxy",
            "memory_supervision": memory_supervision,
            "memory_label_source": memory_label.source if memory_label is not None else None,
            "memory_label_reviewed": memory_label.reviewed if memory_label is not None else None,
            "memory_loss_masked": memory_supervision == MEMORY_SUPERVISION_MASKED_NULL,
            "sft_supervision_profile": (
                SFT_SUPERVISION_TRACKING_CORE
                if memory_supervision == MEMORY_SUPERVISION_MASKED_NULL
                else SFT_SUPERVISION_FULL
            ),
            "initial_target_text": target_text,
            "initial_target_text_source": target_text_source,
            "recent_semantic_memory": semantic_memory,
            # VLT-v6.3 的正式命名。保留上面两个旧键，确保已经生成的 v6 数据仍可重放。
            "initial_identity_description": target_text,
            "initial_identity_description_source": target_text_source,
            "current_target_state": semantic_memory or target_text,
            "current_target_state_source": (
                "accepted_memory" if semantic_memory else target_text_source
            ),
            "uses_semantic_memory_input": bool(semantic_memory),
            "force_history_image": force_history_image,
        },
    }
    if "memory_update" in answer:
        row["memory_update"] = answer["memory_update"]
    return row


def _coerce_memory_label(value: Any) -> MemoryUpdateLabel:
    if isinstance(value, MemoryUpdateLabel):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("显式 memory label 必须是 MemoryUpdateLabel 或 mapping")
    if "memory_update" not in value:
        raise ValueError("显式 memory label mapping 缺少 memory_update")
    return MemoryUpdateLabel(
        value=value.get("memory_update"),
        source=str(value.get("source", "")),
        reviewed=value.get("reviewed", False),
    )


def load_memory_labels_jsonl(
    path: str | Path,
) -> dict[str, dict[int, MemoryUpdateLabel]]:
    """读取可审计的逐帧 ``memory_update`` 标签。

    JSONL 每行对应一个 current frame，必须包含 ``dataset``、``sequence``、
    ``frame_id``、``memory_update`` 和非空 ``source``。把读取逻辑放在训练模块而非
    某个 CLI 中，保证单数据集 dry-run 与多数据集正式构建执行完全相同的校验。
    """

    label_path = Path(path).expanduser().resolve()
    labels: dict[str, dict[int, MemoryUpdateLabel]] = {}
    with label_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"memory label JSONL 解析失败：{label_path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"memory label 每行必须是对象：{label_path}:{line_no}")
            dataset = str(row.get("dataset", row.get("source_dataset", ""))).strip()
            sequence = str(row.get("sequence", row.get("source_sequence", ""))).strip()
            if not dataset or not sequence or "frame_id" not in row or "memory_update" not in row:
                raise ValueError(
                    "memory label 缺少 dataset/sequence/frame_id/memory_update："
                    f"{label_path}:{line_no}"
                )
            frame_id = int(row["frame_id"])
            if frame_id < 0:
                raise ValueError(f"memory label frame_id 不能为负数：{label_path}:{line_no}")
            key = f"{dataset}::{sequence}"
            per_sequence = labels.setdefault(key, {})
            if frame_id in per_sequence:
                raise ValueError(f"重复 memory label：{key} frame={frame_id}")
            per_sequence[frame_id] = MemoryUpdateLabel(
                value=row["memory_update"],
                source=str(row.get("source", "")),
                reviewed=row.get("reviewed", False),
            )
    if not labels:
        raise ValueError(f"memory label 文件为空：{label_path}")
    return labels


def _memory_label_for_frame(
    *,
    memory_supervision: str,
    labels_by_sequence: Mapping[str, Mapping[int, Any]] | None,
    sequence_key: str,
    frame_id: int,
) -> MemoryUpdateLabel | None:
    if memory_supervision == MEMORY_SUPERVISION_DISABLED:
        return None
    if memory_supervision == MEMORY_SUPERVISION_FEASIBILITY_NULL:
        return MemoryUpdateLabel(
            value=None,
            source="feasibility_null_only_not_for_formal_training",
            reviewed=False,
        )
    if memory_supervision == MEMORY_SUPERVISION_MASKED_NULL:
        return MemoryUpdateLabel(
            value=None,
            source="masked_null_unsupervised_memory_v1",
            reviewed=False,
        )
    if labels_by_sequence is None or sequence_key not in labels_by_sequence:
        raise ValueError(f"显式 memory 监督缺少序列标签：{sequence_key}")
    frame_labels = labels_by_sequence[sequence_key]
    if frame_id not in frame_labels:
        raise ValueError(f"显式 memory 监督缺少帧标签：{sequence_key} frame={frame_id}")
    return _coerce_memory_label(frame_labels[frame_id])


def build_tracking_samples(
    sequences: Iterable[Sequence],
    output_dir: str | Path,
    *,
    config: TrackingSampleConfig | None = None,
    overwrite: bool = False,
    frame_ids_by_sequence: Mapping[str, TypingSequence[int]] | None = None,
    anchor_frame_ids_by_sequence: Mapping[str, int] | None = None,
    reference_frame_ids_by_sequence: Mapping[str, TypingSequence[int]] | None = None,
    memory_labels_by_sequence: Mapping[str, Mapping[int, Any]] | None = None,
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
        memory_labels_by_sequence: ``memory_supervision=explicit`` 时必填；键为
            ``dataset::sequence``，内层键为 current frame_id。每条标签必须同时记录
            ``memory_update``、``source`` 和可选 ``reviewed``。
    """

    options = config or TrackingSampleConfig()
    if options.memory_supervision == MEMORY_SUPERVISION_EXPLICIT and memory_labels_by_sequence is None:
        raise ValueError("memory_supervision=explicit 必须提供 memory_labels_by_sequence")
    if options.memory_supervision != MEMORY_SUPERVISION_EXPLICIT and memory_labels_by_sequence is not None:
        raise ValueError("只有 memory_supervision=explicit 才能提供 memory_labels_by_sequence")
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
                reference_prefix = (
                    "reference_boxed"
                    if options.reference_mode == REFERENCE_MODE_VISUAL_BOX
                    else "reference"
                )
                reference_asset = asset_dir / f"{reference_prefix}_{anchor_frame_id:08d}.jpg"
                rendered_anchor = (
                    draw_reference_box(anchor_image, anchor_bbox)
                    if options.reference_mode == REFERENCE_MODE_VISUAL_BOX
                    else anchor_image
                )
                _write_rgb(
                    reference_asset,
                    _resize_long_side(
                        rendered_anchor,
                        options.max_image_side,
                    ),
                    jpeg_quality=options.jpeg_quality,
                    overwrite=overwrite,
                    reuse_existing=options.reuse_existing_assets,
                )
                image_cache[anchor_frame_id] = anchor_image
                written_assets.add(reference_asset)
                anchor_reference_relative = reference_asset.relative_to(root).as_posix()
                target_text, target_text_source = _initial_target_text(
                    sequence,
                    enabled=options.use_language_description,
                    prompt_profile=options.prompt_profile,
                )

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
                    reference_relative = anchor_reference_relative
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
                            reference_asset_for_case = asset_dir / (
                                f"{reference_prefix}_{reference_frame_id:08d}.jpg"
                            )
                            reference_image = image_cache.get(reference_frame_id)
                            if reference_image is None:
                                # 即使复用输出资产，也必须从未画框源帧读取；否则 visual_box
                                # 重放会在已有框上二次绘制，破坏数据确定性。
                                reference_image = _read_rgb(sequence.frames[reference_frame_id])
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
                                rendered_reference = (
                                    draw_reference_box(reference_image, reference_bbox)
                                    if options.reference_mode == REFERENCE_MODE_VISUAL_BOX
                                    else reference_image
                                )
                                _write_rgb(
                                    reference_asset_for_case,
                                    _resize_long_side(rendered_reference, options.max_image_side),
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
                    memory_label = _memory_label_for_frame(
                        memory_supervision=options.memory_supervision,
                        labels_by_sequence=memory_labels_by_sequence,
                        sequence_key=plan_key,
                        frame_id=frame_id,
                    )
                    semantic_memory = _latest_prior_semantic_memory(
                        memory_supervision=options.memory_supervision,
                        labels_by_sequence=memory_labels_by_sequence,
                        sequence_key=plan_key,
                        frame_id=frame_id,
                    )
                    include_memory_update = options.memory_supervision != MEMORY_SUPERVISION_DISABLED

                    contexts: list[tuple[str, str, str | None, list[int], PromptSpec, str | None]] = []
                    if options.mode in {"pair", "both"}:
                        if options.reference_mode == REFERENCE_MODE_VISUAL_BOX:
                            prompt_builder = (
                                build_vlt_tracking_prompt
                                if options.prompt_profile == PROMPT_PROFILE_VLT_V6
                                else build_visual_tracking_prompt
                            )
                            pair_prompt = prompt_builder(
                                history_count=0,
                                target_text=target_text,
                                semantic_memory=semantic_memory,
                                include_memory_update=include_memory_update,
                            )
                        else:
                            pair_prompt = build_pair_prompt(
                                target_text=target_text,
                                reference_has_box=False,
                                reference_bbox_norm1000_xyxy=reference_bbox_norm,
                                include_memory_update=include_memory_update,
                            )
                        contexts.append(
                            (
                                "pair",
                                "pair",
                                None,
                                [],
                                pair_prompt,
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
                            # 当前 VLT 主线固定为左到右的近期三帧。少于三帧时在右侧
                            # 复制最近可用历史；visual-v5 继续保留旧紧凑网格。
                            panels = list(
                                arrange_history_items(
                                    panels,
                                    layout=history_layout_for_prompt_profile(
                                        options.prompt_profile
                                    ),
                                )
                            )
                            if options.reference_mode == REFERENCE_MODE_VISUAL_BOX:
                                prompt_builder = (
                                    build_vlt_tracking_prompt
                                    if options.prompt_profile == PROMPT_PROFILE_VLT_V6
                                    else build_visual_tracking_prompt
                                )
                                prompt = prompt_builder(
                                    history_count=len(panels),
                                    target_text=target_text,
                                    semantic_memory=semantic_memory,
                                    include_memory_update=include_memory_update,
                                )
                            else:
                                prompt = build_mosaic_prompt(
                                    len(panels),
                                    target_text=target_text,
                                    reference_bbox_norm1000_xyxy=reference_bbox_norm,
                                    include_memory_update=include_memory_update,
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
                                        build_history_mosaic(
                                            [(item[1], item[2]) for item in variant_panels],
                                            panel_height=options.mosaic_panel_height,
                                            layout=history_layout_for_prompt_profile(
                                                options.prompt_profile
                                            ),
                                        ),
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
                        elif options.force_history_image:
                            # 最早观测尚无动态历史时，用初始化锚点填满三格。它严格早于
                            # current；重复格只是 padding，不伪造三次预测。
                            fallback_panels = arrange_history_items(
                                ((anchor_frame_id, anchor_image, anchor_bbox),),
                                layout=history_layout_for_prompt_profile(
                                    options.prompt_profile
                                ),
                            )
                            history_asset = asset_dir / (
                                f"history_{reference_frame_id:08d}_before_{frame_id:08d}"
                                "_anchor_fallback.jpg"
                            )
                            _write_rgb(
                                history_asset,
                                _resize_long_side(
                                    build_history_mosaic(
                                        tuple(
                                            (item[1], item[2]) for item in fallback_panels
                                        ),
                                        panel_height=options.mosaic_panel_height,
                                        layout=history_layout_for_prompt_profile(
                                            options.prompt_profile
                                        ),
                                    ),
                                    options.max_image_side,
                                ),
                                jpeg_quality=options.jpeg_quality,
                                overwrite=overwrite,
                                reuse_existing=options.reuse_existing_assets,
                            )
                            prompt = build_vlt_tracking_prompt(
                                history_count=len(fallback_panels),
                                target_text=target_text,
                                semantic_memory=semantic_memory,
                                include_memory_update=include_memory_update,
                            )
                            contexts.append(
                                (
                                    "mosaic",
                                    "mosaic",
                                    history_asset.relative_to(root).as_posix(),
                                    [item[0] for item in fallback_panels],
                                    prompt,
                                    None,
                                )
                            )
                        elif options.mode == "mosaic":
                            # 纯 mosaic 构建需要与在线推理保持一致：尚无可信历史时
                            # 退化为 pair。both 模式已经生成 pair，不再复制同一条样本。
                            if options.reference_mode == REFERENCE_MODE_VISUAL_BOX:
                                prompt_builder = (
                                    build_vlt_tracking_prompt
                                    if options.prompt_profile == PROMPT_PROFILE_VLT_V6
                                    else build_visual_tracking_prompt
                                )
                                fallback_prompt = prompt_builder(
                                    history_count=0,
                                    target_text=target_text,
                                    semantic_memory=semantic_memory,
                                    include_memory_update=include_memory_update,
                                )
                            else:
                                fallback_prompt = build_pair_prompt(
                                    target_text=target_text,
                                    reference_has_box=False,
                                    reference_bbox_norm1000_xyxy=reference_bbox_norm,
                                    include_memory_update=include_memory_update,
                                )
                            contexts.append(
                                (
                                    "mosaic",
                                    "pair",
                                    None,
                                    [],
                                    fallback_prompt,
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
                            reference_mode=options.reference_mode,
                            memory_supervision=options.memory_supervision,
                            memory_label=memory_label,
                            target_text=target_text,
                            target_text_source=target_text_source,
                            semantic_memory=semantic_memory,
                            prompt_profile=options.prompt_profile,
                            force_history_image=options.force_history_image,
                        )
                        row["metadata"]["source_split"] = sequence.metadata.get("split")
                        row["metadata"]["uses_initial_target_text"] = bool(target_text)
                        row["metadata"]["used_language_description"] = (
                            target_text_source == "dataset_initial_language"
                        )
                        row["metadata"]["temporal_case"] = presence
                        row["metadata"]["history_corruption"] = corruption
                        row["metadata"]["history_anchor_fallback"] = bool(
                            options.force_history_image
                            and effective_mode == "mosaic"
                            and bool(history_ids)
                            and set(history_ids) == {anchor_frame_id}
                        )
                        if corruption is not None:
                            row["id"] = f"{row['id']}::{corruption}"
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        sequence_samples += 1
                        stats.sample_count += 1
                        stats.present_count += int(presence == "present")
                        stats.absent_count += int(presence == "absent")
                        stats.pair_count += int(effective_mode == "pair")
                        stats.mosaic_count += int(effective_mode == "mosaic")
                        if memory_label is not None:
                            stats.memory_null_count += int(memory_label.value is None)
                            stats.memory_non_null_count += int(memory_label.value is not None)
                        stats.semantic_memory_input_count += int(bool(semantic_memory))
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
        schema_version=(
            SOURCE_SCHEMA_VERSION
            if options.prompt_profile == PROMPT_PROFILE_VLT_V6
            else SOURCE_SCHEMA_VERSION_V5
        ),
        requested_mode=options.mode,
        balance_presence=options.balance_presence,
        present_only=options.present_only,
        use_language_description=options.use_language_description,
        max_image_side=options.max_image_side,
        sampling_plan_applied=frame_ids_by_sequence is not None,
        reference_mode=options.reference_mode,
        prompt_profile=options.prompt_profile,
        force_history_image=options.force_history_image,
        history_layout_version=(
            history_layout_for_prompt_profile(options.prompt_profile)
            if options.mode in {"mosaic", "both"}
            else None
        ),
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
        memory_supervision=options.memory_supervision,
        memory_null_count=stats.memory_null_count,
        memory_non_null_count=stats.memory_non_null_count,
        semantic_memory_input_count=stats.semantic_memory_input_count,
        visual_marker_version=(
            VISUAL_MARKER_VERSION if options.reference_mode == REFERENCE_MODE_VISUAL_BOX else None
        ),
    )
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "MEMORY_SUPERVISION_DISABLED",
    "MEMORY_SUPERVISION_EXPLICIT",
    "MEMORY_SUPERVISION_FEASIBILITY_NULL",
    "MEMORY_SUPERVISION_MASKED_NULL",
    "MEMORY_SUPERVISION_MODES",
    "MemoryUpdateLabel",
    "SOURCE_SCHEMA_VERSION",
    "TrackingSampleBuildReport",
    "TrackingSampleConfig",
    "build_tracking_samples",
    "load_memory_labels_jsonl",
]
