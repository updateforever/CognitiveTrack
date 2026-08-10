"""构建候选级、同类别跨序列身份困难负样本。

基础 SOT 标注只能可靠地产生 ``same`` 与 ``not_applicable``。本模块只在两个
不同物理实例组具有相同且显式的 ``object_class`` 时构造 ``different`` 监督，
类别缺失、初始化框无效或实例来源可能相同的情况一律跳过或报错，绝不根据
序列名称猜测类别。

每个实例最多进入一个确定性配对组；组内可生成双向 reference/candidate 样本。
最终 JSONL 的 ``metadata.source_sequence`` 固定为配对组 ID，可直接复用现有
按序列划分器，并保证本身份负样本数据集内部不会把同一实例分到多个 split。

该模块是隔离的未来研究工具，不属于 CognitiveTrack v4 的 presence + memory
SFT/GRPO 数据入口。其输出会被当前主协议校验器主动拒绝，避免身份辅助标签
误混入二分类训练集。
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
from typing import Any, Iterable
from typing import Sequence as TypingSequence

import cv2
import numpy as np

from cogtrack.prompts import (
    CANDIDATE_IDENTITY_PROMPT_NAME,
    CANDIDATE_IDENTITY_PROMPT_VERSION,
    build_candidate_identity_prompt,
)
from cogtrack.protocol import BoundingBoxError, clip_xywh, pixel_xywh_to_norm1000_xyxy
from pytracking.evaluation.data import Sequence

IDENTITY_SOURCE_SCHEMA_VERSION = "cogtrack.training.identity.source.v2"
IDENTITY_PROMPT_NAME = CANDIDATE_IDENTITY_PROMPT_NAME
IDENTITY_PROMPT_VERSION = CANDIDATE_IDENTITY_PROMPT_VERSION
LABEL_SOURCE = "cross_sequence_same_class"


@dataclass(frozen=True)
class IdentitySampleConfig:
    """身份困难负样本构建参数。"""

    frame_stride: int = 1
    max_candidate_frames: int | None = 8
    seed: int = 20260805
    bidirectional: bool = True
    keyframes_only: bool = False
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if isinstance(self.frame_stride, bool) or self.frame_stride <= 0:
            raise ValueError("frame_stride 必须是正整数")
        if self.max_candidate_frames is not None:
            if isinstance(self.max_candidate_frames, bool) or self.max_candidate_frames <= 0:
                raise ValueError("max_candidate_frames 必须为正整数或 None")
        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional 必须是 bool")
        if not isinstance(self.keyframes_only, bool):
            raise TypeError("keyframes_only 必须是 bool")
        if isinstance(self.jpeg_quality, bool) or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality 必须位于 [1, 100]")


@dataclass(frozen=True)
class IdentitySampleBuildReport:
    """构建统计；缺失类别和不安全标注均可追溯。"""

    schema_version: str
    label_source: str
    source_jsonl: str
    image_root: str
    input_sequence_count: int
    eligible_instance_count: int
    missing_class_count: int
    invalid_reference_count: int
    duplicate_alias_count: int
    unpaired_instance_count: int
    pair_group_count: int
    sample_count: int
    skipped_absent_candidate_frames: int
    skipped_invalid_candidate_bbox: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _InstanceEntry:
    sequence: Sequence
    class_key: str
    class_label: str
    instance_key: tuple[str, str]


@dataclass
class _MutableStats:
    input_sequence_count: int = 0
    eligible_instance_count: int = 0
    missing_class_count: int = 0
    invalid_reference_count: int = 0
    duplicate_alias_count: int = 0
    unpaired_instance_count: int = 0
    pair_group_count: int = 0
    sample_count: int = 0
    skipped_absent_candidate_frames: int = 0
    skipped_invalid_candidate_bbox: int = 0


def _safe_component(value: str) -> str:
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


def _write_rgb(path: Path, image: np.ndarray, *, jpeg_quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
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


def _clip_bbox(
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


def _draw_box(
    image: np.ndarray,
    bbox_xywh: TypingSequence[float],
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    """在 RGB 副本上绘框；参考目标用绿色，候选目标用红色。"""

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


def _class_info(sequence: Sequence) -> tuple[str, str] | None:
    """只接受显式 object_class，不从名称或自然语言推断类别。"""

    if not isinstance(sequence.object_class, str):
        return None
    label = sequence.object_class.strip()
    if not label:
        return None
    return label.casefold(), label


def _instance_key(sequence: Sequence) -> tuple[str, str]:
    """解析物理实例来源，识别包装数据集中的同源序列别名。"""

    metadata = sequence.metadata if isinstance(sequence.metadata, dict) else {}
    source_dataset = str(metadata.get("source_dataset") or sequence.dataset).strip().casefold()
    source_sequence = str(metadata.get("source_sequence") or sequence.name).strip()
    if not source_dataset or not source_sequence:
        raise ValueError(f"序列 {sequence.dataset}/{sequence.name} 缺少可追溯实例来源")
    return source_dataset, source_sequence


def _collect_instances(sequences: Iterable[Sequence], stats: _MutableStats) -> list[_InstanceEntry]:
    by_instance: dict[tuple[str, str], _InstanceEntry] = {}
    observed_sequences: dict[tuple[str, str], tuple[str, str]] = {}
    for sequence in sequences:
        if not isinstance(sequence, Sequence):
            raise TypeError("sequences 中的元素必须是 pytracking.evaluation.data.Sequence")
        stats.input_sequence_count += 1
        class_info = _class_info(sequence)
        if class_info is None:
            stats.missing_class_count += 1
            continue
        if sequence.ground_truth_rect is None or _raw_bbox(sequence, 0) is None:
            stats.invalid_reference_count += 1
            continue
        if sequence.target_visible is not None and not bool(sequence.target_visible[0]):
            stats.invalid_reference_count += 1
            continue

        class_key, class_label = class_info
        instance_key = _instance_key(sequence)
        observed_key = (sequence.dataset.casefold(), sequence.name)
        previous_source = observed_sequences.get(observed_key)
        if previous_source is not None and previous_source != instance_key:
            raise ValueError(
                f"同一序列 {sequence.dataset}/{sequence.name} 声明了冲突实例来源: "
                f"{previous_source} 与 {instance_key}"
            )
        observed_sequences[observed_key] = instance_key
        entry = _InstanceEntry(sequence, class_key, class_label, instance_key)
        existing = by_instance.get(instance_key)
        if existing is not None:
            if existing.class_key != class_key:
                raise ValueError(
                    f"同一实例来源 {instance_key} 出现冲突类别: "
                    f"{existing.class_label!r} 与 {class_label!r}"
                )
            # 同源别名不能互相充当 different。稳定保留 dataset/name 更小者。
            stats.duplicate_alias_count += 1
            choices = sorted(
                (existing, entry),
                key=lambda item: (item.sequence.dataset, item.sequence.name),
            )
            by_instance[instance_key] = choices[0]
            continue
        by_instance[instance_key] = entry

    entries = list(by_instance.values())
    stats.eligible_instance_count = len(entries)
    return entries


def _class_pairs(
    entries: list[_InstanceEntry],
    *,
    seed: int,
    stats: _MutableStats,
) -> list[tuple[_InstanceEntry, _InstanceEntry]]:
    # 不跨来源数据集混用类别词表：即使两个数据集都写 ``car``，标注粒度也可能
    # 不一致。只有 canonical dataset 和显式类别同时相同才允许配对。
    grouped: dict[tuple[str, str], list[_InstanceEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.instance_key[0], entry.class_key), []).append(entry)

    pairs: list[tuple[_InstanceEntry, _InstanceEntry]] = []
    for source_dataset, class_key in sorted(grouped):
        values = sorted(
            grouped[(source_dataset, class_key)],
            key=lambda item: (item.instance_key, item.sequence.dataset, item.sequence.name),
        )
        seed_material = f"{seed}\0{source_dataset}\0{class_key}".encode("utf-8")
        class_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="big")
        random.Random(class_seed).shuffle(values)
        stats.unpaired_instance_count += len(values) % 2
        for index in range(0, len(values) - 1, 2):
            first, second = values[index], values[index + 1]
            if first.instance_key == second.instance_key:
                raise ValueError("内部错误：同一实例不能被标注为 different")
            if (
                first.sequence.dataset.casefold(),
                first.sequence.name,
            ) == (
                second.sequence.dataset.casefold(),
                second.sequence.name,
            ):
                raise ValueError("内部错误：同一序列不能被标注为 different")
            pairs.append((first, second))
    return pairs


def _stable_sample(
    frame_ids: list[int],
    count: int | None,
    *,
    candidate: _InstanceEntry,
    seed: int,
) -> list[int]:
    if count is None or len(frame_ids) <= count:
        return frame_ids
    material = (
        f"{seed}\0{candidate.instance_key[0]}\0{candidate.instance_key[1]}"
    ).encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big")
    return sorted(random.Random(stable_seed).sample(frame_ids, count))


def _candidate_frame_ids(
    entry: _InstanceEntry,
    config: IdentitySampleConfig,
    stats: _MutableStats,
) -> list[int]:
    sequence = entry.sequence
    if config.keyframes_only:
        source_ids = sorted(frame_id for frame_id in sequence.keyframe_indices if frame_id > 0)
        source_ids = source_ids[:: config.frame_stride]
    else:
        source_ids = list(range(1, len(sequence), config.frame_stride))

    valid: list[int] = []
    for frame_id in source_ids:
        if sequence.target_visible is not None and not bool(sequence.target_visible[frame_id]):
            stats.skipped_absent_candidate_frames += 1
            continue
        if _raw_bbox(sequence, frame_id) is None:
            stats.skipped_invalid_candidate_bbox += 1
            continue
        valid.append(frame_id)
    return _stable_sample(
        valid,
        config.max_candidate_frames,
        candidate=entry,
        seed=config.seed,
    )


def _pair_group_id(first: _InstanceEntry, second: _InstanceEntry) -> str:
    members = sorted((first.instance_key, second.instance_key))
    serialized = json.dumps(members, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"identity-{_safe_component(first.class_key)}-{digest}"


def _answer(
    bbox_norm1000_xyxy: tuple[float, float, float, float],
    target_text: str,
) -> dict[str, Any]:
    return {
        "target_presence": "present",
        "identity_match": "different",
        "localizability": "localizable",
        "bbox_norm1000_xyxy": list(bbox_norm1000_xyxy),
        "target_text": target_text,
        "reasoning": "The marked candidate is visible but is a different instance of the same class.",
    }


def _record(
    *,
    reference: _InstanceEntry,
    candidate: _InstanceEntry,
    candidate_frame_id: int,
    group_id: str,
    reference_path: str,
    candidate_path: str,
    bbox_norm1000_xyxy: tuple[float, float, float, float],
) -> dict[str, Any]:
    if reference.instance_key == candidate.instance_key:
        raise ValueError("同一物理实例不能构造 different 监督")
    if reference.class_key != candidate.class_key:
        raise ValueError("身份困难负样本必须来自显式相同 object_class")

    target_text = (
        reference.sequence.language_query
        or reference.sequence.object_class
        or reference.class_label
    ).strip()
    prompt = build_candidate_identity_prompt(target_text)
    answer = _answer(bbox_norm1000_xyxy, target_text)
    reference_member = f"{reference.instance_key[0]}::{reference.instance_key[1]}"
    candidate_member = f"{candidate.instance_key[0]}::{candidate.instance_key[1]}"
    return {
        "schema_version": IDENTITY_SOURCE_SCHEMA_VERSION,
        "id": (
            f"{group_id}::{reference.sequence.dataset}::{reference.sequence.name}::"
            f"{candidate.sequence.dataset}::{candidate.sequence.name}::{candidate_frame_id:08d}"
        ),
        "task": "candidate_identity_verification",
        "system_prompt": prompt.system_prompt,
        "user_prompt": "<image><image>\n" + prompt.user_prompt,
        "assistant": answer,
        "images": [reference_path, candidate_path],
        "target_presence": "present",
        "identity_match": "different",
        "localizability": "localizable",
        "bbox_norm1000_xyxy": answer["bbox_norm1000_xyxy"],
        "bbox_format": "norm1000_xyxy",
        "metadata": {
            # 现有 exporter 使用 source_sequence 分组；每个实例只属于一个配对组。
            "dataset": "identity_cross_sequence",
            "source_dataset": "identity_cross_sequence",
            "sequence": group_id,
            "source_sequence": group_id,
            "split_group": group_id,
            "group_members": sorted((reference_member, candidate_member)),
            "label_source": LABEL_SOURCE,
            "object_class": reference.class_label,
            "object_class_normalized": reference.class_key,
            "reference_dataset": reference.sequence.dataset,
            "reference_sequence": reference.sequence.name,
            "reference_instance_group": reference_member,
            "reference_frame_id": 0,
            "candidate_dataset": candidate.sequence.dataset,
            "candidate_sequence": candidate.sequence.name,
            "candidate_instance_group": candidate_member,
            "candidate_frame_id": candidate_frame_id,
            "prompt_name": IDENTITY_PROMPT_NAME,
            "prompt_version": IDENTITY_PROMPT_VERSION,
            "bbox_format": "norm1000_xyxy",
        },
    }


def build_identity_samples(
    sequences: Iterable[Sequence],
    output_dir: str | Path,
    *,
    config: IdentitySampleConfig | None = None,
    overwrite: bool = False,
) -> IdentitySampleBuildReport:
    """构建候选级身份困难负样本及自包含图片资产。"""

    options = config or IdentitySampleConfig()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "identity_source_samples.jsonl"
    report_path = root / "identity_build_report.json"
    if not overwrite and (source_path.exists() or report_path.exists()):
        raise FileExistsError(f"输出已存在；如需重建请启用 overwrite: {root}")

    stats = _MutableStats()
    entries = _collect_instances(sequences, stats)
    pairs = _class_pairs(entries, seed=options.seed, stats=stats)
    if not pairs:
        raise ValueError(
            "没有可安全配对的同类别不同实例：需要至少两个具有显式相同 "
            f"object_class 的不同序列；missing_class={stats.missing_class_count}, "
            f"eligible_instances={stats.eligible_instance_count}"
        )

    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=root,
        prefix=".identity_source_samples.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    try:
        with temporary_handle as handle:
            for first, second in pairs:
                group_id = _pair_group_id(first, second)
                directions = [(first, second)]
                if options.bidirectional:
                    directions.append((second, first))
                group_samples = 0
                for reference, candidate in directions:
                    frame_ids = _candidate_frame_ids(candidate, options, stats)
                    if not frame_ids:
                        continue

                    reference_image = _read_rgb(reference.sequence.frames[0])
                    reference_bbox = _clip_bbox(_raw_bbox(reference.sequence, 0), reference_image)
                    if reference_bbox is None:
                        raise ValueError(
                            f"参考框无法落入图像: {reference.sequence.dataset}/{reference.sequence.name}@0"
                        )
                    asset_dir = root / "identity_images" / group_id
                    reference_stem = (
                        f"reference_{_safe_component(reference.sequence.dataset)}_"
                        f"{_safe_component(reference.sequence.name)}"
                    )
                    reference_asset = asset_dir / f"{reference_stem}.jpg"
                    _write_rgb(
                        reference_asset,
                        _draw_box(reference_image, reference_bbox, color=(0, 255, 0)),
                        jpeg_quality=options.jpeg_quality,
                        overwrite=overwrite,
                    )
                    reference_relative = reference_asset.relative_to(root).as_posix()

                    for frame_id in frame_ids:
                        candidate_image = _read_rgb(candidate.sequence.frames[frame_id])
                        candidate_bbox = _clip_bbox(_raw_bbox(candidate.sequence, frame_id), candidate_image)
                        if candidate_bbox is None:
                            stats.skipped_invalid_candidate_bbox += 1
                            continue
                        height, width = candidate_image.shape[:2]
                        bbox_norm = pixel_xywh_to_norm1000_xyxy(candidate_bbox, width, height)
                        candidate_stem = (
                            f"candidate_{_safe_component(candidate.sequence.dataset)}_"
                            f"{_safe_component(candidate.sequence.name)}_{frame_id:08d}.jpg"
                        )
                        candidate_asset = asset_dir / candidate_stem
                        _write_rgb(
                            candidate_asset,
                            _draw_box(candidate_image, candidate_bbox, color=(255, 0, 0)),
                            jpeg_quality=options.jpeg_quality,
                            overwrite=overwrite,
                        )
                        row = _record(
                            reference=reference,
                            candidate=candidate,
                            candidate_frame_id=frame_id,
                            group_id=group_id,
                            reference_path=reference_relative,
                            candidate_path=candidate_asset.relative_to(root).as_posix(),
                            bbox_norm1000_xyxy=bbox_norm,
                        )
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        group_samples += 1
                        stats.sample_count += 1
                stats.pair_group_count += int(group_samples > 0)

        if stats.sample_count == 0:
            raise ValueError("配对成功，但候选序列没有任何可安全使用的可见有效框")
        os.replace(temporary_path, source_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    report = IdentitySampleBuildReport(
        schema_version=IDENTITY_SOURCE_SCHEMA_VERSION,
        label_source=LABEL_SOURCE,
        source_jsonl=source_path.relative_to(root).as_posix(),
        image_root=".",
        input_sequence_count=stats.input_sequence_count,
        eligible_instance_count=stats.eligible_instance_count,
        missing_class_count=stats.missing_class_count,
        invalid_reference_count=stats.invalid_reference_count,
        duplicate_alias_count=stats.duplicate_alias_count,
        unpaired_instance_count=stats.unpaired_instance_count,
        pair_group_count=stats.pair_group_count,
        sample_count=stats.sample_count,
        skipped_absent_candidate_frames=stats.skipped_absent_candidate_frames,
        skipped_invalid_candidate_bbox=stats.skipped_invalid_candidate_bbox,
    )
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "IDENTITY_SOURCE_SCHEMA_VERSION",
    "LABEL_SOURCE",
    "IdentitySampleBuildReport",
    "IdentitySampleConfig",
    "build_identity_samples",
]
