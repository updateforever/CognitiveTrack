"""认知跟踪帧级 JSONL 的读取与字段归一化。

新框架的协议模块仍可能演进，因此评测代码不直接依赖某个 dataclass 或
Pydantic 模型。本模块只通过 ``dict``/属性访问读取字段，并兼容常见的
扁平格式与嵌套格式。这样协议字段做小幅重排时，评测器仍可读取历史结果。

评测内部统一约定：

* 边界框使用像素坐标 ``[x, y, width, height]``；
* presence 只保留 ``present / absent / uncertain``；
* 执行错误与目标 absent 完全分离，错误帧不会被推断成 absent；
* 未提供 presence 的传统跟踪结果，可由有效/无效预测框推断状态。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

PRESENCE_VALUES = {"present", "absent", "uncertain"}
IDENTITY_VALUES = {"same", "different", "uncertain", "not_applicable"}
ERROR_EXECUTION_STATES = {
    "api_error",
    "image_error",
    "model_error",
    "parse_error",
    "runtime_error",
    "internal_error",
    "exception",
    "error",
    "failed",
}


def is_execution_error(status: str) -> bool:
    """识别协议内置及未来扩展的执行错误状态。"""

    normalized = str(status).strip().lower()
    return normalized in ERROR_EXECUTION_STATES or normalized.endswith("_error")


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
    """评测器使用的最小帧记录。

    使用 ``slots`` 且不保留原始 JSON 对象，避免完整 benchmark 的百万级帧
    评测因重复保存 reasoning/raw response 占用数 GB 内存。
    """

    sequence: str
    # pytracking 的 ``seq.dataset``。不是元数据：原版 calc_seq_err_robust 用它
    # 选分支（lasot 把不可见帧的 err_center_norm 置 Inf，uav 允许 GT 里有 NaN），
    # 所以这个字段直接影响 Pnorm 数值，必须从 JSONL 原样带过来而不是从序列名猜。
    dataset: str
    frame_id: int
    gt_bbox: Optional[tuple[float, float, float, float]]
    pred_bbox: Optional[tuple[float, float, float, float]]
    gt_presence: Optional[str]
    pred_presence: Optional[str]
    gt_identity: Optional[str]
    pred_identity: Optional[str]
    execution_status: str
    is_observation_frame: Optional[bool]


def _read_member(value: Any, key: str) -> Any:
    """同时支持字典和普通对象的安全字段读取。"""

    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def get_path(value: Any, dotted_path: str) -> Any:
    """读取 ``prediction.target_presence`` 一类点分隔路径。"""

    current = value
    for key in dotted_path.split("."):
        if current is None:
            return None
        current = _read_member(current, key)
    return current


def first_value(value: Any, paths: Sequence[str]) -> Any:
    """返回第一个非 ``None`` 的候选字段，保留 ``False`` 和数值零。"""

    for path in paths:
        candidate = get_path(value, path)
        if candidate is not None:
            return candidate
    return None


def has_path(value: Any, dotted_path: str) -> bool:
    """判断字段路径是否真实存在；与值为 ``None`` 区分。"""

    current = value
    for key in dotted_path.split("."):
        if isinstance(current, Mapping):
            if key not in current:
                return False
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return False
    return True


def normalize_presence(value: Any) -> Optional[str]:
    """将常见布尔、数值和字符串状态归一化为 presence 标签。"""

    if value is None:
        return None
    if isinstance(value, bool):
        return "present" if value else "absent"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) == 1.0:
            return "present"
        if float(value) == 0.0:
            return "absent"

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "visible": "present",
        "found": "present",
        "exist": "present",
        "exists": "present",
        "not_present": "absent",
        "invisible": "absent",
        "not_found": "absent",
        "unknown": "uncertain",
        "unsure": "uncertain",
    }
    text = aliases.get(text, text)
    return text if text in PRESENCE_VALUES else None


def normalize_identity(value: Any) -> Optional[str]:
    """归一化实例身份标签；无身份标注时返回 ``None``。"""

    if value is None:
        return None
    if isinstance(value, bool):
        return "same" if value else "different"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "match": "same",
        "matched": "same",
        "positive": "same",
        "not_same": "different",
        "mismatch": "different",
        "negative": "different",
        "unknown": "uncertain",
        "n/a": "not_applicable",
        "na": "not_applicable",
        "none": "not_applicable",
    }
    text = aliases.get(text, text)
    return text if text in IDENTITY_VALUES else None


def normalize_execution_status(record: Mapping[str, Any]) -> str:
    """读取执行状态；不把目标状态误当成执行状态。"""

    status = first_value(
        record,
        (
            "execution.status",
            "execution_status",
            "vlm_output.execution.status",
            "vlm_output.execution_status",
        ),
    )
    if status is None and bool(first_value(record, ("skipped", "vlm_output.skipped"))):
        return "skipped"
    if status is None:
        error_type = first_value(
            record,
            ("execution.error_type", "error_type", "vlm_output.error_type"),
        )
        if error_type:
            return "internal_error"
        return "ok"
    text = str(status).strip().lower().replace("-", "_").replace(" ", "_")
    return text or "ok"


def _float4(value: Any) -> Optional[tuple[float, float, float, float]]:
    """把列表或具名坐标字典转换为四个有限浮点数。"""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if all(k in value for k in ("x", "y", "w", "h")):
            value = [value["x"], value["y"], value["w"], value["h"]]
        elif all(k in value for k in ("x", "y", "width", "height")):
            value = [value["x"], value["y"], value["width"], value["height"]]
        elif all(k in value for k in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        else:
            return None
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) < 4:  # type: ignore[arg-type]
            return None
        coords = tuple(float(value[i]) for i in range(4))  # type: ignore[index]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in coords):
        return None
    return coords


def _image_size(record: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    size = first_value(
        record,
        (
            "image_size",
            "current_image_size",
            "context.image_size",
            "prediction.image_size",
        ),
    )
    try:
        width, height = float(size[0]), float(size[1])
    except (TypeError, ValueError, IndexError):
        width = first_value(record, ("image_width", "context.image_width"))
        height = first_value(record, ("image_height", "context.image_height"))
        try:
            width, height = float(width), float(height)
        except (TypeError, ValueError):
            return None
    if width <= 0 or height <= 0 or not (math.isfinite(width) and math.isfinite(height)):
        return None
    return width, height


def _bbox_to_xywh(
    value: Any,
    bbox_format: Optional[str],
    record: Mapping[str, Any],
) -> Optional[tuple[float, float, float, float]]:
    """将明确标注的 xyxy/norm1000 框转换为像素 xywh。

    未标注格式时按 pytracking 的标准 ``xywh`` 解释，绝不根据坐标大小猜测
    格式。归一化框若缺少图像尺寸会返回 ``None``，以免产生看似有效但尺度
    错误的指标。
    """

    coords = _float4(value)
    if coords is None:
        return None
    fmt = (bbox_format or "xywh").strip().lower().replace("-", "_")
    x1, y1, third, fourth = coords

    normalized = "norm" in fmt or "1000" in fmt
    is_xyxy = "xyxy" in fmt
    if normalized:
        size = _image_size(record)
        if size is None:
            return None
        width, height = size
        x1 = x1 / 1000.0 * width
        third = third / 1000.0 * width
        y1 = y1 / 1000.0 * height
        fourth = fourth / 1000.0 * height

    if is_xyxy:
        bbox = (x1, y1, third - x1, fourth - y1)
    else:
        bbox = (x1, y1, third, fourth)
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return None
    return bbox


def _extract_bbox(record: Mapping[str, Any], prediction: bool) -> Optional[tuple[float, float, float, float]]:
    if prediction:
        # 新 runner 的顶层 target_bbox 是经过身份/置信门控后的唯一 benchmark
        # 框。字段即使显式为 null 也必须终止查找，不能回退读取 prediction
        # 中仅供诊断的 candidate bbox。
        for committed_path in ("target_bbox", "tracker_output.target_bbox"):
            if has_path(record, committed_path):
                return _bbox_to_xywh(get_path(record, committed_path), None, record)
        candidates = (
            ("prediction.bbox_xywh", "xywh"),
            ("tracker_output.prediction.bbox_xywh", "xywh"),
            ("tracker_output.cognitive_output.bbox_xywh", "xywh"),
            ("tracker_output.frame_result.prediction.bbox_xywh", "xywh"),
            ("pred_bbox_xywh", "xywh"),
            ("prediction.bbox_xyxy", "xyxy"),
            ("tracker_output.prediction.bbox_xyxy", "xyxy"),
            ("pred_bbox_xyxy", "xyxy"),
            ("prediction.bbox", None),
            ("tracker_output.prediction.bbox", None),
            ("tracker_output.cognitive_output.bbox", None),
            ("pred_bbox", None),
            ("vlm_output.bbox_xywh", "xywh"),
            ("vlm_output.bbox", None),
        )
        format_paths = (
            "prediction.bbox_format",
            "tracker_output.prediction.bbox_format",
            "tracker_output.cognitive_output.bbox_format",
            "pred_bbox_format",
            "bbox_format",
            "vlm_output.bbox_format",
        )
    else:
        candidates = (
            ("ground_truth.bbox_xywh", "xywh"),
            ("gt_bbox_xywh", "xywh"),
            ("ground_truth.bbox_xyxy", "xyxy"),
            ("gt_bbox_xyxy", "xyxy"),
            ("ground_truth.bbox", None),
            ("gt_bbox", None),
            ("ground_truth_rect", None),
        )
        format_paths = ("ground_truth.bbox_format", "gt_bbox_format")

    explicit_format = first_value(record, format_paths)
    for path, field_format in candidates:
        value = get_path(record, path)
        if value is not None:
            return _bbox_to_xywh(value, field_format or explicit_format, record)
    return None


def canonicalize_record(
    record: Mapping[str, Any],
    *,
    source_line: int,
    default_sequence: str,
) -> CanonicalFrame:
    """将任意兼容帧记录转换为 :class:`CanonicalFrame`。"""

    sequence = first_value(record, ("sequence", "sequence_name", "video", "video_name"))
    sequence = sys.intern(str(sequence) if sequence is not None else default_sequence)
    dataset = first_value(record, ("dataset", "dataset_name"))
    # 缺失时退回 "default"，等价于原版对非 lasot/uav 数据集走的通用分支。
    dataset = sys.intern(str(dataset) if dataset is not None else "default")
    frame_id_value = first_value(record, ("frame_id", "frame_index", "frame_num"))
    try:
        frame_id = int(frame_id_value) if frame_id_value is not None else source_line - 1
    except (TypeError, ValueError):
        frame_id = source_line - 1

    execution_status = sys.intern(normalize_execution_status(record))
    gt_bbox = _extract_bbox(record, prediction=False)
    pred_bbox = _extract_bbox(record, prediction=True)

    gt_presence = normalize_presence(
        first_value(
            record,
            (
                "ground_truth.target_presence",
                "ground_truth.presence",
                "gt_target_presence",
                "gt_target_status",
                "gt_presence",
                "target_visible",
            ),
        )
    )
    pred_presence = normalize_presence(
        first_value(
            record,
            (
                "committed_target_presence",
                "tracker_output.committed_target_presence",
                "prediction.target_presence",
                "tracker_output.prediction.target_presence",
                "tracker_output.cognitive_output.target_presence",
                "tracker_output.frame_result.prediction.target_presence",
                "prediction.presence",
                "pred_target_presence",
                "pred_target_status",
                "pred_presence",
                "vlm_output.target_presence",
                "vlm_output.target_status",
            ),
        )
    )

    # 仅在没有显式标签时才从框推断。执行错误帧保持“未决策”，绝不把
    # parser/API/model 错误等价成 absent。
    gt_bbox_paths = (
        "ground_truth.bbox_xywh",
        "gt_bbox_xywh",
        "ground_truth.bbox_xyxy",
        "gt_bbox_xyxy",
        "ground_truth.bbox",
        "gt_bbox",
        "ground_truth_rect",
    )
    if gt_presence is None and any(has_path(record, path) for path in gt_bbox_paths):
        gt_presence = "present" if gt_bbox is not None else "absent"
    if pred_presence is None:
        if pred_bbox is not None and not is_execution_error(execution_status):
            pred_presence = "present"
        elif execution_status in {"ok", "initialized"}:
            # 成功执行但返回无效框时，兼容传统 long-term tracker 的 absent 表达。
            pred_presence = "absent"
    if gt_presence is not None:
        gt_presence = sys.intern(gt_presence)
    if pred_presence is not None:
        pred_presence = sys.intern(pred_presence)

    gt_identity = normalize_identity(
        first_value(
            record,
            (
                "ground_truth.identity_match",
                "gt_identity_match",
                "gt_identity",
            ),
        )
    )
    pred_identity = normalize_identity(
        first_value(
            record,
            (
                "prediction.identity_match",
                "tracker_output.prediction.identity_match",
                "tracker_output.cognitive_output.identity_match",
                "tracker_output.frame_result.prediction.identity_match",
                "pred_identity_match",
                "pred_identity",
                "vlm_output.identity_match",
            ),
        )
    )
    if gt_identity is not None:
        gt_identity = sys.intern(gt_identity)
    if pred_identity is not None:
        pred_identity = sys.intern(pred_identity)
    observation = first_value(
        record,
        ("is_observation_frame", "context.is_observation_frame", "is_keyframe"),
    )
    is_observation_frame = None if observation is None else bool(observation)

    return CanonicalFrame(
        sequence=sequence,
        dataset=dataset,
        frame_id=frame_id,
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        gt_presence=gt_presence,
        pred_presence=pred_presence,
        gt_identity=gt_identity,
        pred_identity=pred_identity,
        execution_status=execution_status,
        is_observation_frame=is_observation_frame,
    )


def read_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """逐行读取 JSONL，并在异常中保留文件与行号。"""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败：{file_path}:{line_no}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL 每行必须是对象：{file_path}:{line_no}")
            yield value


def load_frame_records(paths: Iterable[str | Path]) -> list[CanonicalFrame]:
    """读取多个结果文件，返回按序列和帧号排序的帧记录。"""

    frames: list[CanonicalFrame] = []
    for input_path in paths:
        path = Path(input_path)
        default_sequence = path.stem
        for suffix in ("_frames", ".frames"):
            if default_sequence.endswith(suffix):
                default_sequence = default_sequence[: -len(suffix)]
        for line_no, record in enumerate(read_jsonl(path), start=1):
            frames.append(
                canonicalize_record(
                    record,
                    source_line=line_no,
                    default_sequence=default_sequence,
                )
            )
    return frames


def discover_jsonl_files(input_path: str | Path, pattern: str = "*_frames.jsonl") -> list[Path]:
    """发现待评测文件。

    目录下优先匹配标准 ``*_frames.jsonl``；若一个都没有，则回退到所有
    ``*.jsonl``，便于评测单独导出的实验结果。
    """

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"评测输入必须是 JSONL 文件：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"评测输入不存在：{path}")
    files = sorted(p for p in path.rglob(pattern) if p.is_file())
    if not files and pattern == "*_frames.jsonl":
        files = sorted(p for p in path.rglob("*.jsonl") if p.is_file())
    if not files:
        raise FileNotFoundError(f"目录中没有找到 JSONL 结果：{path}（pattern={pattern}）")
    return files


def is_debug_limited(frames_file: str | Path) -> bool:
    """判断结果文件是否来自 ``--debug-frames`` 截断跑。

    截断序列只有前 N 帧，但在按序列宏平均的指标里会和完整序列等权，
    足以显著扭曲 AUC。runner 把标记写在 manifest 的 ``extra.debug_limited``。
    manifest 缺失或不可解析时按“非 debug”处理，避免误删正常结果。
    """

    frames_path = Path(frames_file)
    suffix = "_frames.jsonl"
    name = frames_path.name
    stem = name[: -len(suffix)] if name.endswith(suffix) else frames_path.stem
    manifest_path = frames_path.with_name(f"{stem}_manifest.json")
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    extra = manifest.get("extra")
    if not isinstance(extra, dict):
        return False
    return bool(extra.get("debug_limited"))


def partition_debug_limited(
    files: Sequence[str | Path],
) -> tuple[list[Path], list[Path]]:
    """把结果文件拆成（完整, debug 截断）两组。"""

    full: list[Path] = []
    debug: list[Path] = []
    for item in files:
        path = Path(item)
        (debug if is_debug_limited(path) else full).append(path)
    return full, debug
