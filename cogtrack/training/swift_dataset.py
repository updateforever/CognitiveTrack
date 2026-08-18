"""ms-swift 多模态数据的转换、校验与按序列划分。

该模块只处理通用 ``messages + images`` JSONL，不读取具体跟踪数据集。后续
pair、mosaic、身份困难负样本等构造器只需生成规范化样本，即可复用这里的
校验与 split 逻辑。

SFT 样本格式：

.. code-block:: json

    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "<image><image>..."},
        {"role": "assistant", "content": "{...}"}
      ],
      "images": ["images/reference.jpg", "images/current.jpg"],
      "metadata": {"source_sequence": "airplane-1"}
    }

GRPO 样本会移除 assistant 消息，并把参考答案放入 ``solution``，防止答案
泄漏到模型输入；自定义 reward 从 ``solution`` 读取监督信号。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from cogtrack.protocol import (
    BBOX_PROTOCOL_NORM1000,
    BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    MEMORY_UPDATE_JSON_KEY,
    TARGET_STATUS_JSON_KEY,
    ModelOutputParseError,
    bbox_protocol_json_key,
)
from cogtrack.vlm import parse_tracking_output

VALID_ROLES = {"system", "user", "assistant", "tool"}
VALID_PRESENCE = {"present", "absent"}
# 模型可见的输出字段名统一由 cogtrack.protocol 定义；这里不再硬编码，避免协议
# 改名后校验器与导出器不一致。
TRACKING_OUTPUT_KEYS = frozenset(
    {
        TARGET_STATUS_JSON_KEY,
        bbox_protocol_json_key(BBOX_PROTOCOL_NORM1000),
    }
)
TRACKING_OUTPUT_KEYS_WITH_MEMORY = TRACKING_OUTPUT_KEYS | {MEMORY_UPDATE_JSON_KEY}
TRACKING_PIXEL_OUTPUT_KEYS = frozenset(
    {
        TARGET_STATUS_JSON_KEY,
        bbox_protocol_json_key(BBOX_PROTOCOL_QWEN_ABS_PIXEL),
    }
)
TRACKING_PIXEL_OUTPUT_KEYS_WITH_MEMORY = TRACKING_PIXEL_OUTPUT_KEYS | {
    MEMORY_UPDATE_JSON_KEY
}


@dataclass(frozen=True)
class ValidationIssue:
    """单条校验问题。``error`` 会阻止样本进入训练集。"""

    severity: str
    code: str
    message: str
    sample_index: Optional[int] = None
    source: Optional[str] = None


@dataclass
class ValidationReport:
    """一批样本的结构化校验报告。"""

    total_samples: int
    valid_samples: int
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(issue) for issue in self.issues],
        }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """读取普通对象 JSONL，并提供精确的错误位置。"""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败：{file_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 每行必须是对象：{file_path}:{line_no}")
            # 输入来源只用于报告，不会写入最终训练样本。
            row.setdefault("_source", f"{file_path}:{line_no}")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """以 UTF-8 紧凑 JSON 写出数据，并返回样本数。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "_source"}
            handle.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_images(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError("images 必须是路径字符串或路径列表")


def _collect_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    for key in (
        "id",
        "task",
        "dataset",
        "source_dataset",
        "sequence",
        "source_sequence",
        "frame_id",
        "bbox_format",
    ):
        if key in row and key not in metadata:
            metadata[key] = row[key]
    return metadata


def to_ms_swift_record(
    row: Mapping[str, Any],
    *,
    default_system_prompt: str = "You are a cognitive visual tracking assistant.",
) -> dict[str, Any]:
    """把已是 messages 格式或简化 canonical 样本转换为 ms-swift 格式。

    简化样本可使用 ``user_prompt/prompt``、``assistant/answer/target`` 和
    ``images`` 字段。若 prompt 没有 ``<image>`` 标记，本函数会按图片数量
    自动补在最前面。
    """

    images = _normalize_images(row.get("images"))
    if isinstance(row.get("messages"), list):
        messages = [dict(message) for message in row["messages"]]
    else:
        user = row.get("user_prompt", row.get("prompt"))
        assistant = row.get("assistant", row.get("answer", row.get("target")))
        if user is None:
            raise ValueError("样本缺少 messages 或 user_prompt/prompt")
        if assistant is None:
            raise ValueError("样本缺少 assistant/answer/target")
        system = row.get("system_prompt", default_system_prompt)
        messages = [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": str(user)},
            {"role": "assistant", "content": _json_text(assistant)},
        ]

    # ms-swift 的独立 images 列需要在文本中有等量占位符。仅在完全没有
    # 占位符时自动补齐；已有但数量错误时交给校验器报错，避免改变语义顺序。
    image_tokens = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if images and image_tokens == 0:
        for message in messages:
            if message.get("role") == "user":
                message["content"] = "<image>" * len(images) + str(message.get("content", ""))
                break

    result: dict[str, Any] = {
        "messages": messages,
        "images": images,
        "metadata": _collect_metadata(row),
    }
    if isinstance(row.get("objects"), Mapping):
        # ``<bbox>`` 由 ms-swift 根据模型模板替换；objects 必须原样保留。
        result["objects"] = dict(row["objects"])
    # 保留 GRPO reward 可能需要的额外监督字段。
    for key in (
        "solution",
        "target_status",
        "target_presence",
        "memory_update",
        "bbox",
        "bbox_norm1000_xyxy",
        "bbox_pixel_xyxy",
        "bbox_format",
    ):
        if key in row:
            result[key] = row[key]
    if "_source" in row:
        result["_source"] = row["_source"]
    return result


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else value


def parse_answer_json(value: Any) -> Optional[dict[str, Any]]:
    """解析训练参考答案；只接受完整 JSON 对象。"""

    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"重复 JSON 字段：{key}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"非标准 JSON 常量：{constant}")

    try:
        parsed = json.loads(
            _strip_code_fence(value),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_training_tracking_answer(value: Any) -> dict[str, Any]:
    """按推理侧 ``parse_tracking_output`` 的同一协议校验训练答案。

    训练标签和线上推理共享一套状态组合约束，避免 SFT/GRPO 教会模型输出
    推理解析器随后会拒绝的答案。这里用 1000×1000 的虚拟图像做转换，只为
    复用解析器的严格二/三字段、二分类状态及 norm1000 bbox 校验；不会改变
    或量化原始训练框。

    Returns:
        解析后的原始二字段或三字段字典。

    Raises:
        ValueError: 答案不是完整 JSON 对象，或不符合跟踪输出协议。
    """

    # 官方 ms-swift Grounding 数据在序列化阶段保留 ``<bbox>``；真正 encode 时
    # 才按模型族替换。这里用一个合法框做结构校验，不冒充最终训练坐标。
    materialized = value
    if isinstance(value, str) and "<bbox>" in value:
        materialized = value.replace("<bbox>", "[100,100,200,200]")
    answer = parse_answer_json(materialized)
    if answer is None:
        raise ValueError("assistant/solution 必须是完整 JSON 对象")
    answer_keys = set(answer)
    valid_key_sets = (
        set(TRACKING_OUTPUT_KEYS),
        set(TRACKING_OUTPUT_KEYS_WITH_MEMORY),
        set(TRACKING_PIXEL_OUTPUT_KEYS),
        set(TRACKING_PIXEL_OUTPUT_KEYS_WITH_MEMORY),
    )
    if answer_keys not in valid_key_sets:
        pixel = bbox_protocol_json_key(BBOX_PROTOCOL_QWEN_ABS_PIXEL) in answer
        has_memory = MEMORY_UPDATE_JSON_KEY in answer
        expected = (
            TRACKING_PIXEL_OUTPUT_KEYS_WITH_MEMORY
            if pixel and has_memory
            else TRACKING_PIXEL_OUTPUT_KEYS
            if pixel
            else TRACKING_OUTPUT_KEYS_WITH_MEMORY
            if has_memory
            else TRACKING_OUTPUT_KEYS
        )
        missing = sorted(expected - answer_keys)
        unknown = sorted((repr(key) for key in answer_keys - expected))
        details = []
        if missing:
            details.append(f"缺少字段 {missing}")
        if unknown:
            details.append(f"未知字段 {unknown}")
        raise ValueError(
            "答案必须严格包含 CognitiveTrack presence/bbox 二字段，或增加 memory_update 的三字段："
            f"{'；'.join(details)}"
        )

    try:
        # allow_nan=False 保证 Mapping 输入也遵循标准 JSON；字符串输入已经由
        # parse_answer_json 拒绝了 NaN/Infinity 和重复字段。
        canonical = (
            materialized
            if isinstance(materialized, str)
            else json.dumps(
                answer,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        bbox_protocol = (
            BBOX_PROTOCOL_QWEN_ABS_PIXEL
            if bbox_protocol_json_key(BBOX_PROTOCOL_QWEN_ABS_PIXEL) in answer
            else BBOX_PROTOCOL_NORM1000
        )
        parse_tracking_output(
            canonical,
            image_width=1000,
            image_height=1000,
            bbox_protocol=bbox_protocol,
            model_image_size=(
                (1000, 1000) if bbox_protocol == BBOX_PROTOCOL_QWEN_ABS_PIXEL else None
            ),
            require_memory_update=MEMORY_UPDATE_JSON_KEY in answer,
            strict_memory_update=True,
        )
    except (ModelOutputParseError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"答案不符合推理协议：{error}") from error
    return answer


def _assistant_contents(record: Mapping[str, Any]) -> list[Any]:
    """提取所有 SFT assistant；仅在无 assistant 时回退到 GRPO solution。"""

    messages = record.get("messages")
    if isinstance(messages, list):
        answers = [
            message.get("content")
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ]
        if answers:
            return answers
    return [record["solution"]] if "solution" in record else []


def validate_ms_swift_record(
    record: Mapping[str, Any],
    *,
    image_root: str | Path = ".",
    check_images: bool = True,
    allow_absolute_images: bool = False,
    check_answer_json: bool = True,
    sample_index: Optional[int] = None,
) -> list[ValidationIssue]:
    """校验一条多模态 SFT/GRPO 样本。"""

    source = str(record.get("_source")) if record.get("_source") else None
    issues: list[ValidationIssue] = []

    def add(severity: str, code: str, message: str) -> None:
        issues.append(ValidationIssue(severity, code, message, sample_index, source))

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        add("error", "messages.invalid", "messages 必须是非空列表")
        messages = []
    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            add("error", "message.invalid", f"messages[{index}] 必须是对象")
            continue
        role = str(message.get("role", ""))
        roles.append(role)
        if role not in VALID_ROLES:
            add("error", "message.role", f"messages[{index}] 的 role 非法：{role!r}")
        if not isinstance(message.get("content"), str):
            add("error", "message.content", f"messages[{index}].content 必须是字符串")
    if "user" not in roles:
        add("error", "messages.user_missing", "messages 缺少 user 消息")
    if "assistant" not in roles and "solution" not in record:
        add("error", "answer.missing", "SFT 需要 assistant 消息，GRPO 需要 solution 字段")

    images = record.get("images")
    if not isinstance(images, list) or not images:
        add("error", "images.invalid", "跟踪样本的 images 必须是非空列表")
        images = []
    image_tokens = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if image_tokens != len(images):
        add(
            "error",
            "images.token_mismatch",
            f"<image> 数量 {image_tokens} 与 images 数量 {len(images)} 不一致",
        )

    bbox_tokens = sum(
        str(message.get("content", "")).count("<bbox>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if "solution" in record:
        bbox_tokens += str(record.get("solution", "")).count("<bbox>")
    objects = record.get("objects")
    if bbox_tokens:
        if not isinstance(objects, Mapping):
            add("error", "objects.missing", "包含 <bbox> 时必须提供 ms-swift objects")
        else:
            boxes = objects.get("bbox")
            image_ids = objects.get("image_id")
            bbox_type = objects.get("bbox_type", "real")
            if not isinstance(boxes, list) or len(boxes) != bbox_tokens:
                actual = len(boxes) if isinstance(boxes, list) else None
                add(
                    "error",
                    "objects.bbox_count",
                    f"<bbox> 数量 {bbox_tokens} 与 objects.bbox 数量 {actual} 不一致",
                )
            if bbox_type not in {"real", "norm1"}:
                add("error", "objects.bbox_type", f"bbox_type 非法：{bbox_type!r}")
            if not isinstance(image_ids, list) or len(image_ids) != bbox_tokens:
                actual = len(image_ids) if isinstance(image_ids, list) else None
                add(
                    "error",
                    "objects.image_id_count",
                    f"<bbox> 数量 {bbox_tokens} 与 objects.image_id 数量 {actual} 不一致",
                )
            elif any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= len(images)
                for value in image_ids
            ):
                add("error", "objects.image_id", "objects.image_id 包含越界图片索引")

    root = Path(image_root).resolve()
    for index, image in enumerate(images):
        if not isinstance(image, str) or not image.strip():
            add("error", "image.path", f"images[{index}] 必须是非空路径字符串")
            continue
        image_path = Path(image)
        if image_path.is_absolute() and not allow_absolute_images:
            add("error", "image.absolute", f"禁止绝对图片路径：{image_path}")
            continue
        resolved = image_path if image_path.is_absolute() else root / image_path
        if not image_path.is_absolute():
            try:
                resolved.resolve().relative_to(root)
            except ValueError:
                add("error", "image.escape", f"图片路径逃逸 image_root：{image_path}")
                continue
        if check_images and not resolved.is_file():
            add("error", "image.missing", f"图片不存在：{resolved}")

    if check_answer_json:
        answers = _assistant_contents(record)
        if not answers:
            add("error", "answer.missing", "缺少 assistant/solution 跟踪答案")
        for answer_index, value in enumerate(answers):
            try:
                parse_training_tracking_answer(value)
            except ValueError as error:
                location = f"assistant[{answer_index}]" if len(answers) > 1 else "assistant/solution"
                add("error", "answer.protocol", f"{location}：{error}")

    return issues


def validate_records(
    records: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> ValidationReport:
    """批量校验并汇总样本级错误。"""

    issues: list[ValidationIssue] = []
    valid = 0
    for index, record in enumerate(records):
        current = validate_ms_swift_record(record, sample_index=index, **kwargs)
        issues.extend(current)
        if not any(issue.severity == "error" for issue in current):
            valid += 1
    return ValidationReport(len(records), valid, issues)


def sequence_key(record: Mapping[str, Any]) -> str:
    """生成不会跨数据集碰撞的序列键。"""

    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    sequence = (
        record.get("source_sequence")
        or record.get("sequence")
        or metadata.get("source_sequence")
        or metadata.get("sequence")
    )
    dataset = (
        record.get("source_dataset")
        or record.get("dataset")
        or metadata.get("source_dataset")
        or metadata.get("dataset")
        or "unknown"
    )
    if sequence is None or not str(sequence).strip():
        raise ValueError("按序列划分需要 sequence/source_sequence 字段")
    return f"{dataset}::{sequence}"


def split_records_by_sequence(
    records: Sequence[dict[str, Any]],
    *,
    val_ratio: float = 0.05,
    test_ratio: float = 0.0,
    seed: int = 20260805,
) -> dict[str, list[dict[str, Any]]]:
    """按完整序列划分 train/val/test，杜绝相邻帧数据泄漏。"""

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio/test_ratio 必须非负，且两者之和小于 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(sequence_key(record), []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n_groups = len(keys)
    n_test = int(round(n_groups * test_ratio))
    n_val = int(round(n_groups * val_ratio))
    if test_ratio > 0 and n_groups >= 2:
        n_test = max(1, n_test)
    if val_ratio > 0 and n_groups - n_test >= 2:
        n_val = max(1, n_val)
    # 始终至少保留一个训练序列；小数据 dry-run 不应被 ratio 全部分走。
    overflow = max(0, n_test + n_val - max(0, n_groups - 1))
    reduce_val = min(overflow, n_val)
    n_val -= reduce_val
    n_test -= overflow - reduce_val

    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test : n_test + n_val])
    splits = {"train": [], "val": [], "test": []}
    for key, values in groups.items():
        split = "test" if key in test_keys else "val" if key in val_keys else "train"
        splits[split].extend(values)
    return splits


def to_grpo_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """把 SFT record 转成不会泄漏答案的 GRPO record。"""

    result = dict(record)
    messages = [dict(message) for message in result.get("messages", [])]
    solution = result.get("solution")
    prompt_messages = []
    for message in messages:
        if message.get("role") == "assistant":
            if solution is None:
                solution = message.get("content")
            continue
        prompt_messages.append(message)
    if solution is None:
        raise ValueError("GRPO 样本缺少 assistant 参考答案或 solution")
    result["messages"] = prompt_messages
    result["solution"] = _json_text(solution)

    # 把监督字段同时保留为独立列，兼容 ms-swift reward kwargs 的列传递。
    answer = parse_answer_json(solution)
    if answer:
        for key in (
            TARGET_STATUS_JSON_KEY,
            "target_status",
            "target_presence",
            MEMORY_UPDATE_JSON_KEY,
            "bbox",
            bbox_protocol_json_key(BBOX_PROTOCOL_NORM1000),
            bbox_protocol_json_key(BBOX_PROTOCOL_QWEN_ABS_PIXEL),
            "bbox_norm1000_xyxy",
            "bbox_format",
        ):
            if key in answer and key not in result:
                result[key] = answer[key]
        # reward 侧统一读 ``target_presence``；模型输出字段名为 ``status``。
        if TARGET_STATUS_JSON_KEY in answer and "target_presence" not in result:
            result["target_presence"] = answer[TARGET_STATUS_JSON_KEY]
        elif "target_status" in answer and "target_presence" not in result:
            result["target_presence"] = answer["target_status"]
    return result
