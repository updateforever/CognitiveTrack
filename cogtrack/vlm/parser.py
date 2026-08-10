"""CognitiveTrack v4 的严格 VLM JSON 解析器。

解析器只允许完整 JSON 对象（可有一层 ``json`` code fence），不使用正则从
自然语言中“捞”边界框，也不猜测坐标系。任何失败都会抛出
``ModelOutputParseError``，由运行层记录为 ``execution=parse_error``。
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

from ..protocol.bbox import (
    BBOX_PROTOCOL_NORM1000,
    BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    BBoxXYXY,
    bbox_protocol_json_key,
    model_pixel_xyxy_to_pixel_xywh,
    norm1000_xyxy_to_pixel_xywh,
    validate_bbox_protocol,
    validate_norm1000_xyxy,
    validate_xyxy,
)
from ..protocol.enums import IdentityMatch, Localizability, TargetPresence
from ..protocol.exceptions import ModelOutputParseError, ProtocolValidationError
from ..protocol.schema import CognitionInfo, Prediction

_JSON_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(?P<body>.*)\n```\s*\Z", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedTrackingOutput:
    """已转换到内部像素坐标的跟踪输出。"""

    prediction: Prediction
    cognition: CognitionInfo
    #: 模型原样给出的 bbox，坐标系由 ``bbox_protocol`` 决定；字段名沿用
    #: ``bbox_norm1000_xyxy`` 是为了不破坏既有调用方，含义已推广为“模型协议
    #: 坐标系下的框”。
    bbox_norm1000_xyxy: Optional[BBoxXYXY]
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000


@dataclass(frozen=True)
class ParsedIdentityOutput:
    """候选实例身份核验输出。"""

    identity_match: IdentityMatch
    reasoning: str


@dataclass(frozen=True)
class ParsedMemoryOutput:
    """VLM 对语义记忆更新请求的结构化建议。"""

    update_memory: bool
    memory_type: str
    summary: str
    reasoning: str


def _load_json_object(raw_text: str) -> Dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ModelOutputParseError("模型输出为空")
    stripped = raw_text.strip()
    fence = _JSON_FENCE.fullmatch(stripped)
    if fence:
        stripped = fence.group("body").strip()

    def reject_non_standard_constant(value: str) -> None:
        # Python json 默认会接受 NaN/Infinity；标准 JSON 和本协议均不允许。
        raise ModelOutputParseError(f"JSON 中不允许非有限常量 {value}")

    def unique_object(pairs: Any) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ModelOutputParseError(f"JSON 对象包含重复字段 {key!r}")
            output[key] = value
        return output

    try:
        data = json.loads(
            stripped,
            parse_constant=reject_non_standard_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as error:
        raise ModelOutputParseError(f"模型输出不是完整合法 JSON：第 {error.lineno} 行第 {error.colno} 列") from error
    if not isinstance(data, dict):
        raise ModelOutputParseError("模型输出的 JSON 根节点必须是对象")
    return data


def _require_exact_keys(data: Mapping[str, Any], expected: Set[str], task: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"缺少字段 {missing}")
        if unknown:
            details.append(f"未知字段 {unknown}")
        raise ModelOutputParseError(f"{task} 输出字段不符合 v4 协议：{'；'.join(details)}")


def _enum(value: Any, enum_class: Any, field_name: str) -> Any:
    if not isinstance(value, str):
        raise ModelOutputParseError(f"{field_name} 必须是字符串枚举")
    try:
        return enum_class(value)
    except ValueError as error:
        allowed = [item.value for item in enum_class]
        raise ModelOutputParseError(f"{field_name}={value!r} 非法，可选值为 {allowed}") from error


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ModelOutputParseError(f"{field_name} 必须是字符串")
    return value.strip()


def parse_tracking_output(
    raw_text: str,
    image_width: int,
    image_height: int,
    *,
    bbox_protocol: str = BBOX_PROTOCOL_NORM1000,
    model_image_size: Optional[Tuple[int, int]] = None,
    require_memory_update: bool = True,
) -> ParsedTrackingOutput:
    """严格解析 pair/mosaic 跟踪输出，并转换 bbox 坐标。

    ``image_width/image_height`` 始终是原始视频帧尺寸，即返回的像素框所在坐标系。
    ``bbox_protocol`` 必须与生成该输出的 Prompt 一致：

    ``norm1000``
        模型给 ``[0,1000]`` 归一化坐标，直接按原图尺寸反归一化。
    ``qwen_abs_pixel``
        模型给它自己看到的那张图的绝对像素坐标，必须额外提供
        ``model_image_size``（processor 实际使用的 ``(width, height)``）才能线性
        映回原图。缺失时直接报错而不用原图尺寸兜底：两者通常不等，兜底会引入
        一个与目标位置相关的系统性偏移，并被静默记成正常预测。
    """

    validate_bbox_protocol(bbox_protocol)
    bbox_key = bbox_protocol_json_key(bbox_protocol)
    data = _load_json_object(raw_text)
    expected = {
        "target_status",
        bbox_key,
    }
    if require_memory_update:
        expected.add("memory_update")
    _require_exact_keys(data, expected, "tracking")

    presence = _enum(data["target_status"], TargetPresence, "target_status")
    if presence is TargetPresence.UNCERTAIN:
        raise ModelOutputParseError("v4 主跟踪任务的 target_status 只允许 present/absent")
    normalized_bbox: Optional[BBoxXYXY]
    bbox_value = data[bbox_key]
    if bbox_value is None:
        normalized_bbox = None
        pixel_bbox = None
    elif bbox_protocol == BBOX_PROTOCOL_NORM1000:
        try:
            normalized_bbox = validate_norm1000_xyxy(bbox_value)
            pixel_bbox = norm1000_xyxy_to_pixel_xywh(
                normalized_bbox,
                image_width,
                image_height,
            )
        except (TypeError, ValueError, ProtocolValidationError) as error:
            raise ModelOutputParseError(f"{bbox_key} 非法：{error}") from error
    else:
        if model_image_size is None:
            raise ModelOutputParseError(
                f"{BBOX_PROTOCOL_QWEN_ABS_PIXEL} 协议需要 model_image_size，"
                "即 processor 实际喂给模型的图像尺寸；缺失时无法把绝对像素坐标映回原图"
            )
        try:
            normalized_bbox = validate_xyxy(bbox_value)
            pixel_bbox = model_pixel_xyxy_to_pixel_xywh(
                normalized_bbox,
                model_image_size[0],
                model_image_size[1],
                image_width,
                image_height,
            )
        except (TypeError, ValueError, ProtocolValidationError) as error:
            raise ModelOutputParseError(f"{bbox_key} 非法：{error}") from error

    memory_update: Optional[str] = None
    if require_memory_update:
        raw_memory = data["memory_update"]
        if raw_memory is not None:
            if not isinstance(raw_memory, str):
                raise ModelOutputParseError("memory_update 必须是 null 或字符串")
            memory_update = raw_memory.strip()
            if not memory_update:
                raise ModelOutputParseError("memory_update 字符串不能为空；不更新时必须使用 null")
            if len(memory_update) > 300:
                raise ModelOutputParseError("memory_update 不能超过 300 个字符")

    # v4 的身份和可定位性不再由模型生成，而由二分类标签与 bbox 确定性派生。
    if presence is TargetPresence.ABSENT:
        if normalized_bbox is not None:
            raise ModelOutputParseError("absent 时 bbox 必须为 null")
        identity = IdentityMatch.NOT_APPLICABLE
        localizability = Localizability.NOT_APPLICABLE
        if memory_update is not None:
            raise ModelOutputParseError("absent 时 memory_update 必须为 null")
    elif normalized_bbox is None:
        raise ModelOutputParseError(f"present + localizable 时必须输出 {bbox_key}")
    else:
        # ``present`` 在 Prompt 中已经定义为“初始化实例可见且可定位”。这里的
        # same/localizable 只是兼容内部状态机和历史结果结构的派生事实。
        identity = IdentityMatch.SAME
        localizability = Localizability.LOCALIZABLE

    try:
        prediction = Prediction(
            target_presence=presence,
            identity_match=identity,
            localizability=localizability,
            bbox_xywh=pixel_bbox,
        )
    except ProtocolValidationError as error:
        raise ModelOutputParseError(f"tracking 输出字段语义冲突：{error}") from error

    cognition = CognitionInfo(memory_update_proposal=memory_update)
    return ParsedTrackingOutput(prediction, cognition, normalized_bbox, bbox_protocol)


def parse_identity_output(raw_text: str) -> ParsedIdentityOutput:
    """严格解析独立身份核验任务输出。"""

    data = _load_json_object(raw_text)
    _require_exact_keys(data, {"identity_match", "reasoning"}, "identity")
    identity_match = _enum(data["identity_match"], IdentityMatch, "identity_match")
    if identity_match is IdentityMatch.NOT_APPLICABLE:
        raise ModelOutputParseError("独立 identity 任务存在明确候选，不能返回 not_applicable")
    return ParsedIdentityOutput(
        identity_match=identity_match,
        reasoning=_string(data["reasoning"], "reasoning"),
    )


def parse_memory_output(raw_text: str) -> ParsedMemoryOutput:
    """严格解析语义记忆更新建议；真正写入仍需经过门控策略。"""

    data = _load_json_object(raw_text)
    expected = {"update_memory", "memory_type", "summary", "reasoning"}
    _require_exact_keys(data, expected, "memory")
    if not isinstance(data["update_memory"], bool):
        raise ModelOutputParseError("update_memory 必须是 JSON 布尔值")
    memory_type = _string(data["memory_type"], "memory_type")
    allowed_types = {"positive", "negative", "semantic", "none"}
    if memory_type not in allowed_types:
        raise ModelOutputParseError(f"memory_type 必须是 {sorted(allowed_types)} 之一")
    if data["update_memory"] and memory_type == "none":
        raise ModelOutputParseError("update_memory=true 时 memory_type 不能为 none")
    if not data["update_memory"] and memory_type != "none":
        raise ModelOutputParseError("update_memory=false 时 memory_type 必须为 none")
    summary = _string(data["summary"], "summary")
    if data["update_memory"] and not summary:
        raise ModelOutputParseError("需要更新记忆时 summary 不能为空")
    if not data["update_memory"] and summary:
        raise ModelOutputParseError("不更新记忆时 summary 必须为空字符串")
    return ParsedMemoryOutput(
        update_memory=data["update_memory"],
        memory_type=memory_type,
        summary=summary,
        reasoning=_string(data["reasoning"], "reasoning"),
    )
