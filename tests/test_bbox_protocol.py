"""锁定 bbox 坐标协议行为。

这里守的是一个已经真实发生过的错误：Qwen2.5-VL 输出的是它自己看到那张图的
绝对像素坐标，而解析端按 norm1000 除以 1000，导致 68/69 个观测帧 IoU 恰好为
0，而模型的文字推理其实是对的。以下测试确保两条协议都不会被静默混用。
"""

import json

import pytest

from cogtrack.prompts import build_mosaic_prompt, build_pair_prompt
from cogtrack.protocol import (
    BBOX_PROTOCOL_NORM1000,
    BBOX_PROTOCOL_QWEN_ABS_PIXEL,
    BoundingBoxError,
    bbox_iou_xywh,
    bbox_protocol_json_key,
    model_pixel_xyxy_to_pixel_xywh,
    pixel_xywh_to_model_pixel_xyxy,
    validate_bbox_protocol,
)
from cogtrack.protocol.exceptions import ModelOutputParseError
from cogtrack.vlm import parse_tracking_output

# 探针序列 videoPlayer_video_09_done 的真实数字：原图 930x510，processor 实测
# 模型空间 644x364（648x355 经 smart_resize 到 28 的整数倍）。
ORIGINAL_SIZE = (930, 510)
MODEL_SIZE = (644, 364)


def _payload(bbox, bbox_key):
    return json.dumps(
        {
            "target_status": "present",
            bbox_key: bbox,
            "memory_update": None,
        }
    )


def test_json_key_differs_per_protocol() -> None:
    # 字段名必须随协议变化，否则模型会被 norm1000 这个名字带向归一化。
    assert bbox_protocol_json_key(BBOX_PROTOCOL_NORM1000) == "bbox_norm1000_xyxy"
    assert bbox_protocol_json_key(BBOX_PROTOCOL_QWEN_ABS_PIXEL) == "bbox_pixel_xyxy"


def test_illegal_protocol_is_rejected() -> None:
    with pytest.raises(BoundingBoxError):
        validate_bbox_protocol("norm1")


def test_prompt_carries_protocol_and_matching_key() -> None:
    for builder in (build_pair_prompt, build_mosaic_prompt):
        kwargs = {"history_count": 3} if builder is build_mosaic_prompt else {}
        prompt = builder(bbox_protocol=BBOX_PROTOCOL_QWEN_ABS_PIXEL, **kwargs)
        assert prompt.bbox_protocol == BBOX_PROTOCOL_QWEN_ABS_PIXEL
        assert "bbox_pixel_xyxy" in prompt.user_prompt
        assert "bbox_norm1000_xyxy" not in prompt.user_prompt
        # 绝对像素协议下绝不能再要求模型归一化。
        assert "[0,1000]" not in prompt.user_prompt
        assert 'Never write a string such as\n  "no change"' in prompt.user_prompt


def test_prompt_default_stays_norm1000_for_training_compatibility() -> None:
    # 通用 builder 默认保持 Qwen3/Qwen2-VL 的 norm1000；Qwen2.5 导出器会显式覆盖。
    assert build_pair_prompt().bbox_protocol == BBOX_PROTOCOL_NORM1000
    assert "bbox_norm1000_xyxy" in build_pair_prompt().user_prompt


def test_presence_only_training_prompt_and_runtime_schema_are_explicitly_versioned() -> None:
    prompt = build_pair_prompt(include_memory_update=False)
    assert prompt.include_memory_update is False
    assert "exactly these two keys" in prompt.user_prompt
    assert "memory_update" not in prompt.user_prompt


def test_full_reference_prompt_uses_coordinates_without_visual_box() -> None:
    prompt = build_pair_prompt(
        reference_has_box=False,
        reference_bbox_norm1000_xyxy=[100, 200, 300, 400],
    )
    assert "unmodified full initialization frame" in prompt.user_prompt
    assert "[100.0, 200.0, 300.0, 400.0]" in prompt.user_prompt
    assert "tight crop" not in prompt.user_prompt


def test_model_pixel_roundtrip_is_exact() -> None:
    original = (698.0, 297.0, 231.0, 155.0)
    model_box = pixel_xywh_to_model_pixel_xyxy(
        original, MODEL_SIZE[0], MODEL_SIZE[1], *ORIGINAL_SIZE, decimals=None
    )
    restored = model_pixel_xyxy_to_pixel_xywh(
        model_box, MODEL_SIZE[0], MODEL_SIZE[1], *ORIGINAL_SIZE
    )
    for got, want in zip(restored, original, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def test_qwen_abs_pixel_recovers_the_box_that_norm1000_destroys() -> None:
    """探针序列 frame 3 的真实数字。

    GT 像素 xywh = (698, 297, 231, 155)，模型给出 [588, 240, 778, 386]。
    按 norm1000 解释 IoU 恰好为 0；按 Qwen 原生绝对像素解释则有实质重叠。
    0.228 不是好定位，那是 7B 模型自身的精度问题；这里守的是“协议正确后不再
    恒为 0”，而不是模型质量。
    """

    gt = (698.0, 297.0, 231.0, 155.0)
    model_numbers = [588, 240, 778, 386]

    norm_parsed = parse_tracking_output(
        _payload(model_numbers, "bbox_norm1000_xyxy"),
        *ORIGINAL_SIZE,
        bbox_protocol=BBOX_PROTOCOL_NORM1000,
    )
    assert bbox_iou_xywh(norm_parsed.prediction.bbox_xywh, gt) == 0.0

    abs_parsed = parse_tracking_output(
        _payload(model_numbers, "bbox_pixel_xyxy"),
        *ORIGINAL_SIZE,
        bbox_protocol=BBOX_PROTOCOL_QWEN_ABS_PIXEL,
        model_image_size=MODEL_SIZE,
    )
    assert abs_parsed.bbox_protocol == BBOX_PROTOCOL_QWEN_ABS_PIXEL
    assert bbox_iou_xywh(abs_parsed.prediction.bbox_xywh, gt) == pytest.approx(0.228, abs=0.01)


def test_missing_model_image_size_is_a_parse_error_not_a_silent_fallback() -> None:
    # 用原图尺寸兜底会引入与目标位置相关的系统性偏移，且被记成正常预测。
    with pytest.raises(ModelOutputParseError, match="model_image_size"):
        parse_tracking_output(
            _payload([588, 240, 778, 386], "bbox_pixel_xyxy"),
            *ORIGINAL_SIZE,
            bbox_protocol=BBOX_PROTOCOL_QWEN_ABS_PIXEL,
        )


def test_wrong_key_for_protocol_is_rejected() -> None:
    with pytest.raises(ModelOutputParseError):
        parse_tracking_output(
            _payload([588, 240, 778, 386], "bbox_norm1000_xyxy"),
            *ORIGINAL_SIZE,
            bbox_protocol=BBOX_PROTOCOL_QWEN_ABS_PIXEL,
            model_image_size=MODEL_SIZE,
        )


def test_mild_overshoot_is_clipped_like_the_norm1000_path() -> None:
    # 模型常给略微越界的坐标；norm1000 靠值域隐式裁剪，这里显式裁剪。
    parsed = parse_tracking_output(
        _payload([600, 300, MODEL_SIZE[0] + 40, MODEL_SIZE[1] + 30], "bbox_pixel_xyxy"),
        *ORIGINAL_SIZE,
        bbox_protocol=BBOX_PROTOCOL_QWEN_ABS_PIXEL,
        model_image_size=MODEL_SIZE,
    )
    x, y, width, height = parsed.prediction.bbox_xywh
    assert x + width == pytest.approx(ORIGINAL_SIZE[0])
    assert y + height == pytest.approx(ORIGINAL_SIZE[1])


def test_fully_out_of_frame_box_still_errors() -> None:
    with pytest.raises(ModelOutputParseError):
        parse_tracking_output(
            _payload([700, 400, 900, 500], "bbox_pixel_xyxy"),
            *ORIGINAL_SIZE,
            bbox_protocol=BBOX_PROTOCOL_QWEN_ABS_PIXEL,
            model_image_size=(200, 200),
        )
