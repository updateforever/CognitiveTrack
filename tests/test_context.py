import numpy as np
import pytest

from cogtrack.context import (
    HISTORY_LAYOUT_COMPACT_GRID_V1,
    HISTORY_LAYOUT_RECENT_STRIP_3_V1,
    HISTORY_LAYOUT_RECENT_STRIP_3_V2,
    HISTORY_STRIP_SEPARATOR_COLOR_RGB,
    PROMPT_PROFILE_VLT_V6,
    REFERENCE_MODE_VISUAL_BOX,
    SAFE_INIT_LANGUAGE_SCOPES,
    VISUAL_MARKER_VERSION,
    TrackingContextBuilder,
    arrange_history_items,
    build_history_mosaic,
    draw_reference_box,
    is_unsafe_init_language_scope,
)
from cogtrack.memory import IdentityAnchor, MemoryKind, MemoryRecord, MemorySource


def test_mosaic_without_trusted_history_falls_back_to_pair():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    builder = TrackingContextBuilder(IdentityAnchor(0, (10, 10, 20, 30), image=image))
    result = builder.build_mosaic(image, (), "target")
    assert result.effective_mode == "pair"
    assert len(result.images) == 2
    assert np.array_equal(result.images[0], image)
    assert "[62.5, 100.0, 187.5, 400.0]" in result.prompt.user_prompt


def test_mosaic_uses_only_explicit_records():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    builder = TrackingContextBuilder(IdentityAnchor(0, (10, 10, 20, 30), image=image))
    record = MemoryRecord(
        record_id="positive-1",
        kind=MemoryKind.POSITIVE,
        frame_id=4,
        source=MemorySource.VLM_PREDICTION,
        bbox_xywh=(20, 20, 30, 30),
        image=image,
    )
    result = builder.build_mosaic(image, (record,), "target")
    assert result.effective_mode == "mosaic"
    assert len(result.images) == 3
    assert result.reference_frames == (0, 4)
    assert result.images[1].shape[:2] == (240, 384)


def test_mosaic_uses_compact_grid_without_frame_number_header():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    builder = TrackingContextBuilder(IdentityAnchor(0, (10, 10, 20, 30), image=image))
    records = tuple(
        MemoryRecord(
            record_id=f"positive-{frame_id}",
            kind=MemoryKind.POSITIVE,
            frame_id=frame_id,
            source=MemorySource.VLM_PREDICTION,
            bbox_xywh=(20, 20, 30, 30),
            image=image,
        )
        for frame_id in (16, 4, 12, 8)
    )
    result = builder.build_mosaic(image, records, "target")

    # 四帧按 2x2 排列；没有旧实现额外添加的 30px 帧号标题行。
    assert result.images[1].shape[:2] == (480, 768)
    assert result.reference_frames == (0, 4, 8, 12, 16)
    assert result.history_layout_version == HISTORY_LAYOUT_COMPACT_GRID_V1


def test_recent_history_strip_keeps_latest_three_and_pads_on_the_right():
    assert arrange_history_items(
        (4,), layout=HISTORY_LAYOUT_RECENT_STRIP_3_V2
    ) == (4, 4, 4)
    assert arrange_history_items(
        (4, 8), layout=HISTORY_LAYOUT_RECENT_STRIP_3_V2
    ) == (4, 8, 8)
    assert arrange_history_items(
        (4, 8, 12, 16), layout=HISTORY_LAYOUT_RECENT_STRIP_3_V2
    ) == (8, 12, 16)

    panels = tuple(
        (np.full((100, 160, 3), value, dtype=np.uint8), (10, 10, 20, 20))
        for value in (40, 80)
    )
    legacy_strip = build_history_mosaic(
        panels,
        panel_height=240,
        layout=HISTORY_LAYOUT_RECENT_STRIP_3_V1,
    )
    assert legacy_strip.shape[:2] == (240, 1152)

    strip = build_history_mosaic(
        panels,
        panel_height=240,
        layout=HISTORY_LAYOUT_RECENT_STRIP_3_V2,
    )
    assert strip.shape[:2] == (240, 1166)
    assert np.all(strip[180, 192] == 40)
    assert np.all(strip[180, 583] == 80)
    assert np.all(strip[180, 974] == 80)
    assert np.all(strip[:, 384:391] == HISTORY_STRIP_SEPARATOR_COLOR_RGB)
    assert np.all(strip[:, 775:782] == HISTORY_STRIP_SEPARATOR_COLOR_RGB)


def test_vlt_history_strip_drops_old_records_and_preserves_order():
    anchor = np.zeros((100, 160, 3), dtype=np.uint8)
    builder = TrackingContextBuilder(
        IdentityAnchor(0, (10, 10, 20, 30), image=anchor),
        reference_mode=REFERENCE_MODE_VISUAL_BOX,
        prompt_profile=PROMPT_PROFILE_VLT_V6,
        force_history_image=True,
    )
    records = tuple(
        MemoryRecord(
            record_id=f"positive-{frame_id}",
            kind=MemoryKind.POSITIVE,
            frame_id=frame_id,
            source=MemorySource.VLM_PREDICTION,
            bbox_xywh=(20, 20, 30, 30),
            image=np.full_like(anchor, frame_id),
        )
        for frame_id in (12, 4, 16, 8)
    )

    result = builder.build_mosaic(anchor, records)

    assert result.reference_frames == (0, 8, 12, 16)
    assert result.images[1].shape[:2] == (240, 1166)
    assert result.history_layout_version == HISTORY_LAYOUT_RECENT_STRIP_3_V2


def test_visual_box_pair_marks_only_past_anchor_and_has_no_coordinate_text():
    anchor_image = np.zeros((100, 160, 3), dtype=np.uint8)
    current_image = np.full((100, 160, 3), 17, dtype=np.uint8)
    builder = TrackingContextBuilder(
        IdentityAnchor(0, (10, 10, 20, 30), image=anchor_image),
        reference_mode=REFERENCE_MODE_VISUAL_BOX,
    )

    result = builder.build_pair(current_image)

    assert result.reference_mode == REFERENCE_MODE_VISUAL_BOX
    assert result.visual_marker_version == VISUAL_MARKER_VERSION
    assert result.prompt.name == "cognitive_visual_pair"
    assert result.prompt.version == "5.0.0"
    assert "red box" in result.prompt.user_prompt
    assert "normalized 0-to-1000 xyxy coordinates" not in result.prompt.user_prompt
    assert "[62.5, 100.0, 187.5, 400.0]" not in result.prompt.user_prompt
    assert np.array_equal(result.images[0], draw_reference_box(anchor_image, (10, 10, 20, 30)))
    assert np.array_equal(result.images[-1], current_image)
    assert np.array_equal(anchor_image, np.zeros_like(anchor_image))


def test_visual_box_online_mosaic_uses_shared_renderer():
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    builder = TrackingContextBuilder(
        IdentityAnchor(0, (10, 10, 20, 30), image=image),
        reference_mode=REFERENCE_MODE_VISUAL_BOX,
    )
    record = MemoryRecord(
        record_id="positive-shared-renderer",
        kind=MemoryKind.POSITIVE,
        frame_id=4,
        source=MemorySource.VLM_PREDICTION,
        bbox_xywh=(20, 20, 30, 30),
        image=image,
    )

    result = builder.build_mosaic(image, (record,))
    expected = build_history_mosaic(((image, (20, 20, 30, 30)),), panel_height=240)

    assert np.array_equal(result.images[1], expected)
    assert result.prompt.name == "cognitive_visual_mosaic"
    assert "accepted past observations" in result.prompt.user_prompt
    assert "may be imperfect" in result.prompt.user_prompt


def test_vlt_v6_keeps_fixed_three_image_interface_before_dynamic_history():
    anchor = np.zeros((100, 160, 3), dtype=np.uint8)
    current = np.full((100, 160, 3), 23, dtype=np.uint8)
    builder = TrackingContextBuilder(
        IdentityAnchor(0, (10, 10, 20, 30), image=anchor),
        reference_mode=REFERENCE_MODE_VISUAL_BOX,
        prompt_profile=PROMPT_PROFILE_VLT_V6,
        force_history_image=True,
    )

    result = builder.build_mosaic(
        current,
        (),
        target_text="a small gray vehicle",
        semantic_memory="rear view now exposes two white stripes",
    )

    assert result.effective_mode == "mosaic"
    assert len(result.images) == 3
    assert result.reference_frames == (0, 0, 0, 0)
    assert result.prompt.name == "cognitive_vlt_mosaic"
    assert result.prompt.version == "6.4.0"
    assert result.prompt.expected_image_count == 3
    assert "a small gray vehicle" in result.prompt.user_prompt
    assert "rear view now exposes two white stripes" in result.prompt.user_prompt
    assert "Decision order" not in result.prompt.user_prompt
    assert "permanent anchor" in result.prompt.system_prompt
    assert "significant state changes" in result.prompt.system_prompt
    assert "With two images" not in result.prompt.system_prompt
    assert "padding" not in result.prompt.system_prompt
    assert "Return only" not in result.prompt.system_prompt
    assert "bbox_norm1000_xyxy" not in result.prompt.system_prompt
    assert result.images[1].shape[:2] == (240, 1166)
    assert result.history_layout_version == HISTORY_LAYOUT_RECENT_STRIP_3_V2
    assert np.array_equal(result.images[-1], current)


@pytest.mark.parametrize(
    ("scope", "dataset"),
    [
        ("initial_target", "lasot"),
        ("initial_target", "tnl2k"),
        # MGIT 官方 action 层首段：start_frame 经 91/91 序列核对均为 0，不泄漏未来。
        ("first_action_description", "mgit"),
        ("FIRST_ACTION_DESCRIPTION", "mgit"),
        ("  initial_target  ", "lasot"),
    ],
)
def test_safe_init_language_scopes_are_accepted(scope, dataset):
    assert is_unsafe_init_language_scope(scope, dataset=dataset) is False


@pytest.mark.parametrize(
    ("scope", "dataset"),
    [
        ("full_video_story", "mgit"),
        ("full_video_story", "lasot"),
        # MGIT 描述文件含 action/activity/story 三层；scope 缺失或未知时无法确认边界。
        ("", "mgit"),
        ("story", "mgit"),
        ("activity", "mgit"),
    ],
)
def test_unsafe_init_language_scopes_are_rejected(scope, dataset):
    assert is_unsafe_init_language_scope(scope, dataset=dataset) is True


def test_missing_scope_keeps_non_mgit_loader_description():
    """非 MGIT 数据集未声明 scope 时不应被误判为整段剧情。"""

    assert is_unsafe_init_language_scope("", dataset="lasot") is False
    assert is_unsafe_init_language_scope("", dataset="") is False


def test_safe_scope_set_is_explicit():
    """冻结安全集合，避免未来新增 scope 时被静默放行。"""

    assert SAFE_INIT_LANGUAGE_SCOPES == frozenset(
        {"initial_target", "first_action_description"}
    )
