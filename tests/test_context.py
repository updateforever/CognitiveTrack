import numpy as np

from cogtrack.context import TrackingContextBuilder
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
        for frame_id in (4, 8, 12, 16)
    )
    result = builder.build_mosaic(image, records, "target")

    # 四帧按 2x2 排列；没有旧实现额外添加的 30px 帧号标题行。
    assert result.images[1].shape[:2] == (480, 768)
    assert result.reference_frames == (0, 4, 8, 12, 16)
