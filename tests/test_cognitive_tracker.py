import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml

from cogtrack.vlm import (
    QwenVLBackend,
    QwenVLConfig,
    VLMBackend,
    VLMResponse,
    clear_qwen_model_cache,
    qwen_model_cache_size,
)
from pytracking.trackers.base import TrackerParams
from pytracking.trackers.cognitive_vlm import CognitiveVLMTracker

#: 替身后端上报的模型像素空间。真实 Qwen 后端从 processor 的 image_grid_thw
#: 回推该值；替身必须同样上报，否则 qwen_abs_pixel 协议会正确地报 parse_error。
FAKE_MODEL_IMAGE_SIZE = (100, 50)


class FakeBackend(VLMBackend):
    def __init__(self, responses, model_image_size=FAKE_MODEL_IMAGE_SIZE):
        self.responses = list(responses)
        self.image_counts = []
        self.model_image_size = model_image_size

    @property
    def model_name(self):
        return "fake-qwen"

    @property
    def is_loaded(self):
        return True

    def generate(self, images, prompt, *, system_prompt=None, generation_config=None):
        del prompt, system_prompt, generation_config
        self.image_counts.append(len(images))
        text = self.responses.pop(0)
        sizes = None
        if self.model_image_size is not None:
            sizes = tuple(self.model_image_size for _ in range(len(images)))
        return VLMResponse(
            text=text,
            model_name=self.model_name,
            latency_ms=1.0,
            prompt_tokens=128,
            generated_tokens=24,
            image_sizes=sizes,
        )


def _response(
    *,
    target_status="present",
    bbox_key="bbox_pixel_xyxy",
    # 测试图为 200x100，替身模型空间为 100x50，缩放系数恰为 2，因此该框映回
    # 原图后与改协议之前的 norm1000 [100,100,300,400] 完全等价，既有断言不变。
    bbox=(10, 5, 30, 20),
    memory_update=None,
    include_memory_update=True,
):
    if target_status == "absent":
        bbox = None
    # 模型可见字段顺序：bbox -> status -> memory_update（必须最后）。
    payload = {
        bbox_key: list(bbox) if bbox is not None else None,
        "status": target_status,
    }
    if include_memory_update:
        payload["memory_update"] = memory_update
    return json.dumps(payload)


def _tracker(
    context_mode="mosaic",
    *,
    responses=None,
    memory_output_enabled=True,
    reference_mode="bbox_text",
    use_init_language=True,
    prompt_profile="visual_v5",
    force_history_image=False,
    bbox_protocol=None,
):
    model_config = Path(__file__).resolve().parents[1] / "configs/models/qwen25vl_7b.yaml"
    if bbox_protocol is None:
        bbox_protocol = "norm1000" if prompt_profile == "vlt_v6" else "qwen_abs_pixel"
    tracker = CognitiveVLMTracker(
        TrackerParams(
            {
                "context_mode": context_mode,
                "reference_mode": reference_mode,
                "prompt_profile": prompt_profile,
                "force_history_image": force_history_image,
                "use_init_language": use_init_language,
                "bbox_protocol": bbox_protocol,
                "model_config": str(model_config),
                "_config_path": str(Path(__file__).resolve()),
                "memory": {
                    "enabled": context_mode == "mosaic",
                    "model_output_enabled": memory_output_enabled,
                    "store_negative": True,
                    "max_positive_records": 3,
                    "max_negative_records": 3,
                    "confirmations": 2,
                    "semantic_confirmations": 2,
                    "max_semantic_confirmation_gap": 300,
                    "min_positive_frame_gap": 0,
                },
            }
        )
    )
    tracker.backend = FakeBackend(responses or [_response(), _response(), _response()])
    return tracker


def test_visual_box_runtime_is_explicit_and_marks_only_anchor():
    tracker = _tracker(
        context_mode="pair",
        reference_mode="visual_box",
        use_init_language=False,
    )
    anchor = np.zeros((100, 200, 3), dtype=np.uint8)
    current = np.full((100, 200, 3), 19, dtype=np.uint8)
    tracker.initialize(
        anchor,
        {
            "init_bbox": [20, 10, 40, 30],
            "init_nlp": "a description that must not enter visual-v5",
            "sequence_name": "visual-v5",
            "frame_path": "0000.jpg",
        },
    )

    context = tracker._build_context(current)
    runtime = tracker.describe_runtime()

    assert context.prompt.name == "cognitive_visual_pair"
    assert context.reference_mode == "visual_box"
    assert np.any(context.images[0] != anchor)
    assert np.array_equal(context.images[-1], current)
    assert "a description that must not enter" not in context.prompt.user_prompt
    assert runtime["reference_mode"] == "visual_box"
    assert runtime["use_init_language"] is False
    assert runtime["mosaic_panel_height"] == 240
    assert runtime["memory_policy"]["semantic_confirmations"] == 2
    assert runtime["memory_policy"]["max_semantic_confirmation_gap"] == 300


def test_presence_only_sft_runtime_uses_the_matching_two_field_protocol():
    tracker = _tracker(
        context_mode="pair",
        responses=[_response(include_memory_update=False)],
        memory_output_enabled=False,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [20, 10, 40, 30], "sequence_name": "sft", "frame_path": "0000.jpg"},
    )

    output = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg"})

    assert output["execution"]["status"] == "ok"
    assert output["target_bbox"] == [20.0, 10.0, 40.0, 30.0]
    assert output["cognition"]["memory_update_proposal"] is None
    assert output["memory_decision"]["semantic"]["accepted"] is False
    assert "二字段" in output["memory_decision"]["semantic"]["reason"]
    assert "memory_update" not in tracker._build_context(image).prompt.user_prompt


def test_tracker_pair_warmup_then_mosaic_and_skip():
    tracker = _tracker()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    init = tracker.initialize(
        image,
        {
            "init_bbox": [20, 10, 40, 30],
            "init_nlp": "red target",
            "sequence_name": "synthetic",
            "frame_path": "0000.jpg",
        },
    )
    assert init["prediction"]["target_presence"] == "present"
    assert init["candidate_bbox"] == [20.0, 10.0, 40.0, 30.0]
    assert init["target_bbox"] == init["candidate_bbox"]
    assert init["committed_target_presence"] == "present"
    assert init["commit_decision"]["accepted"] is True
    assert init["commit_decision"]["source"] == "initialization_ground_truth"

    first = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg", "is_observation_frame": True})
    second = tracker.track(image, {"frame_num": 2, "frame_path": "0002.jpg", "is_observation_frame": True})
    third = tracker.track(image, {"frame_num": 3, "frame_path": "0003.jpg", "is_observation_frame": True})
    skipped = tracker.track(image, {"frame_num": 4, "frame_path": "0004.jpg", "is_observation_frame": False})

    assert first["target_bbox"] == [20.0, 10.0, 40.0, 30.0]
    assert first["target_bbox"] == first["candidate_bbox"]
    assert first["commit_decision"]["accepted"] is True
    assert first["context"]["prompt_tokens"] == 128
    assert first["context"]["generated_tokens"] == 24
    assert second["memory_decision"]["visual"]["accepted"] is True
    assert second["cognition"]["memory_updated"] is False
    assert tracker.backend.image_counts == [2, 2, 3]
    assert third["execution"]["status"] == "ok"
    assert skipped["execution"]["status"] == "skipped"
    assert skipped["prediction"] is None
    assert skipped["candidate_bbox"] is None
    assert skipped["committed_target_presence"] is None
    assert skipped["commit_decision"]["accepted"] is False


def test_parse_failure_is_not_absent():
    tracker = _tracker(context_mode="pair")
    tracker.backend = FakeBackend(["invalid output"])
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [20, 10, 40, 30], "sequence_name": "synthetic", "frame_path": "0000.jpg"},
    )
    output = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg", "is_observation_frame": True})
    assert output["execution"]["status"] == "parse_error"
    assert output["prediction"] is None
    assert output["target_bbox"] is None
    assert output["committed_target_presence"] is None
    assert output["candidate_bbox"] is None
    assert output["commit_decision"]["accepted"] is False


def test_absent_prediction_is_not_committed():
    tracker = _tracker(
        context_mode="pair",
        responses=[_response(target_status="absent")],
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [20, 10, 40, 30], "sequence_name": "weak", "frame_path": "0000.jpg"},
    )

    output = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg", "is_observation_frame": True})

    assert output["candidate_bbox"] is None
    assert output["prediction"]["bbox_xywh"] is None
    assert output["target_bbox"] is None
    assert output["committed_target_presence"] == "absent"
    assert output["commit_decision"]["accepted"] is False
    assert "target_presence" in output["commit_decision"]["reason"]
    # 未提交候选不能污染内部 last_trusted_bbox。
    assert output["cognitive_state"]["last_trusted_bbox"] == [20.0, 10.0, 40.0, 30.0]
    assert output["cognitive_state"]["last_seen_frame"] == 0


def test_absent_prediction_does_not_enter_positive_memory():
    tracker = _tracker(
        context_mode="pair",
        responses=[_response(target_status="absent")],
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [5, 5, 20, 20], "sequence_name": "distractor", "frame_path": "0000.jpg"},
    )

    output = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg", "is_observation_frame": True})

    assert output["candidate_bbox"] is None
    assert output["target_bbox"] is None
    assert output["committed_target_presence"] == "absent"
    assert output["commit_decision"]["accepted"] is False
    assert output["memory_decision"]["semantic"]["accepted"] is False
    assert output["memory"]["records"]["negative"] == []
    assert output["memory"]["records"]["positive"] == []


def test_model_controlled_semantic_memory_requires_two_proposals_then_is_reused():
    tracker = _tracker(
        context_mode="pair",
        responses=[
            _response(memory_update="Rear view reveals two stable white stripes."),
            _response(memory_update="Rear view now reveals two stable white stripes."),
            _response(),
        ],
    )
    prompts = []
    original_generate = tracker.backend.generate

    def recording_generate(images, prompt, **kwargs):
        prompts.append(prompt)
        return original_generate(images, prompt, **kwargs)

    tracker.backend.generate = recording_generate
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [20, 10, 40, 30], "sequence_name": "memory", "frame_path": "0000.jpg"},
    )

    first = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg"})
    second = tracker.track(image, {"frame_num": 2, "frame_path": "0002.jpg"})
    third = tracker.track(image, {"frame_num": 3, "frame_path": "0003.jpg"})

    assert first["cognition"]["memory_update_proposal"] == (
        "Rear view reveals two stable white stripes."
    )
    assert first["cognition"]["memory_updated"] is False
    assert first["memory_decision"]["semantic"]["accepted"] is False
    assert second["memory_decision"]["semantic"]["accepted"] is True
    assert second["memory"]["records"]["semantic"][0]["text"] == (
        "Rear view now reveals two stable white stripes."
    )
    assert "Rear view now reveals two stable white stripes." in prompts[2]
    assert third["cognition"]["memory_updated"] is False


def test_absent_output_preserves_disappearance_memory_proposal():
    tracker = _tracker(
        context_mode="pair",
        responses=[
            _response(
                target_status="absent",
                memory_update="The target changed appearance.",
            )
        ],
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [20, 10, 40, 30], "sequence_name": "invalid", "frame_path": "0000.jpg"},
    )

    output = tracker.track(image, {"frame_num": 1, "frame_path": "0001.jpg"})

    assert output["execution"]["status"] == "ok"
    assert output["prediction"]["target_presence"] == "absent"
    assert output["cognition"]["memory_update_proposal"] == "The target changed appearance."
    assert output["cognition"]["memory_update_error"] is None
    assert output["memory_decision"]["semantic"]["accepted"] is True
    assert output["memory"]["records"]["semantic"][0]["metadata"]["temporal_event"] == (
        "disappearance"
    )


def test_disappearance_and_reappearance_updates_bypass_regular_semantic_cooldown():
    tracker = _tracker(
        context_mode="mosaic",
        reference_mode="visual_box",
        prompt_profile="vlt_v6",
        force_history_image=True,
        responses=[
            _response(
                target_status="absent",
                bbox_key="bbox_2d",
                memory_update="The initialized target is currently absent from the scene.",
            ),
            _response(
                bbox_key="bbox_2d",
                bbox=(100, 100, 300, 400),
                memory_update=(
                    "The initialized target has reappeared, now showing its rear white stripes."
                ),
            ),
        ],
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {"init_bbox": [20, 10, 40, 30], "sequence_name": "transition-memory"},
    )

    disappeared = tracker.track(image, {"frame_num": 1})
    reappeared = tracker.track(image, {"frame_num": 2})

    assert disappeared["memory_decision"]["semantic"]["accepted"] is True
    assert reappeared["memory_decision"]["semantic"]["accepted"] is True
    assert reappeared["memory"]["records"]["semantic"][-1]["metadata"][
        "temporal_event"
    ] == "reappearance"
    assert "has reappeared" in tracker._build_context(image).prompt.user_prompt


def test_continued_absence_cannot_rewrite_disappearance_memory():
    tracker = _tracker(
        context_mode="mosaic",
        reference_mode="visual_box",
        prompt_profile="vlt_v6",
        force_history_image=True,
        responses=[
            _response(
                target_status="absent",
                bbox_key="bbox_2d",
                memory_update="The initialized target is currently absent from the scene.",
            ),
            _response(
                target_status="absent",
                bbox_key="bbox_2d",
                memory_update="The target remains absent.",
            ),
        ],
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(image, {"init_bbox": [20, 10, 40, 30], "sequence_name": "absent"})

    first = tracker.track(image, {"frame_num": 1})
    second = tracker.track(image, {"frame_num": 2})

    assert first["memory_decision"]["semantic"]["accepted"] is True
    assert second["memory_decision"]["semantic"]["accepted"] is False
    assert "首次消失" in second["memory_decision"]["semantic"]["reason"]
    assert len(second["memory"]["records"]["semantic"]) == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_mode": "pair"}, "context_mode=mosaic"),
        ({"force_history_image": False}, "force_history_image=true"),
        ({"bbox_protocol": "qwen_abs_pixel"}, "bbox_protocol=norm1000"),
    ],
)
def test_vlt_v6_rejects_runtime_protocol_drift(overrides, message):
    kwargs = {
        "context_mode": "mosaic",
        "reference_mode": "visual_box",
        "prompt_profile": "vlt_v6",
        "force_history_image": True,
        "bbox_protocol": "norm1000",
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        _tracker(**kwargs)


def test_qwen_runtime_cache_is_process_shared_and_thread_safe():
    """fake factory 不加载真实权重，仍能验证并发首帧只加载一次。"""

    clear_qwen_model_cache()
    factory_calls = []
    shared_model = object()
    shared_processor = object()

    def fake_factory(config):
        factory_calls.append(config.model_path)
        return shared_model, shared_processor

    config = QwenVLConfig(model_path="/tmp/fake-qwen-for-cache-test", attn_implementation=None)
    backends = [QwenVLBackend(config, runtime_factory=fake_factory) for _ in range(8)]
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda backend: backend._load_once(), backends))

        assert factory_calls == [config.model_path]
        assert qwen_model_cache_size() == 1
        assert all(backend.is_loaded for backend in backends)
        assert all(backend._runtime.model is shared_model for backend in backends)

        # 关闭一个序列只丢弃句柄，其他序列仍复用权重。
        backends[0].release()
        assert backends[0].is_loaded is False
        assert backends[1].is_loaded is True
        assert qwen_model_cache_size() == 1
    finally:
        clear_qwen_model_cache()

    assert all(backend.is_loaded is False for backend in backends)
    assert qwen_model_cache_size() == 0


def test_tracker_sequence_state_is_not_shared():
    first = _tracker(context_mode="pair", responses=[_response(target_status="absent")])
    second = _tracker(context_mode="pair", responses=[_response()])
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    first.initialize(
        image,
        {"init_bbox": [1, 2, 30, 40], "sequence_name": "sequence-a", "frame_path": "a0.jpg"},
    )
    first.track(image, {"frame_num": 1, "frame_path": "a1.jpg", "is_observation_frame": True})
    second_output = second.initialize(
        image,
        {"init_bbox": [50, 20, 10, 15], "sequence_name": "sequence-b", "frame_path": "b0.jpg"},
    )

    assert first.sequence_name == "sequence-a"
    assert second.sequence_name == "sequence-b"
    assert first.state_machine.state.frame_id == 1
    assert second.state_machine.state.frame_id == 0
    assert len(first.memory_bank) == 0
    assert len(second.memory_bank) == 0
    assert second_output["memory"]["anchor"]["bbox_xywh"] == [50.0, 20.0, 10.0, 15.0]


def test_vlt_v6_runtime_uses_safe_initial_text_latest_memory_and_three_images():
    tracker = _tracker(
        context_mode="mosaic",
        reference_mode="visual_box",
        prompt_profile="vlt_v6",
        force_history_image=True,
        use_init_language=True,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {
            "init_bbox": [20, 10, 40, 30],
            "init_nlp": "a small gray vehicle",
            "init_language_scope": "initial_target",
            "dataset_name": "tnl2k",
            "sequence_name": "vlt-v6",
        },
    )

    context = tracker._build_context(image)

    assert len(context.images) == 3
    assert context.prompt.name == "cognitive_vlt_mosaic"
    assert context.prompt.version == "6.4.0"
    assert context.reference_frames == (0, 0, 0, 0)
    assert context.images[1].shape[:2] == (240, 1454)
    assert "a small gray vehicle" in context.prompt.user_prompt
    runtime = tracker.describe_runtime()
    assert runtime["target_text_source"] == "dataset_initial_language"
    assert runtime["history_layout_version"] == "recent_strip_3_v2"


def test_vlt_v6_rejects_mgit_full_video_story_from_online_input():
    tracker = _tracker(
        context_mode="mosaic",
        reference_mode="visual_box",
        prompt_profile="vlt_v6",
        force_history_image=True,
        use_init_language=True,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tracker.initialize(
        image,
        {
            "init_bbox": [20, 10, 40, 30],
            "init_nlp": "Later in the video the target disappears behind a tree.",
            "init_language_scope": "full_video_story",
            "dataset_name": "mgit",
            "sequence_name": "no-future-leak",
        },
    )

    prompt = tracker._build_context(image).prompt.user_prompt
    assert "Later in the video" not in prompt
    assert "target marked by the red box" in prompt
    assert tracker.target_text_source == "visual_anchor_fallback"


def test_vlt_v6_accepts_mgit_first_action_description_online():
    """MGIT 官方 action 首段是初始化时刻可得的描述，在线必须使用它。

    这条与训练导出侧共用 ``is_unsafe_init_language_scope``；若两侧判据漂移，同一
    序列会在训练和推理拿到不同初始文本，因此这里显式锁定在线行为。
    """

    tracker = _tracker(
        context_mode="mosaic",
        reference_mode="visual_box",
        prompt_profile="vlt_v6",
        force_history_image=True,
        use_init_language=True,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    description = "The garfield waks up a orange striped clothes actor in the bedroom"
    tracker.initialize(
        image,
        {
            "init_bbox": [20, 10, 40, 30],
            "init_nlp": description,
            "init_language_scope": "first_action_description",
            "dataset_name": "mgit",
            "sequence_name": "002",
        },
    )

    prompt = tracker._build_context(image).prompt.user_prompt
    assert description in prompt
    assert "target marked by the red box" not in prompt
    assert tracker.describe_runtime()["target_text_source"] == "dataset_initial_language"
    assert tracker.target_text_source == "dataset_initial_language"


def test_vlt_v640_full_sft_configs_do_not_expect_lora_adapter():
    project_root = Path(__file__).resolve().parents[1]
    local_model = yaml.safe_load(
        (project_root / "configs/models/qwen3vl_4b_vlt_v640_sft.yaml").read_text(
            encoding="utf-8"
        )
    )
    local_tracker = yaml.safe_load(
        (project_root / "configs/trackers/qwen3vl_4b_vlt_v640_sft.yaml").read_text(
            encoding="utf-8"
        )
    )
    vllm_tracker = yaml.safe_load(
        (project_root / "configs/trackers/qwen3vl_4b_vlt_v640_sft_vllm.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert local_model["model_path"] == "${COGTRACK_QWEN3_4B_SFT_MODEL}"
    assert "adapter_path" not in local_model
    for tracker in (local_tracker, vllm_tracker):
        assert tracker["context_mode"] == "mosaic"
        assert tracker["reference_mode"] == "visual_box"
        assert tracker["bbox_protocol"] == "norm1000"
