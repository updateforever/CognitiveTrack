#!/usr/bin/env python3
"""快速测试数据生成流程（只处理 1 个序列）。

用于验证代码逻辑，不需要运行完整数据集。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cogtrack.training.sample_builder import SampleBuilder, SampleType
from cogtrack.training.state_generator import StateGenerator
from pytracking.datasets.mgit import MGITDataset
from pytracking.evaluation.environment import EnvironmentSettings


def main():
    """快速测试"""
    print("=" * 80)
    print("VLT-v6.3.1 Core SFT Data Generation - Quick Test")
    print("=" * 80)
    print()

    # 配置
    output_dir = Path("/tmp/vlt_v631_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_base_url = os.getenv("LOCAL_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("LOCAL_VLLM_API_KEY", "local-test-key")

    print(f"Output directory: {output_dir}")
    print(f"vLLM API: {api_base_url}")
    print()

    # 初始化
    env = EnvironmentSettings()
    mgit_test = MGITDataset(env, split="test", version="tiny")

    sample_builder = SampleBuilder(
        max_frame_span=200,
        history_buffer_size=3,
        history_sample_interval=10,
    )

    state_generator = StateGenerator(
        api_base_url=api_base_url,
        api_key=api_key,
        temperature=0.7,
    )

    # 只处理第一个序列
    sequence_list = mgit_test.get_sequence_list()
    seq = sequence_list[0]

    print(f"Testing with sequence: {seq.name}")
    print(f"Sequence length: {len(seq.frames)} frames")
    print()

    # 采样 3 个样本
    frames = [Path(f) for f in seq.frames]
    gt_visible = seq.target_visible if seq.target_visible is not None else [True] * len(frames)

    samples = sample_builder.sample_from_sequence(
        sequence_name=seq.name,
        dataset_name="mgit",
        frames=frames,
        gt_bboxes=seq.ground_truth_rect,
        gt_visible=gt_visible,
        num_samples=3,
    )

    print(f"Generated {len(samples)} samples")
    print()

    # 为每个样本生成状态描述
    for i, sample in enumerate(samples, 1):
        print(f"Processing sample {i}/{len(samples)}...")

        # 生成初始身份
        sample.initial_identity = state_generator.generate_initial_identity(
            init_frame_path=sample.init_frame.frame_path,
            bbox=sample.init_frame.bbox,
            language_query=seq.language_query,
        )
        print(f"  Initial identity: {sample.initial_identity}")

        # 设置 previous_state
        sample.previous_state = sample.initial_identity

        # 确定 target_status
        sample.target_status = "present" if sample.current_frame.visible else "absent"
        print(f"  Target status: {sample.target_status}")

        # 生成状态更新
        sample.memory_update = state_generator.generate_state_update(
            initial_identity=sample.initial_identity,
            previous_state=sample.previous_state,
            current_frame_path=sample.current_frame.frame_path,
            target_status=sample.target_status,
            bbox=sample.current_frame.bbox if sample.current_frame.visible else None,
        )
        print(f"  Memory update: {sample.memory_update}")

        # 保存
        annotation = {
            "sample_id": i,
            "sequence_name": sample.sequence_name,
            "dataset_name": sample.dataset_name,
            "sample_type": sample.sample_type.value,
            "initial_identity": sample.initial_identity,
            "previous_state": sample.previous_state,
            "target_status": sample.target_status,
            "memory_update": sample.memory_update,
        }

        output_path = output_dir / f"sample_{i:03d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {output_path}")
        print()

    print("=" * 80)
    print("✅ Quick test complete!")
    print(f"Check results in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
