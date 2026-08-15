"""VLT-v6.3.1 Core SFT 数据生成主脚本。

从 LaSOT/TNL2K/MGIT 训练集生成约 30k 训练样本。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cogtrack.training.sample_builder import SampleBuilder, SampleType, TrainingSample
from cogtrack.training.state_generator import StateGenerator
from pytracking.datasets.lasot import LaSOTDataset
from pytracking.datasets.mgit import MGITDataset
from pytracking.datasets.tnl2k import TNL2KDataset
from pytracking.evaluation.environment import EnvironmentSettings


def get_samples_per_sequence(seq_length: int) -> int:
    """根据序列长度确定采样数量（调研优化：增加密度以达到 50K+ 目标）"""
    if seq_length < 300:
        return 8    # 原 5
    elif seq_length < 1000:
        return 15   # 原 10
    else:
        return 30   # 原 20


def save_sample_annotation(sample: TrainingSample, output_dir: Path, sample_id: int):
    """保存单个样本的标注文件"""
    annotation = {
        "sample_id": sample_id,
        "sequence_name": sample.sequence_name,
        "dataset_name": sample.dataset_name,
        "sample_type": sample.sample_type.value,
        # 帧信息
        "init_frame": {
            "frame_id": sample.init_frame.frame_id,
            "frame_path": str(sample.init_frame.frame_path),
            "bbox": sample.init_frame.bbox,
            "visible": sample.init_frame.visible,
        },
        "history_frames": [
            {
                "frame_id": f.frame_id,
                "frame_path": str(f.frame_path),
                "bbox": f.bbox,
                "visible": f.visible,
            }
            for f in sample.history_frames
        ],
        "current_frame": {
            "frame_id": sample.current_frame.frame_id,
            "frame_path": str(sample.current_frame.frame_path),
            "bbox": sample.current_frame.bbox,
            "visible": sample.current_frame.visible,
        },
        # 标注信息
        "initial_identity": sample.initial_identity,
        "previous_state": sample.previous_state,
        "target_status": sample.target_status,
        "memory_update": sample.memory_update,
    }

    # 保存 JSON
    output_path = output_dir / f"sample_{sample_id:06d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


def generate_from_dataset(
    dataset_name: str,
    dataset,
    sample_builder: SampleBuilder,
    state_generator: StateGenerator,
    output_dir: Path,
    start_sample_id: int,
) -> int:
    """从单个数据集生成训练样本"""
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} dataset...")
    print(f"{'='*80}\n")

    sequence_list = dataset.get_sequence_list()
    sample_id = start_sample_id

    for seq in tqdm(sequence_list, desc=f"{dataset_name}", unit="seq"):
        seq_length = len(seq.frames)
        num_samples = get_samples_per_sequence(seq_length)

        # 构造 gt_visible
        if seq.target_visible is not None:
            gt_visible = seq.target_visible
        else:
            # 如果没有 visible 标注，从 bbox 推断（面积 > 0 认为可见）
            gt_visible = np.array([
                (bbox[2] > 0 and bbox[3] > 0) for bbox in seq.ground_truth_rect
            ])

        # 采样训练样本
        samples = sample_builder.sample_from_sequence(
            sequence_name=seq.name,
            dataset_name=dataset_name,
            frames=[Path(f) for f in seq.frames],
            gt_bboxes=seq.ground_truth_rect,
            gt_visible=gt_visible,
            num_samples=num_samples,
        )

        # 为每个样本生成状态描述
        for sample in samples:
            # 1. 生成初始身份描述
            sample.initial_identity = state_generator.generate_initial_identity(
                init_frame_path=sample.init_frame.frame_path,
                bbox=sample.init_frame.bbox,
                language_query=seq.language_query,
            )

            # 2. 确定 previous_state（从历史最后一帧）
            if sample.history_frames:
                last_history = sample.history_frames[-1]
                # 简化：初始时 previous_state = initial_identity
                # TODO: 可以用大模型生成历史帧的状态
                sample.previous_state = sample.initial_identity
            else:
                sample.previous_state = sample.initial_identity

            # 3. 确定当前帧的 target_status
            sample.target_status = "present" if sample.current_frame.visible else "absent"

            # 4. 生成状态更新
            sample.memory_update = state_generator.generate_state_update(
                initial_identity=sample.initial_identity,
                previous_state=sample.previous_state,
                current_frame_path=sample.current_frame.frame_path,
                target_status=sample.target_status,
                bbox=sample.current_frame.bbox if sample.current_frame.visible else None,
            )

            # 5. 保存标注
            save_sample_annotation(sample, output_dir, sample_id)
            sample_id += 1

    return sample_id


def main():
    """主函数"""
    # 配置参数
    output_dir = Path("/data2/wyp/VLMTrack/CognitiveTrack/data/vlt_v631_core_sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从环境变量读取 vLLM 配置
    api_base_url = os.getenv("LOCAL_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("LOCAL_VLLM_API_KEY", "local-test-key")

    print(f"Output directory: {output_dir}")
    print(f"vLLM API: {api_base_url}")
    print()

    # 初始化环境
    env = EnvironmentSettings()

    # 初始化构造器
    sample_builder = SampleBuilder(
        max_frame_span=200,
        history_buffer_size=3,
        history_sample_interval=10,
        sample_type_ratios={
            SampleType.PURE_POSITIVE: 0.70,   # 调研优化：70% (原 60%)
            SampleType.CURRENT_ABSENT: 0.15,  # 保持 15%
            SampleType.HISTORY_NOISY: 0.10,   # 调整：10% (原 15%)
            SampleType.MIXED_HARD: 0.05,      # 调整：5% (原 10%)
        },
    )

    state_generator = StateGenerator(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name="Qwen2.5-VL-32B-Instruct",
        temperature=0.7,
        max_tokens=256,
    )

    # 加载数据集
    print("Loading datasets...")
    lasot_train = LaSOTDataset(env, split="train")
    tnl2k_train = TNL2KDataset(env, split="train")
    mgit_train = MGITDataset(env, split="train", version="full")

    print(f"LaSOT train: {len(lasot_train)} sequences")
    print(f"TNL2K train: {len(tnl2k_train)} sequences")
    print(f"MGIT train: {len(mgit_train)} sequences")
    print()

    # 生成数据
    sample_id = 0

    # LaSOT
    sample_id = generate_from_dataset(
        dataset_name="lasot",
        dataset=lasot_train,
        sample_builder=sample_builder,
        state_generator=state_generator,
        output_dir=output_dir,
        start_sample_id=sample_id,
    )

    # TNL2K
    sample_id = generate_from_dataset(
        dataset_name="tnl2k",
        dataset=tnl2k_train,
        sample_builder=sample_builder,
        state_generator=state_generator,
        output_dir=output_dir,
        start_sample_id=sample_id,
    )

    # MGIT
    sample_id = generate_from_dataset(
        dataset_name="mgit",
        dataset=mgit_train,
        sample_builder=sample_builder,
        state_generator=state_generator,
        output_dir=output_dir,
        start_sample_id=sample_id,
    )

    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"Total samples: {sample_id}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # 保存统计信息
    stats = {
        "total_samples": sample_id,
        "datasets": {
            "lasot": len(lasot_train),
            "tnl2k": len(tnl2k_train),
            "mgit": len(mgit_train),
        },
        "sample_builder_config": {
            "max_frame_span": sample_builder.max_frame_span,
            "history_buffer_size": sample_builder.history_buffer_size,
            "history_sample_interval": sample_builder.history_sample_interval,
            "sample_type_ratios": {k.value: v for k, v in sample_builder.sample_type_ratios.items()},
        },
    }

    with open(output_dir / "generation_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
