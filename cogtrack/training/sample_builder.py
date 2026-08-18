"""VLT-v6.3.1 训练样本构造器。

负责从序列中采样帧、构造正负样本、拼接 history mosaic。
不包含状态描述生成（由 state_generator 负责）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class SampleType(Enum):
    """训练样本类型"""

    PURE_POSITIVE = "pure_positive"  # 当前帧存在，历史全部正确
    CURRENT_ABSENT = "current_absent"  # 当前帧目标缺失
    HISTORY_NOISY = "history_noisy"  # 历史帧有预测错误
    MIXED_HARD = "mixed_hard"  # 当前缺失 + 历史噪声


@dataclass
class FrameInfo:
    """单帧信息"""

    frame_id: int
    frame_path: Path
    bbox: tuple[float, float, float, float]  # (x, y, w, h)
    visible: bool


@dataclass
class TrainingSample:
    """训练样本结构"""

    # 序列信息
    sequence_name: str
    dataset_name: str

    # 帧信息
    init_frame: FrameInfo
    history_frames: list[FrameInfo]  # 3 帧
    current_frame: FrameInfo

    # 样本类型
    sample_type: SampleType

    # 标注信息（待填充）
    initial_identity: str = ""
    previous_state: str = ""
    target_status: str = ""  # "present" / "absent"
    memory_update: str = ""


class SampleBuilder:
    """训练样本构造器"""

    def __init__(
        self,
        *,
        max_frame_span: int = 200,
        history_buffer_size: int = 3,
        history_sample_interval: int = 10,
        sample_type_ratios: dict[SampleType, float] | None = None,
    ):
        """
        Args:
            max_frame_span: 初始化帧到当前帧的最大间隔
            history_buffer_size: 历史帧采样数量
            history_sample_interval: 历史帧采样间隔
            sample_type_ratios: 各类样本的占比
        """
        self.max_frame_span = max_frame_span
        self.history_buffer_size = history_buffer_size
        self.history_sample_interval = history_sample_interval

        # 默认样本类型分布（调研优化：70/15/10/5，present:absent ≈ 8:2）
        self.sample_type_ratios = sample_type_ratios or {
            SampleType.PURE_POSITIVE: 0.70,
            SampleType.CURRENT_ABSENT: 0.15,
            SampleType.HISTORY_NOISY: 0.10,
            SampleType.MIXED_HARD: 0.05,
        }

        # 验证比例和为 1
        total = sum(self.sample_type_ratios.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"样本类型比例之和必须为 1，当前为 {total}")

    def allocate_sample_types(self, num_samples: int) -> list[SampleType]:
        """按比例分配样本类型"""
        types = []
        for sample_type, ratio in self.sample_type_ratios.items():
            count = int(num_samples * ratio)
            types.extend([sample_type] * count)

        # 补齐剩余样本（用 PURE_POSITIVE）
        while len(types) < num_samples:
            types.append(SampleType.PURE_POSITIVE)

        # 打乱顺序
        random.shuffle(types)
        return types

    def sample_from_sequence(
        self,
        sequence_name: str,
        dataset_name: str,
        frames: list[Path],
        gt_bboxes: np.ndarray,
        gt_visible: np.ndarray | None,
        num_samples: int,
    ) -> list[TrainingSample]:
        """从单个序列中采样多个训练样本"""
        if len(frames) != len(gt_bboxes):
            raise ValueError(f"帧数 {len(frames)} 与 bbox 数量 {len(gt_bboxes)} 不匹配")

        if gt_visible is not None and len(gt_visible) != len(frames):
            raise ValueError(f"帧数 {len(frames)} 与 visible 数量 {len(gt_visible)} 不匹配")

        # 分配样本类型
        sample_types = self.allocate_sample_types(num_samples)

        samples = []
        for sample_type in sample_types:
            sample = self._construct_single_sample(
                sequence_name=sequence_name,
                dataset_name=dataset_name,
                frames=frames,
                gt_bboxes=gt_bboxes,
                gt_visible=gt_visible,
                sample_type=sample_type,
            )
            if sample:
                samples.append(sample)

        return samples

    def _construct_single_sample(
        self,
        sequence_name: str,
        dataset_name: str,
        frames: list[Path],
        gt_bboxes: np.ndarray,
        gt_visible: np.ndarray | None,
        sample_type: SampleType,
    ) -> TrainingSample | None:
        """构造单个训练样本"""
        seq_len = len(frames)

        # 1. 采样初始化帧
        max_init = max(0, seq_len - self.max_frame_span - 1)
        if max_init < 0:
            return None
        init_idx = random.randint(0, max_init)

        # 2. 根据样本类型选择当前帧
        current_idx = self._sample_current_frame(
            init_idx=init_idx,
            seq_len=seq_len,
            gt_visible=gt_visible,
            sample_type=sample_type,
        )
        if current_idx is None:
            return None

        # 3. 采样历史帧
        history_indices = self._sample_history_frames(init_idx, current_idx)
        if not history_indices:
            return None

        # 4. 构造 FrameInfo
        init_frame = FrameInfo(
            frame_id=init_idx,
            frame_path=frames[init_idx],
            bbox=tuple(gt_bboxes[init_idx]),
            visible=gt_visible[init_idx] if gt_visible is not None else True,
        )

        history_frames = [
            FrameInfo(
                frame_id=idx,
                frame_path=frames[idx],
                bbox=tuple(gt_bboxes[idx]),
                visible=gt_visible[idx] if gt_visible is not None else True,
            )
            for idx in history_indices
        ]

        current_frame = FrameInfo(
            frame_id=current_idx,
            frame_path=frames[current_idx],
            bbox=tuple(gt_bboxes[current_idx]),
            visible=gt_visible[current_idx] if gt_visible is not None else True,
        )

        # 5. 根据样本类型添加噪声
        if sample_type == SampleType.HISTORY_NOISY or sample_type == SampleType.MIXED_HARD:
            history_frames = self._add_noise_to_history(history_frames)

        if sample_type == SampleType.CURRENT_ABSENT or sample_type == SampleType.MIXED_HARD:
            current_frame.visible = False
            current_frame.bbox = (0.0, 0.0, 0.0, 0.0)

        return TrainingSample(
            sequence_name=sequence_name,
            dataset_name=dataset_name,
            init_frame=init_frame,
            history_frames=history_frames,
            current_frame=current_frame,
            sample_type=sample_type,
        )

    def _sample_current_frame(
        self,
        init_idx: int,
        seq_len: int,
        gt_visible: np.ndarray | None,
        sample_type: SampleType,
    ) -> int | None:
        """根据样本类型采样当前帧"""
        min_span = max(10, self.history_buffer_size * self.history_sample_interval)
        max_span = min(self.max_frame_span, seq_len - init_idx - 1)

        if max_span < min_span:
            return None

        # 构造候选范围
        candidates = list(range(init_idx + min_span, init_idx + max_span + 1))

        # 根据样本类型过滤候选
        if sample_type == SampleType.CURRENT_ABSENT or sample_type == SampleType.MIXED_HARD:
            # 优先选择 GT 标注为 absent 的帧
            if gt_visible is not None:
                absent_candidates = [idx for idx in candidates if not gt_visible[idx]]
                if absent_candidates:
                    return random.choice(absent_candidates)
            # 如果没有标注，随机选择（后续会手动设置为 absent）
            return random.choice(candidates)
        else:
            # 优先选择 GT 标注为 present 的帧
            if gt_visible is not None:
                present_candidates = [idx for idx in candidates if gt_visible[idx]]
                if present_candidates:
                    return random.choice(present_candidates)
            return random.choice(candidates)

    def _sample_history_frames(self, init_idx: int, current_idx: int) -> list[int]:
        """在 [init_idx, current_idx) 之间采样历史帧"""
        span = current_idx - init_idx
        if span < self.history_sample_interval:
            return []

        # 从 current_idx 往前采样
        history_indices = []
        for i in range(self.history_buffer_size):
            offset = (i + 1) * self.history_sample_interval
            idx = current_idx - offset
            if idx > init_idx:
                history_indices.append(idx)

        # 反转，使其按时间顺序排列
        history_indices.reverse()
        return history_indices

    def _add_noise_to_history(self, history_frames: list[FrameInfo]) -> list[FrameInfo]:
        """对历史帧添加噪声（bbox 扰动或设置为 absent）"""
        if not history_frames:
            return history_frames

        # 随机选择 1-2 个帧添加噪声
        num_noisy = random.randint(1, min(2, len(history_frames)))
        noisy_indices = random.sample(range(len(history_frames)), num_noisy)

        for idx in noisy_indices:
            frame = history_frames[idx]
            if random.random() < 0.5:
                # 方式 1：bbox 扰动
                x, y, w, h = frame.bbox
                shift_x = random.uniform(-0.2, 0.2) * w
                shift_y = random.uniform(-0.2, 0.2) * h
                scale = random.uniform(0.7, 1.3)
                frame.bbox = (x + shift_x, y + shift_y, w * scale, h * scale)
            else:
                # 方式 2：设置为 absent
                frame.visible = False
                frame.bbox = (0.0, 0.0, 0.0, 0.0)

        return history_frames

    def create_history_mosaic(self, history_frames: list[FrameInfo], target_height: int = 256) -> Image.Image:
        """将历史帧拼接成 1×N 的 mosaic 图"""
        if not history_frames:
            raise ValueError("历史帧列表为空")

        # 加载图像
        images = []
        for frame_info in history_frames:
            img = Image.open(frame_info.frame_path).convert("RGB")
            # 调整高度，保持宽高比
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            img = img.resize((new_width, target_height), Image.LANCZOS)
            images.append(img)

        # 水平拼接
        total_width = sum(img.width for img in images)
        mosaic = Image.new("RGB", (total_width, target_height))

        x_offset = 0
        for img in images:
            mosaic.paste(img, (x_offset, 0))
            x_offset += img.width

        return mosaic


__all__ = ["SampleBuilder", "SampleType", "FrameInfo", "TrainingSample"]
