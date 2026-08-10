"""SUTrack 独立运行时的裁剪、归一化、坐标变换与 Hann window。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
import torch

from cogtrack.protocol import validate_xywh


class ImagePreprocessor:
    """把 RGB/RGB+辅助三通道图像转成 ImageNet 归一化 Tensor。"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        base_mean = [0.485, 0.456, 0.406]
        base_std = [0.229, 0.224, 0.225]
        self.mean = {
            3: torch.tensor(base_mean, device=device).view(1, 3, 1, 1),
            6: torch.tensor(base_mean * 2, device=device).view(1, 6, 1, 1),
        }
        self.std = {
            3: torch.tensor(base_std, device=device).view(1, 3, 1, 1),
            6: torch.tensor(base_std * 2, device=device).view(1, 6, 1, 1),
        }

    def process(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3 or image.shape[2] not in self.mean:
            raise ValueError(f"SUTrack patch 必须是 HxWx3/6，收到 {image.shape}")
        channels = int(image.shape[2])
        tensor = torch.as_tensor(np.ascontiguousarray(image), device=self.device)
        tensor = tensor.float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return (tensor - self.mean[channels]) / self.std[channels]


def sample_target(
    image: np.ndarray,
    bbox_xywh: Sequence[float],
    search_area_factor: float,
    output_size: int,
) -> tuple[np.ndarray, float]:
    """以目标为中心提取正方形区域，越界部分补零并缩放到固定尺寸。"""

    x, y, width, height = validate_xywh(bbox_xywh)
    crop_size = math.ceil(math.sqrt(width * height) * float(search_area_factor))
    if crop_size < 1:
        raise ValueError("SUTrack crop 尺寸必须大于 0")
    x1 = round(x + 0.5 * width - 0.5 * crop_size)
    y1 = round(y + 0.5 * height - 0.5 * crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    left = max(0, -x1)
    right = max(x2 - image.shape[1] + 1, 0)
    top = max(0, -y1)
    bottom = max(y2 - image.shape[0] + 1, 0)
    crop = image[y1 + top : y2 - bottom, x1 + left : x2 - right]
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
    resized = cv2.resize(crop, (int(output_size), int(output_size)))
    return resized, float(output_size) / crop_size


def transform_image_to_crop(
    bbox_xywh: Sequence[float],
    extraction_bbox_xywh: Sequence[float],
    resize_factor: float,
    crop_size: int,
) -> torch.Tensor:
    """把原图 ``xywh`` 转为裁剪图中的归一化 ``xywh``。"""

    box = torch.tensor(validate_xywh(bbox_xywh), dtype=torch.float32)
    extraction = torch.tensor(validate_xywh(extraction_bbox_xywh), dtype=torch.float32)
    extraction_center = extraction[:2] + 0.5 * extraction[2:]
    box_center = box[:2] + 0.5 * box[2:]
    crop_extent = torch.tensor([crop_size, crop_size], dtype=torch.float32)
    output_center = (crop_extent - 1) / 2 + (box_center - extraction_center) * resize_factor
    output_size = box[2:] * resize_factor
    return torch.cat((output_center - 0.5 * output_size, output_size)) / (crop_size - 1)


def hann2d(size: int, device: torch.device) -> torch.Tensor:
    """创建与原 SUTrack centered Hann window 一致的 ``1x1xHxW`` Tensor。"""

    positions = torch.arange(1, size + 1, device=device, dtype=torch.float32)
    window = 0.5 * (1 - torch.cos((2 * math.pi / (size + 1)) * positions))
    return window.view(1, 1, -1, 1) * window.view(1, 1, 1, -1)


def clip_box(
    bbox_xywh: Sequence[float],
    image_height: int,
    image_width: int,
    margin: float = 10.0,
) -> list[float]:
    """将预测框限制到图像内，并维持与原 SUTrack 相同的最小边长。"""

    x1, y1, width, height = (float(value) for value in bbox_xywh)
    x2, y2 = x1 + width, y1 + height
    x1 = min(max(0.0, x1), max(0.0, image_width - margin))
    x2 = min(max(margin, x2), float(image_width))
    y1 = min(max(0.0, y1), max(0.0, image_height - margin))
    y2 = min(max(margin, y2), float(image_height))
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]
