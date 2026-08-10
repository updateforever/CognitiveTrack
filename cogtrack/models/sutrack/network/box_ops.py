"""SUTrack 推理网络所需的最小 Tensor bbox 运算。"""

from __future__ import annotations

import torch


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """把 ``xyxy`` Tensor 转换为 ``cxcywh``，保持前导维度不变。"""

    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(
        ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0),
        dim=-1,
    )
