"""不依赖 OpenAI ``clip`` 包的 CLIP 文本编码器。

SUTrack checkpoint 已经包含 CLIP 文本分支权重。这里仅重建与原始 OpenAI CLIP
相同的文本模块命名和前向计算，不实例化从未用于跟踪推理的视觉分支，从而显著
减少加载内存。默认的全零 token 与原 SUTrack 在 ``USE_NLP=False`` 时一致。
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

try:  # timm >= 0.9
    from timm.layers import trunc_normal_
except ImportError:  # SUTrack 官方环境使用 timm 0.5.x
    from timm.models.layers import trunc_normal_


class LayerNorm(nn.LayerNorm):
    """在 fp16/bf16 下用 fp32 计算 LayerNorm。"""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        return super().forward(value.float()).to(dtype=dtype)


class QuickGELU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(1.702 * value)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, attention_mask: torch.Tensor):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                (
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                )
            )
        )
        self.ln_2 = LayerNorm(width)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mask = self.attention_mask.to(device=value.device, dtype=value.dtype)
        normalized = self.ln_1(value)
        attended = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=False,
            attn_mask=mask,
        )[0]
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attention_mask: torch.Tensor):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attention_mask) for _ in range(layers)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.resblocks(value)


class CLIPTextTower(nn.Module):
    """与 ViT-L/14 checkpoint 文本键完全对齐的轻量文本塔。"""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
    ) -> None:
        super().__init__()
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        self.context_length = context_length
        self.transformer = Transformer(width, layers, heads, mask)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=width**-0.5)

    @property
    def dtype(self) -> torch.dtype:
        # OpenAI CLIP 以视觉塔卷积权重的 dtype 作为文本计算 dtype。最小文本塔
        # 不保留视觉分支，因此使用同样由 convert_weights() 转为 fp16 的
        # text_projection 作为等价的 dtype 标记。
        return self.text_projection.dtype

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.context_length:
            raise ValueError(f"CLIP token shape 必须为 [B,{self.context_length}]，收到 {tuple(tokens.shape)}")
        value = self.token_embedding(tokens).to(dtype=self.dtype)
        value = value + self.positional_embedding.to(dtype=self.dtype)
        value = self.transformer(value.permute(1, 0, 2)).permute(1, 0, 2)
        value = self.ln_final(value).to(dtype=self.dtype)
        # OpenAI CLIP 约定 EOT token id 最大；全零序列会稳定选择位置 0。
        indices = tokens.argmax(dim=-1)
        rows = torch.arange(value.shape[0], device=value.device)
        return value[rows, indices] @ self.text_projection


class TextEncoder(nn.Module):
    """CLIP 文本特征到 SUTrack encoder 通道的投影。"""

    def __init__(self, output_channels: int):
        super().__init__()
        self.clip = CLIPTextTower()
        self.norm1 = nn.LayerNorm(768)
        self.text_proj = nn.Linear(768, output_channels)
        self.norm2 = nn.LayerNorm(output_channels)
        trunc_normal_(self.text_proj.weight, std=0.02)
        nn.init.constant_(self.text_proj.bias, 0)

    @property
    def dtype(self) -> torch.dtype:
        return self.text_proj.weight.dtype

    def forward(self, text_data: torch.Tensor) -> torch.Tensor:
        text = self.clip.encode_text(text_data).to(dtype=self.dtype)
        return self.text_proj(text).unsqueeze(1)


def build_textencoder(cfg, encoder) -> TextEncoder:
    del cfg
    return TextEncoder(encoder.num_channels)


def convert_clip_text_weights_to_fp16(module: nn.Module) -> None:
    """复现 OpenAI CLIP ``convert_weights`` 的文本分支 dtype 布局。

    官方 CLIP 只把 Linear、MultiheadAttention 与投影矩阵转为 fp16；Embedding、
    位置编码和 LayerNorm 仍为 fp32。保持这个混合布局是复现官方 SUTrack 数值
    输出所必需的。该函数只应在 CUDA 推理模型加载 checkpoint 前调用。
    """

    def convert(layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()
        if isinstance(layer, nn.MultiheadAttention):
            for name in (
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        projection = getattr(layer, "text_projection", None)
        if projection is not None:
            projection.data = projection.data.half()

    module.apply(convert)
