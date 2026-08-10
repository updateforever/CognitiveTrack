"""SUTrack Fast-iTPN 编码器的独立推理构造器。"""

from __future__ import annotations

from torch import nn

from . import fastitpn


class Encoder(nn.Module):
    """只保留公开 checkpoint 所需的 Fast-iTPN 推理路径。"""

    def __init__(self, cfg):
        super().__init__()
        name = str(cfg.MODEL.ENCODER.TYPE)
        if not name.lower().startswith("fastitpn"):
            raise ValueError(f"CognitiveTrack 当前只支持 Fast-iTPN SUTrack encoder，收到 {name!r}")
        factory = getattr(fastitpn, name, None)
        if not callable(factory):
            raise ValueError(f"未知 Fast-iTPN encoder: {name!r}")

        # 完整 SUTrack checkpoint 已经包含 encoder 权重，因此禁止构造阶段再次
        # 加载预训练文件，避免隐式网络下载和机器路径依赖。
        self.body = factory(
            pretrained=False,
            search_size=int(cfg.DATA.SEARCH.SIZE),
            template_size=int(cfg.DATA.TEMPLATE.SIZE),
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            init_values=0.1,
            drop_block_rate=None,
            use_mean_pooling=True,
            grad_ckpt=False,
            cls_token=bool(cfg.MODEL.ENCODER.CLASS_TOKEN),
            pos_type=str(cfg.MODEL.ENCODER.POS_TYPE),
            token_type_indicate=bool(cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE),
            pretrain_type=str(cfg.MODEL.ENCODER.PRETRAIN_TYPE),
            patchembed_init=str(cfg.MODEL.ENCODER.PATCHEMBED_INIT),
        )
        if "itpnl" in name:
            self.num_channels = 768
        elif "itpnt" in name or "itpns" in name:
            self.num_channels = 384
        else:
            self.num_channels = 512

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        return self.body(template_list, search_list, template_anno_list, text_src, task_index)


def build_encoder(cfg) -> Encoder:
    return Encoder(cfg)
