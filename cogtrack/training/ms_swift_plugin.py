"""ms-swift 外部插件：仅屏蔽未知 ``memory_update`` 值的 SFT loss。

训练入口通过 ``--external_plugins`` 加载本文件，再使用
``--loss_scale cogtrack_tracking_sft``。不要直接修改 site-packages 中的 ms-swift。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 外部插件可能在数据集目录作为 cwd 被导入。即使项目尚未 editable install，也要能
# 从当前文件稳定定位仓库根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swift.loss_scale import LossScale, loss_scale_map  # noqa: E402

from cogtrack.training.loss_mask import split_tracking_sft_response  # noqa: E402

LOSS_SCALE_NAME = "cogtrack_tracking_sft"
LEGACY_LOSS_SCALE_NAME = "cogtrack_tracking_core"


class CognitiveTrackingSFTLossScale(LossScale):
    """保留跟踪与 JSON 结构 loss，只屏蔽末尾未知状态更新值。"""

    is_binary = True

    def get_loss_scale(self, context: str, **kwargs):  # type: ignore[override]
        del kwargs
        # ms-swift 会分别把 assistant 正文和 chat-template suffix（例如结束 token）
        # 交给本函数。严格的三字段检查已在训练 preflight 完成；suffix 保持权重 1。
        return split_tracking_sft_response(context, require_memory_field=False)


for name in (LOSS_SCALE_NAME, LEGACY_LOSS_SCALE_NAME):
    if name in loss_scale_map and loss_scale_map[name] is not CognitiveTrackingSFTLossScale:
        raise RuntimeError(f"ms-swift loss scale 名称冲突：{name}")
    loss_scale_map[name] = CognitiveTrackingSFTLossScale

# 历史 import 兼容；新日志和新命令只使用 CognitiveTrackingSFTLossScale。
CognitiveTrackingCoreLossScale = CognitiveTrackingSFTLossScale


__all__ = [
    "CognitiveTrackingCoreLossScale",
    "CognitiveTrackingSFTLossScale",
    "LEGACY_LOSS_SCALE_NAME",
    "LOSS_SCALE_NAME",
]
