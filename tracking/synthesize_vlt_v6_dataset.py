#!/usr/bin/env python3
"""构造 VLT-v6 三图核心跟踪 SFT 数据。

固定输入为：带框初始化目标图、带框历史轨迹 mosaic、当前无框搜索图；动态文本为
初始化目标描述与最近一条已接受状态记忆。首轮默认 ``masked_null``：完整输出仍包含
``memory_update``，但训练脚本只屏蔽该字段值，监督存在性与 bbox。

示例：

    python tracking/synthesize_vlt_v6_dataset.py \
      --datasets lasot tnl2k mgit \
      --mgit-version tiny --allow-missing-mgit-sequences \
      --env-config configs/env.local.yaml \
      --max-samples-per-sequence 20 --absent-ratio 0.3 \
      --output-dir data/releases/cogtrack_vlt_v6_core
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.synthesize_stage1_dataset import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main(profile="vlt_v6"))
