#!/usr/bin/env python3
"""构造 VLT-v6.4 三图 tracking/state-update SFT 数据。

固定输入为：带框初始化目标图、带框历史轨迹 mosaic、当前无框搜索图；动态文本为
初始化目标描述与最近一条已接受状态记忆。tracking SFT 默认 ``masked_null``；
state-update SFT 使用逐帧显式标签。

示例：

    python tracking/synthesize_vlt_v6_dataset.py \
      --datasets lasot tnl2k mgit \
      --mgit-version tiny --allow-missing-mgit-sequences \
      --env-config configs/env.local.yaml \
      --max-samples-per-sequence 20 --absent-ratio 0.3 \
      --output-dir data/releases/cogtrack_vlt_v640_tracking_sft
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
