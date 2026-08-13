#!/usr/bin/env python3
"""构造 visual-v5 多数据集 SFT probe。

该入口复用旧 Stage-1 已验证的多数据集 split、确定性 presence 采样和 ms-swift 导出
引擎，但固定切换到新协议：过去参考/历史图直接画框、当前图无框、输出三个字段。
默认要求显式 memory label manifest；只有管线 dry-run 才应手工指定
``--memory-supervision feasibility_null``。

先在训练服务器生成可重放 plan：

    python tracking/synthesize_visual_v5_dataset.py \
        --datasets lasot tnl2k mgit --mgit-version tiny \
        --allow-missing-mgit-sequences --env-config configs/env.local.yaml \
        --max-samples-per-sequence 20 --absent-ratio 0.3 \
        --output-dir data/plans/cogtrack_visual_v5_probe --plan-only

完成逐帧 memory 标签后再正式渲染：

    python tracking/synthesize_visual_v5_dataset.py \
        --datasets lasot tnl2k mgit --mgit-version tiny \
        --allow-missing-mgit-sequences --env-config configs/env.local.yaml \
        --sampling-plan data/plans/cogtrack_visual_v5_probe/sampling_plan.json \
        --memory-labels data/labels/cogtrack_visual_v5_memory.jsonl \
        --max-samples-per-sequence 20 --absent-ratio 0.3 \
        --output-dir data/releases/cogtrack_visual_v5_probe
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.synthesize_stage1_dataset import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main(profile="visual_v5"))
