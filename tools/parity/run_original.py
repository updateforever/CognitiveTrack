#!/usr/bin/env python3
"""在原版 SUTrack 仓库环境中跑 parity 序列，dump 逐帧 bbox。

只复用原版的 tracker 类和 config 加载，不经过原版数据集注册。
"""

# 原版模块必须在其仓库路径插入 sys.path 之后导入，刻意保留该导入顺序。
# ruff: noqa: I001

import json
import os
import sys
from pathlib import Path

SUTRACK_ROOT = Path(os.environ.get("PARITY_SUTRACK_ROOT", "/data2/wyp/SUTrack"))
sys.path.insert(0, str(SUTRACK_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lib.models.sutrack.encoder as encoder_module  # noqa: E402

# 原版 yaml 的 PRETRAIN_TYPE 指向作者机器上的 iTPN 预训练权重（本地不存在），
# 且 encoder 用 pretrained=is_main_process() 触发加载。紧随其后的
# load_state_dict(strict=True) 会用 SUTrack checkpoint 覆盖全部权重，
# load_pretrained 也不改变任何张量形状，因此跳过它对最终数值没有影响。
encoder_module.is_main_process = lambda: False

from lib.config.sutrack.config import cfg, update_config_from_file  # noqa: E402
from lib.test.tracker.sutrack import SUTRACK  # noqa: E402
from lib.test.utils import TrackerParams  # noqa: E402

YAML_NAME = os.environ.get("PARITY_YAML", "sutrack_b384")
CHECKPOINT = os.environ["PARITY_CHECKPOINT"]
FRAMES = Path(os.environ.get("PARITY_FRAMES", "/tmp/parity/frames.json"))
OUT = Path(os.environ.get("PARITY_OUT", "/tmp/parity/out_original.json"))


def build_params() -> TrackerParams:
    params = TrackerParams()
    update_config_from_file(str(SUTRACK_ROOT / "experiments" / "sutrack" / f"{YAML_NAME}.yaml"))
    params.cfg = cfg
    params.yaml_name = YAML_NAME
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = CHECKPOINT
    params.save_all_boxes = False
    params.debug = 0
    return params


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    spec = json.loads(FRAMES.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    np.random.seed(0)

    tracker = SUTRACK(build_params(), spec["dataset_name"])
    records = []

    first = read_rgb(spec["frames"][0])
    out = tracker.initialize(first, {"init_bbox": list(spec["init_bbox"])})
    records.append(
        {
            "frame": 0,
            "bbox": [float(v) for v in (out or {}).get("target_bbox", spec["init_bbox"])],
            "score": None,
        }
    )

    for index, frame_path in enumerate(spec["frames"][1:], start=1):
        out = tracker.track(read_rgb(frame_path))
        score = out.get("best_score")
        records.append(
            {
                "frame": index,
                "bbox": [float(v) for v in out["target_bbox"]],
                "score": None if score is None else float(score),
            }
        )

    payload = {
        "impl": "original",
        "yaml": YAML_NAME,
        "checkpoint": CHECKPOINT,
        "torch": torch.__version__,
        "multi_modal_vision": bool(tracker.multi_modal_vision),
        "multi_modal_language": bool(tracker.multi_modal_language),
        "use_nlp": bool(tracker.use_nlp),
        "update_intervals": tracker.update_intervals,
        "update_threshold": tracker.update_threshold,
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"写出 {OUT}: {len(records)} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
