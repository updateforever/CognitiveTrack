#!/usr/bin/env python3
"""在 CognitiveTrack 环境中跑同一 parity 序列，dump 逐帧 bbox。"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cogtrack.models.sutrack.runtime import BuiltinSUTrackRuntime  # noqa: E402

MODEL_CONFIG = os.environ.get(
    "PARITY_MODEL_CONFIG", str(PROJECT_ROOT / "configs/models/sutrack_b384.yaml")
)
CHECKPOINT = os.environ["PARITY_CHECKPOINT"]
FRAMES = Path(os.environ.get("PARITY_FRAMES", "/tmp/parity/frames.json"))
OUT = Path(os.environ.get("PARITY_OUT", "/tmp/parity/out_ours.json"))


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    spec = json.loads(FRAMES.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    np.random.seed(0)

    # torch <= 1.11 默认 matmul.allow_tf32 = True，1.12+ 改为 False。
    # 用这个开关复现旧版 torch 的 TF32 数值行为。
    if os.environ.get("PARITY_TF32") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[info] TF32 matmul 已开启")

    runtime = BuiltinSUTrackRuntime(
        params={"runtime": {"dataset_name": spec["dataset_name"]}},
        model_config=MODEL_CONFIG,
        checkpoint=CHECKPOINT,
        device="cuda",
        amp=False,
        language_mode="auto",
        multi_modal_vision=True,
        multi_modal_language=True,
        use_nlp_datasets=["tnl2k", "otb99", "otb99_lang"],
    )

    records = []
    info = {"init_bbox": list(spec["init_bbox"]), "dataset_name": spec["dataset_name"]}
    out = runtime.initialize(read_rgb(spec["frames"][0]), info)
    records.append({"frame": 0, "bbox": [float(v) for v in out["target_bbox"]], "score": None})

    for index, frame_path in enumerate(spec["frames"][1:], start=1):
        out = runtime.track(read_rgb(frame_path), {})
        records.append(
            {
                "frame": index,
                "bbox": [float(v) for v in out["target_bbox"]],
                "score": float(out["best_score"]),
            }
        )

    payload = {
        "impl": "ours",
        "model_config": MODEL_CONFIG,
        "checkpoint": CHECKPOINT,
        "torch": torch.__version__,
        "multi_modal_vision": runtime.multi_modal_vision,
        "multi_modal_language": runtime.multi_modal_language,
        "update_intervals": runtime.update_interval,
        "update_threshold": runtime.update_threshold,
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"写出 {OUT}: {len(records)} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
