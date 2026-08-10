#!/usr/bin/env python3
"""逐帧核对 CognitiveTrack 的内置 SUTrack 与原版 SUTrack 仓库是否数值一致。

CognitiveTrack 把 SUTrack 的推理路径重新组织进了 ``cogtrack.models.sutrack``，
因此需要一个可复现的证据链证明"重构没有引入误差"。本脚本在同一份帧清单、
同一份 checkpoint、同一个 torch 上分别跑两套实现，然后逐帧比较 bbox。

用法（三步，或用 ``all`` 一次跑完）::

    export COGTRACK_SUTRACK_CHECKPOINT=/path/to/SUTRACK_ep0180.pth.tar
    python tools/verify_sutrack_parity.py all --dataset videocube --frames 400
    python tools/verify_sutrack_parity.py all --dataset tnl2k --frames 200

``--dataset videocube`` 覆盖 ``use_nlp=False`` 分支（全零 token 仍过文本塔），
``--dataset tnl2k`` 覆盖 ``use_nlp=True`` 分支（对 caption 做 CLIP tokenize）。
两条分支都应当得到 0.000000 px 的最大坐标差。

注意：必须两边用同一个 torch。torch 1.11 与 2.9 的 cuBLAS kernel 不同，会带来
约 1e-2 px 的浮点差异并在递归跟踪中累积，那不是移植错误。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUTRACK_ROOT = Path("/data2/wyp/SUTrack")
DEFAULT_WORKDIR = Path("/tmp/cogtrack_parity")

# ---------------------------------------------------------------- 帧清单生成


def make_spec_videocube(frames: int, workdir: Path, root: Path) -> Path:
    """videocube/MGIT 序列 029，use_nlp=False。"""

    seq_dir = root / "data" / "val" / "029" / "frame_029"
    gt_path = root / "attribute" / "groundtruth" / "029.txt"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"缺少序列目录: {seq_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 groundtruth: {gt_path}")

    images = sorted(seq_dir.glob("*.jpg"))[:frames]
    if not images:
        raise FileNotFoundError(f"序列目录没有 jpg: {seq_dir}")
    first = gt_path.read_text(encoding="utf-8").splitlines()[0].strip()
    return _write_spec(
        workdir,
        sequence="029",
        dataset_name="videocube",
        init_bbox=_parse_bbox(first),
        init_nlp=None,
        images=images,
    )


def make_spec_tnl2k(frames: int, workdir: Path, root: Path) -> Path:
    """TNL2K 首个测试序列，use_nlp=True，带真实 caption。"""

    subset = root / "TNL2K_test_subset"
    if not subset.is_dir():
        raise FileNotFoundError(f"缺少 TNL2K 测试子集: {subset}")
    sequences = sorted(p for p in subset.iterdir() if p.is_dir())
    if not sequences:
        raise FileNotFoundError(f"TNL2K 测试子集为空: {subset}")
    seq = sequences[0]

    images = sorted(list((seq / "imgs").glob("*.jpg")) + list((seq / "imgs").glob("*.png")))
    images = images[:frames]
    if not images:
        raise FileNotFoundError(f"序列没有图像: {seq / 'imgs'}")
    first = (seq / "groundtruth.txt").read_text(encoding="utf-8").splitlines()[0].strip()
    caption = (seq / "language.txt").read_text(encoding="utf-8").strip()
    return _write_spec(
        workdir,
        sequence=seq.name,
        dataset_name="tnl2k",
        init_bbox=_parse_bbox(first),
        init_nlp=caption,
        images=images,
    )


def _parse_bbox(line: str) -> list[float]:
    separator = "," if "," in line else None
    values = [float(v) for v in (line.split(separator) if separator else line.split())]
    if len(values) != 4:
        raise ValueError(f"初始框解析失败: {line!r}")
    return values


def _write_spec(
    workdir: Path,
    *,
    sequence: str,
    dataset_name: str,
    init_bbox: list[float],
    init_nlp: str | None,
    images: list[Path],
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    spec_path = workdir / "frames.json"
    payload = {
        "sequence": sequence,
        "dataset_name": dataset_name,
        "init_bbox": init_bbox,
        "init_nlp": init_nlp,
        "frames": [str(p) for p in images],
    }
    spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[spec] {spec_path}: {dataset_name}/{sequence}, {len(images)} 帧")
    if init_nlp is not None:
        print(f"       init_nlp={init_nlp!r}")
    return spec_path


# ------------------------------------------------------------------ 两侧 runner
# 两套实现放在独立子进程里跑，避免原版的 ``lib`` 包与本项目的模块搜索路径互相
# 干扰，也方便将来在不同 conda 环境之间对比。

_RUN_OURS = '''
import json, os, sys
sys.path.insert(0, {project!r})
import cv2, numpy as np, torch
from cogtrack.models.sutrack.runtime import BuiltinSUTrackRuntime

spec = json.loads(open({spec!r}, encoding="utf-8").read())
torch.manual_seed(0); np.random.seed(0)

def rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

runtime = BuiltinSUTrackRuntime(
    params={{"runtime": {{"dataset_name": spec["dataset_name"]}}}},
    model_config={model_config!r},
    checkpoint={checkpoint!r},
    device="cuda",
    amp=False,
    language_mode="auto",
    multi_modal_vision=True,
    multi_modal_language=True,
    use_nlp_datasets=["tnl2k", "otb99", "otb99_lang"],
)

info = {{"init_bbox": list(spec["init_bbox"]), "dataset_name": spec["dataset_name"]}}
if spec.get("init_nlp") is not None:
    info["init_nlp"] = spec["init_nlp"]
out = runtime.initialize(rgb(spec["frames"][0]), info)
records = [{{"frame": 0, "bbox": [float(v) for v in out["target_bbox"]]}}]
for index, path in enumerate(spec["frames"][1:], start=1):
    out = runtime.track(rgb(path), {{}})
    records.append({{"frame": index, "bbox": [float(v) for v in out["target_bbox"]]}})

json.dump(
    {{
        "impl": "ours",
        "torch": torch.__version__,
        "multi_modal_vision": runtime.multi_modal_vision,
        "multi_modal_language": runtime.multi_modal_language,
        "update_intervals": runtime.update_interval,
        "update_threshold": runtime.update_threshold,
        "records": records,
    }},
    open({out!r}, "w", encoding="utf-8"),
    indent=2,
)
print("[ours] 写出 " + {out!r} + f": {{len(records)}} 帧")
'''

_RUN_ORIGINAL = '''
import json, os, sys
sys.path.insert(0, {sutrack_root!r})
import cv2, numpy as np, torch

import lib.models.sutrack.encoder as encoder_module
# 原版 yaml 的 PRETRAIN_TYPE 指向作者机器上的 iTPN 预训练权重（本地通常不存在），
# 且 encoder 用 pretrained=is_main_process() 触发加载。紧随其后的
# load_state_dict(strict=True) 会用 SUTrack checkpoint 覆盖全部权重，
# load_pretrained 也不改变任何张量形状，因此跳过它对最终数值没有影响。
encoder_module.is_main_process = lambda: False

from lib.config.sutrack.config import cfg, update_config_from_file
from lib.test.tracker.sutrack import SUTRACK
from lib.test.utils import TrackerParams

spec = json.loads(open({spec!r}, encoding="utf-8").read())
torch.manual_seed(0); np.random.seed(0)

params = TrackerParams()
update_config_from_file(os.path.join({sutrack_root!r}, "experiments", "sutrack", {yaml_name!r} + ".yaml"))
params.cfg = cfg
params.yaml_name = {yaml_name!r}
params.template_factor = cfg.TEST.TEMPLATE_FACTOR
params.template_size = cfg.TEST.TEMPLATE_SIZE
params.search_factor = cfg.TEST.SEARCH_FACTOR
params.search_size = cfg.TEST.SEARCH_SIZE
params.checkpoint = {checkpoint!r}
params.save_all_boxes = False
params.debug = 0

def rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tracker = SUTRACK(params, spec["dataset_name"])
init_info = {{"init_bbox": list(spec["init_bbox"])}}
if spec.get("init_nlp") is not None:
    init_info["init_nlp"] = spec["init_nlp"]
out = tracker.initialize(rgb(spec["frames"][0]), init_info)
records = [{{
    "frame": 0,
    "bbox": [float(v) for v in (out or {{}}).get("target_bbox", spec["init_bbox"])],
}}]
for index, path in enumerate(spec["frames"][1:], start=1):
    out = tracker.track(rgb(path))
    records.append({{"frame": index, "bbox": [float(v) for v in out["target_bbox"]]}})

json.dump(
    {{
        "impl": "original",
        "torch": torch.__version__,
        "multi_modal_vision": bool(tracker.multi_modal_vision),
        "multi_modal_language": bool(tracker.multi_modal_language),
        "use_nlp": bool(tracker.use_nlp),
        "update_intervals": tracker.update_intervals,
        "update_threshold": tracker.update_threshold,
        "records": records,
    }},
    open({out!r}, "w", encoding="utf-8"),
    indent=2,
)
print("[original] 写出 " + {out!r} + f": {{len(records)}} 帧")
'''


def _run_python(source: str, *, label: str) -> None:
    process = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"{label} 侧运行失败，退出码 {process.returncode}")


def run_ours(spec: Path, out: Path, *, model_config: Path, checkpoint: str) -> None:
    _run_python(
        _RUN_OURS.format(
            project=str(PROJECT_ROOT),
            spec=str(spec),
            out=str(out),
            model_config=str(model_config),
            checkpoint=checkpoint,
        ),
        label="ours",
    )


def run_original(
    spec: Path, out: Path, *, sutrack_root: Path, yaml_name: str, checkpoint: str
) -> None:
    _run_python(
        _RUN_ORIGINAL.format(
            sutrack_root=str(sutrack_root),
            spec=str(spec),
            out=str(out),
            yaml_name=yaml_name,
            checkpoint=checkpoint,
        ),
        label="original",
    )


# ---------------------------------------------------------------------- 比较


def _iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compare(original_path: Path, ours_path: Path, *, tolerance: float = 1e-3) -> int:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    ours = json.loads(ours_path.read_text(encoding="utf-8"))

    print("\n=== 开关对齐 ===")
    for key in ("multi_modal_vision", "multi_modal_language", "update_intervals", "update_threshold"):
        left, right = original.get(key), ours.get(key)
        print(f"  {'OK ' if left == right else '!!!'} {key}: original={left} ours={right}")
    print(f"  --  original use_nlp={original.get('use_nlp')}")
    print(f"  --  torch: original={original.get('torch')} ours={ours.get('torch')}")
    if original.get("torch") != ours.get("torch"):
        print("  !!! torch 版本不同：浮点差异会累积，本次结果不能用来判断移植正确性")

    left_records, right_records = original["records"], ours["records"]
    if len(left_records) != len(right_records):
        print(f"!!! 帧数不同: {len(left_records)} vs {len(right_records)}")
        return 1

    max_delta = 0.0
    worst_frame = -1
    min_iou = 1.0
    for left, right in zip(left_records, right_records, strict=True):
        delta = max(abs(a - b) for a, b in zip(left["bbox"], right["bbox"], strict=True))
        if delta > max_delta:
            max_delta, worst_frame = delta, left["frame"]
        min_iou = min(min_iou, _iou(left["bbox"], right["bbox"]))

    print(f"\n=== 逐帧对比（{len(left_records)} 帧）===")
    print(f"  最大坐标绝对差: {max_delta:.6f} px (frame {worst_frame})")
    print(f"  最小 IoU:       {min_iou:.6f}")

    if max_delta < tolerance:
        print("\n结论: 数值一致，移植没有引入误差")
        return 0
    if min_iou > 0.99:
        print("\n结论: 存在浮点级差异但轨迹一致；先确认两边 torch 版本相同")
        return 1
    print("\n结论: 轨迹分叉，存在移植错误")
    return 1


# ------------------------------------------------------------------------ CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "stage",
        choices=("spec", "original", "ours", "compare", "all"),
        help="要执行的阶段；all 表示依次跑完 spec/original/ours/compare",
    )
    parser.add_argument(
        "--dataset",
        choices=("videocube", "tnl2k"),
        default="videocube",
        help="videocube 覆盖 use_nlp=False 分支，tnl2k 覆盖 use_nlp=True 分支",
    )
    parser.add_argument("--frames", type=int, default=400, help="参与对比的帧数")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("COGTRACK_SUTRACK_CHECKPOINT"),
        help="SUTrack checkpoint；默认取环境变量 COGTRACK_SUTRACK_CHECKPOINT",
    )
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs/models/sutrack_b384.yaml"),
        help="本项目的模型结构配置",
    )
    parser.add_argument("--yaml-name", default="sutrack_b384", help="原版 experiments/sutrack 下的 yaml 名")
    parser.add_argument("--sutrack-root", default=str(DEFAULT_SUTRACK_ROOT), help="原版 SUTrack 仓库根目录")
    parser.add_argument("--videocube-root", default="/data2/DATASETS_PUBLIC/MGIT", help="MGIT/videocube 根目录")
    parser.add_argument("--tnl2k-root", default="/data2/DATASETS_PUBLIC/TNL2K", help="TNL2K 根目录")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="中间产物目录")
    return parser


def main() -> int:
    args = _parser().parse_args()
    workdir = Path(args.workdir)
    spec_path = workdir / "frames.json"
    original_out = workdir / f"{args.dataset}_original.json"
    ours_out = workdir / f"{args.dataset}_ours.json"

    needs_checkpoint = args.stage in {"original", "ours", "all"}
    if needs_checkpoint and not args.checkpoint:
        print(
            "[错误] 需要 SUTrack checkpoint：设置 COGTRACK_SUTRACK_CHECKPOINT 或传 --checkpoint",
            file=sys.stderr,
        )
        return 2

    try:
        if args.stage in {"spec", "all"}:
            if args.dataset == "videocube":
                make_spec_videocube(args.frames, workdir, Path(args.videocube_root))
            else:
                make_spec_tnl2k(args.frames, workdir, Path(args.tnl2k_root))

        if args.stage in {"original", "all"}:
            run_original(
                spec_path,
                original_out,
                sutrack_root=Path(args.sutrack_root),
                yaml_name=args.yaml_name,
                checkpoint=args.checkpoint,
            )

        if args.stage in {"ours", "all"}:
            run_ours(
                spec_path,
                ours_out,
                model_config=Path(args.model_config),
                checkpoint=args.checkpoint,
            )

        if args.stage in {"compare", "all"}:
            return compare(original_out, ours_out)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
