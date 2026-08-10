#!/usr/bin/env python3
"""核对本项目的指标实现与原版 SUTrack analysis 代码是否给出相同数字。

[tools/verify_sutrack_parity.py](verify_sutrack_parity.py) 证明的是"预测一致"，
本脚本证明的是"指标口径一致"：同一份 GT、同一份预测，分别用原版
``lib/test/analysis/extract_results.py`` 的聚合逻辑和本项目的
``cogtrack.evaluation.pytracking_metrics`` 计算 AUC / OP50 / OP75 / P / Pnorm。

原版那个文件不能直接 import：它的 import 链会拖进训练代码，而 torch 2.x 已经
删掉 ``torch._six``，本环境也没装 ``jpeg4py``。所以这里用 AST 抽出需要的三个
纯 torch 函数并原样 exec，保证跑的是原版源码文本而不是手抄版本。

用法::

    python tools/verify_metrics_parity.py \\
        --result-dir outputs/sutrack_adapter/sutrack_b384_builtin_v1/mgit \\
        --dataset mgit
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SUTRACK_ROOT = Path("/data2/wyp/SUTrack")
_WANTED_FUNCTIONS = {"calc_err_center", "calc_iou_overlap", "calc_seq_err_robust"}


def load_original_functions(sutrack_root: Path) -> dict[str, object]:
    """用 AST 从原版 extract_results.py 抽出误差计算函数并原样 exec。"""

    source_path = sutrack_root / "lib/test/analysis/extract_results.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到原版 analysis 源码: {source_path}")

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    picked = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCTIONS
    ]
    missing = _WANTED_FUNCTIONS - {node.name for node in picked}
    if missing:
        raise RuntimeError(f"原版文件里找不到函数: {sorted(missing)}")

    namespace: dict[str, object] = {"torch": torch, "np": np}
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return namespace


def load_table(path: Path, columns: int | None = None) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        separator = "," if "," in line else None
        rows.append([float(v) for v in (line.split(separator) if separator else line.split())])
    array = np.asarray(rows, dtype=np.float64)
    return array if columns is None else array[:, :columns]


def aggregate_with_original(
    names: list[str],
    pred_by_name: dict[str, np.ndarray],
    anno_by_name: dict[str, np.ndarray],
    visible_by_name: dict[str, np.ndarray | None],
    dataset: str,
    sutrack_root: Path,
) -> dict[str, float]:
    """用原版函数复算，聚合方式照抄原版 extract_results 的循环体。"""

    original = load_original_functions(sutrack_root)
    calc = original["calc_seq_err_robust"]

    threshold_overlap = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)
    threshold_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    count = len(names)
    succ = torch.zeros((count, threshold_overlap.numel()), dtype=torch.float32)
    prec = torch.zeros((count, threshold_center.numel()), dtype=torch.float32)
    prec_norm = torch.zeros((count, threshold_center_norm.numel()), dtype=torch.float32)

    for index, name in enumerate(names):
        pred_bb = torch.tensor(pred_by_name[name])
        anno_bb = torch.tensor(anno_by_name[name])
        visible = visible_by_name[name]
        target_visible = None if visible is None else torch.tensor(visible, dtype=torch.uint8)

        err_overlap, err_center, err_center_norm, _valid = calc(
            pred_bb, anno_bb, dataset, target_visible
        )
        seq_length = anno_bb.shape[0]
        succ[index] = (
            err_overlap.view(-1, 1) > threshold_overlap.view(1, -1)
        ).sum(0).float() / seq_length
        prec[index] = (
            err_center.view(-1, 1) <= threshold_center.view(1, -1)
        ).sum(0).float() / seq_length
        prec_norm[index] = (
            err_center_norm.view(-1, 1) <= threshold_center_norm.view(1, -1)
        ).sum(0).float() / seq_length

    succ_avg = succ.mean(dim=0)
    prec_avg = prec.mean(dim=0)
    prec_norm_avg = prec_norm.mean(dim=0)
    return {
        "success_auc": succ_avg.mean().item(),
        "success_op50": succ_avg[10].item(),
        "success_op75": succ_avg[15].item(),
        "precision_p20": prec_avg[20].item(),
        "norm_precision_np20": prec_norm_avg[20].item(),
    }


def aggregate_with_ours(
    names: list[str],
    pred_by_name: dict[str, np.ndarray],
    anno_by_name: dict[str, np.ndarray],
    visible_by_name: dict[str, np.ndarray | None],
    dataset: str,
) -> dict[str, float]:
    """走本项目真实评测路径：CanonicalFrame -> extract -> summarize。"""

    from cogtrack.evaluation.evaluator import summarize_pytracking_curves
    from cogtrack.evaluation.io import CanonicalFrame
    from cogtrack.evaluation.pytracking_metrics import (
        extract_results_from_canonical_frames,
    )

    frames: list[CanonicalFrame] = []
    for name in names:
        pred_bb = pred_by_name[name]
        anno_bb = anno_by_name[name]
        visible = visible_by_name[name]
        for frame_id in range(anno_bb.shape[0]):
            is_visible = True if visible is None else bool(visible[frame_id])
            frames.append(
                CanonicalFrame(
                    sequence=name,
                    dataset=dataset,
                    frame_id=frame_id,
                    gt_bbox=tuple(anno_bb[frame_id].tolist()),
                    pred_bbox=tuple(pred_bb[frame_id].tolist())
                    if frame_id < pred_bb.shape[0]
                    else None,
                    gt_presence="present" if is_visible else "absent",
                    pred_presence="present",
                    gt_identity=None,
                    pred_identity=None,
                    execution_status="ok",
                    is_observation_frame=True,
                )
            )

    summary = summarize_pytracking_curves(extract_results_from_canonical_frames(frames))
    return {key: summary[key] for key in (
        "success_auc",
        "success_op50",
        "success_op75",
        "precision_p20",
        "norm_precision_np20",
    )}


# GT 与 absent 标注的目录布局按数据集不同，这里只登记已核对过的几个。
_LAYOUTS = {
    "mgit": {
        "root": Path("/data2/DATASETS_PUBLIC/MGIT"),
        "groundtruth": "attribute/groundtruth/{name}.txt",
        "absent": "attribute/absent/{name}.txt",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="mgit", choices=sorted(_LAYOUTS))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--sutrack-root", type=Path, default=DEFAULT_SUTRACK_ROOT)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    layout = _LAYOUTS[args.dataset]
    dataset_root = args.dataset_root or layout["root"]

    pred_files = [
        path
        for path in sorted(args.result_dir.glob("*.txt"))
        if not path.name.endswith("_time.txt")
    ]
    if not pred_files:
        print(f"{args.result_dir} 下没有预测结果 .txt", file=sys.stderr)
        return 2

    names = [path.stem for path in pred_files]
    pred_by_name: dict[str, np.ndarray] = {}
    anno_by_name: dict[str, np.ndarray] = {}
    visible_by_name: dict[str, np.ndarray | None] = {}

    for name, pred_path in zip(names, pred_files, strict=True):
        pred_by_name[name] = load_table(pred_path, columns=4)
        anno_by_name[name] = load_table(
            dataset_root / layout["groundtruth"].format(name=name), columns=4
        )
        absent_path = dataset_root / layout["absent"].format(name=name)
        visible_by_name[name] = (
            (load_table(absent_path).reshape(-1) == 0) if absent_path.is_file() else None
        )

    print(f"数据集={args.dataset}  序列数={len(names)}")
    print(f"序列: {' '.join(names)}")

    original = aggregate_with_original(
        names, pred_by_name, anno_by_name, visible_by_name, args.dataset, args.sutrack_root
    )
    ours = aggregate_with_ours(
        names, pred_by_name, anno_by_name, visible_by_name, args.dataset
    )

    print(f"\n{'metric':22s}{'原版':>16s}{'本项目':>16s}{'差':>12s}")
    worst = 0.0
    for key, value in original.items():
        delta = abs(ours[key] - value)
        worst = max(worst, delta)
        print(f"{key:22s}{value:16.10f}{ours[key]:16.10f}{delta:12.2e}")

    print(f"\n最大绝对差 = {worst:.2e}（阈值 {args.tolerance:.0e}）")
    if worst <= args.tolerance:
        print("结论：指标口径与原版一致。")
        return 0
    print("结论：存在超出容差的偏差，需要排查。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
