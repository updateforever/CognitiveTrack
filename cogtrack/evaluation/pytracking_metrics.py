"""移植自 lib/test/analysis/extract_results.py 的 pytracking 评测指标。

与原 SUTrack / pytracking 评测工具链完全对齐，产生可直接与 MODEL_ZOO 比较的
Success / Precision / Normalized Precision 三条曲线。

核心约定：
- IoU 使用离散像素几何（``-1`` 偏移），与 pytracking 一致
- 第 0 帧强制替换为 GT（``pred_bb[0, :] = anno_bb[0, :]``）
- ``target_visible`` 用于把不可见帧标为未命中；默认全序列分母下 absent 帧贡献零分
- 稀疏跟踪的 NaN 帧不产生命中；是否进入分母由显式 sparse convention 决定
- 按序列宏平均，不是帧级微平均
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from .io import CanonicalFrame


def _to_torch_float64(values: Sequence[Sequence[float]]) -> torch.Tensor:
    """转换为 pytracking 常用的 float64 张量。"""
    return torch.tensor(values, dtype=torch.float64)


def calc_iou_overlap(pred_bb: torch.Tensor, anno_bb: torch.Tensor) -> torch.Tensor:
    """计算 IoU，使用 pytracking 的离散像素几何（``-1`` 偏移）。"""
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(
        pred_bb[:, :2] + pred_bb[:, 2:] - 1.0,
        anno_bb[:, :2] + anno_bb[:, 2:] - 1.0,
    )
    sz = (br - tl + 1.0).clamp(0)
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection
    return intersection / union


def calc_err_center(
    pred_bb: torch.Tensor,
    anno_bb: torch.Tensor,
    normalized: bool = False,
) -> torch.Tensor:
    """计算欧氏中心误差，可选归一化到 GT 框尺度。"""
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)
    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]
    err_center = ((pred_center - anno_center) ** 2).sum(1).sqrt()
    return err_center


def calc_seq_err_robust(
    pred_bb: torch.Tensor,
    anno_bb: torch.Tensor,
    dataset: str,
    target_visible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按序列计算三条误差曲线与有效帧掩码，完全复现 pytracking 逻辑。

    返回：
        - err_overlap: IoU 张量，无效帧为 -1
        - err_center: 中心误差（像素），无效帧为 inf
        - err_center_normalized: 归一化中心误差，无效帧为 -1
        - valid: 有效帧布尔掩码
    """
    pred_bb = pred_bb.clone()
    sparse_tracking_mask = torch.isnan(pred_bb).any(dim=1)

    # 检查非 NaN 帧中是否存在负值
    if (pred_bb[~sparse_tracking_mask][:, 2:] < 0.0).any():
        raise ValueError("预测框包含负宽高值")

    # 对 UAV 数据集外的 NaN GT 发出警告
    if torch.isnan(anno_bb).any() and dataset != "uav":
        raise ValueError("GT 中包含 NaN（UAV 数据集除外）")

    # 零宽高预测帧前向填充（排除 GT 也无效的帧）
    if (pred_bb[:, 2:] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i - 1, :]

    # 长度对齐
    if pred_bb.shape[0] != anno_bb.shape[0]:
        if dataset == "lasot":
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[: anno_bb.shape[0], :]
                sparse_tracking_mask = sparse_tracking_mask[: anno_bb.shape[0]]
            else:
                raise ValueError("LaSOT 预测长度短于 GT")
        else:
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[: anno_bb.shape[0], :]
                sparse_tracking_mask = sparse_tracking_mask[: anno_bb.shape[0]]
            else:
                pad = torch.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4), dtype=pred_bb.dtype)
                pred_bb = torch.cat((pred_bb, pad), dim=0)
                pad_mask = torch.zeros(
                    anno_bb.shape[0] - sparse_tracking_mask.shape[0],
                    dtype=torch.bool,
                )
                sparse_tracking_mask = torch.cat((sparse_tracking_mask, pad_mask), dim=0)

    # 第 0 帧强制替换为 GT（pytracking 约定）
    pred_bb[0, :] = anno_bb[0, :]

    # 计算有效帧掩码
    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = (
            ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible & (~sparse_tracking_mask)
        )
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & (~sparse_tracking_mask)

    # 稀疏跟踪的 NaN 帧填充 0 避免计算错误
    pred_bb[sparse_tracking_mask] = 0.0

    # 计算三条曲线
    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # 标记无效帧
    if dataset in ["uav"]:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    # 原版用 -1 标记 normalized center 的无效项，但后续命中条件是
    # ``error <= threshold``，因此 NaN 预测会被所有非负阈值误判为命中。稠密
    # tracker 没有该问题；稀疏执行必须把真正缺预测的帧标为 Inf，使 dense-zero
    # 和观测帧执行失败都贡献零分。
    err_center_normalized[sparse_tracking_mask] = float("inf")

    # LaSOT 特殊处理：不可见帧中心误差标记为 inf
    if dataset in {"lasot", "cognitivebench"}:
        err_center_normalized[~target_visible] = float("inf")
        err_center[~target_visible] = float("inf")

    if torch.isnan(err_overlap).any():
        raise ValueError("计算出的 IoU 包含 NaN")

    return err_overlap, err_center, err_center_normalized, valid


# --- 稀疏关键帧计分口径 -------------------------------------------------------
#
# 关键帧稀疏跟踪只在部分帧上真正看图（observation frame），其余帧不产生预测。
# 同一批预测换一个分母约定，数字会差好几倍，所以口径必须显式命名，不能默认。
#
# - ``dense_zero``：分母是全部帧，未观测帧记 0 分（历史行为）。
#   该口径的上限被关键帧率锁死（约 20% 关键帧 ⇒ AUC 上限约 19），且上限随每条
#   序列的关键帧密度浮动，跨序列不可比，仅保留用于向后兼容与稠密跟踪器。
# - ``hold_last``：未观测帧沿用上一次已提交框，分母是全部帧。
#   回答“任意时刻取出跟踪器当前信念，它有多准”。
# - ``observation_only``：只在观测帧上计分，分母是观测帧数。
#   回答“模型真正看图时，它有多准”。
#
# 稠密跟踪器（每帧都有预测）在三个口径下数值完全相同，因此 SUTrack 与
# MODEL_ZOO 的逐值对齐不受影响。
SPARSE_CONVENTION_DENSE_ZERO = "dense_zero"
SPARSE_CONVENTION_HOLD_LAST = "hold_last"
SPARSE_CONVENTION_OBSERVATION_ONLY = "observation_only"

SPARSE_CONVENTIONS: tuple[str, ...] = (
    SPARSE_CONVENTION_DENSE_ZERO,
    SPARSE_CONVENTION_HOLD_LAST,
    SPARSE_CONVENTION_OBSERVATION_ONLY,
)

#: 稀疏跟踪对外报告的两个口径，并列呈现、不分主次。
REPORTED_SPARSE_CONVENTIONS: tuple[str, ...] = (
    SPARSE_CONVENTION_HOLD_LAST,
    SPARSE_CONVENTION_OBSERVATION_ONLY,
)


def _align_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    """把布尔掩码对齐到 ``length``，与 ``calc_seq_err_robust`` 的截断/补零一致。"""
    if mask.shape[0] == length:
        return mask
    if mask.shape[0] > length:
        return mask[:length]
    pad = torch.zeros(length - mask.shape[0], dtype=torch.bool)
    return torch.cat((mask, pad), dim=0)


def _forward_fill_missing(pred_bb: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
    """hold-last：缺预测的帧沿用上一帧已提交框；首个已提交框之前保持 NaN。"""
    filled = pred_bb.clone()
    last: torch.Tensor | None = None
    for i in range(filled.shape[0]):
        if bool(missing[i]):
            if last is not None:
                filled[i, :] = last
        elif not torch.isnan(filled[i, :]).any():
            # 观测帧执行失败或明确输出 absent 时没有 bbox，当前帧仍记为失败；
            # 但不能用 NaN 覆盖此前最后一次合法定位，否则后续非观测帧无法表达
            # tracker 的 hold-last 信念。
            last = filled[i, :].clone()
    return filled


def _prepare_sequence_tensors(
    seq_frames: Sequence[CanonicalFrame],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把一条序列的帧记录转成评测张量。

    返回 ``(anno_bb, pred_bb, missing, non_observation, target_visible)``：

    - ``missing``：该帧没有预测框（跳过的帧与执行失败的帧都算）
    - ``non_observation``：该帧没有执行昂贵 VLM，即
      ``is_observation_frame is False``。Hybrid 即使在该帧有传统 tracker bbox，
      observation-only 仍必须排除它，不能冒充 VLM 已看图。
    """
    anno_bb_list: list[list[float]] = []
    pred_bb_list: list[list[float]] = []
    target_visible_list: list[bool] = []
    missing_list: list[bool] = []
    non_observation_list: list[bool] = []
    has_observation_label = False

    for frame in seq_frames:
        if frame.gt_bbox is None:
            anno_bb_list.append([0.0, 0.0, 0.0, 0.0])
        else:
            anno_bb_list.append(list(frame.gt_bbox))

        if frame.pred_bbox is None:
            pred_bb_list.append([float("nan")] * 4)
            missing_list.append(True)
        else:
            pred_bb_list.append(list(frame.pred_bbox))
            missing_list.append(False)

        if frame.is_observation_frame is not None:
            has_observation_label = True
        non_observation_list.append(frame.is_observation_frame is False)

        if frame.gt_presence == "present":
            target_visible_list.append(True)
        elif frame.gt_presence == "absent":
            target_visible_list.append(False)
        else:
            target_visible_list.append(True)  # 未标注时默认可见

    missing = torch.tensor(missing_list, dtype=torch.bool)
    if has_observation_label:
        non_observation = torch.tensor(non_observation_list, dtype=torch.bool)
    else:
        # 旧结果没有 is_observation_frame 字段：沿用 "NaN 即稀疏跳过" 的原语义。
        non_observation = missing.clone()

    return (
        _to_torch_float64(anno_bb_list),
        _to_torch_float64(pred_bb_list),
        missing,
        non_observation,
        torch.tensor(target_visible_list, dtype=torch.uint8),
    )


def extract_results_from_canonical_frames(
    frames: Iterable[CanonicalFrame],
    *,
    plot_bin_gap: float = 0.05,
    exclude_invalid_frames: bool = False,
    sparse_convention: str = SPARSE_CONVENTION_DENSE_ZERO,
) -> dict[str, Any]:
    """从 CanonicalFrame 列表提取 pytracking 风格的评测结果。

    与 ``lib/test/analysis/extract_results.py`` 语义对齐，输出格式可直接
    用于 ``plot_results.py`` 绘图。

    参数：
        frames: 完整 JSONL 帧列表（可跨序列）
        plot_bin_gap: Success 曲线阈值间隔（默认 0.05）
        exclude_invalid_frames: 是否在分母中排除全部无效帧（含 GT absent 帧）。
            这是原版遗留开关，会同时改变 GT 不可见帧的处理，与稀疏口径无关；
            控制稀疏帧分母请用 ``sparse_convention``。
        sparse_convention: 稀疏关键帧计分口径，取值见 ``SPARSE_CONVENTIONS``。
            默认 ``dense_zero`` 与历史行为一致；稠密跟踪器不受该参数影响。

    返回：
        包含 ``ave_success_rate_plot_overlap`` / ``ave_success_rate_plot_center`` /
        ``ave_success_rate_plot_center_norm`` / ``avg_overlap_all`` 的字典，
        并附带 ``sparse_convention`` 与 ``sparsity`` 统计。
    """
    if sparse_convention not in SPARSE_CONVENTIONS:
        raise ValueError(
            f"未知的 sparse_convention：{sparse_convention!r}，"
            f"可选值：{', '.join(SPARSE_CONVENTIONS)}"
        )

    # 按序列分组
    sequences: dict[str, list[CanonicalFrame]] = {}
    for frame in frames:
        sequences.setdefault(frame.sequence, []).append(frame)

    for seq_frames in sequences.values():
        seq_frames.sort(key=lambda f: f.frame_id)

    # 阈值集合
    threshold_set_overlap = torch.arange(
        0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64
    )
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    num_sequences = len(sequences)
    avg_overlap_all = torch.zeros((num_sequences, 1), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros(
        (num_sequences, 1, threshold_set_overlap.numel()), dtype=torch.float32
    )
    ave_success_rate_plot_center = torch.zeros(
        (num_sequences, 1, threshold_set_center.numel()), dtype=torch.float32
    )
    ave_success_rate_plot_center_norm = torch.zeros(
        (num_sequences, 1, threshold_set_center_norm.numel()), dtype=torch.float32
    )
    valid_sequence = torch.ones(num_sequences, dtype=torch.uint8)
    sequence_names = []
    total_frames = observation_frames = prediction_frames = scored_frames = 0

    for seq_id, (seq_name, seq_frames) in enumerate(sorted(sequences.items())):
        sequence_names.append(seq_name)

        anno_bb, pred_bb, missing, non_observation, target_visible = (
            _prepare_sequence_tensors(seq_frames)
        )
        total_frames += len(seq_frames)
        observation_frames += int((~non_observation).sum().item())
        prediction_frames += int((~missing).sum().item())

        if sparse_convention == SPARSE_CONVENTION_HOLD_LAST:
            # 只填充明确未观测的帧。看过但解析/模型失败的帧必须保留 NaN 并计为失败，
            # 否则工程错误会被错误地伪装成稳定跟踪结果。
            evaluated_pred_bb = _forward_fill_missing(
                pred_bb,
                non_observation & missing,
            )
        else:
            evaluated_pred_bb = pred_bb

        # dataset 名直接取 runner 写进 JSONL 的值，对应原版的 ``seq.dataset``。
        # 这里绝不能从序列名去猜：lasot 的序列叫 ``airplane-1``，猜出来会是
        # "default"，于是原版专门给 lasot 写的那条
        # ``err_center_normalized[~target_visible] = Inf`` 永远不会执行，
        # 不可见帧的 err_center_norm 停留在 -1.0，而 -1.0 <= 任何阈值都成立，
        # 会被当成命中，直接把 Pnorm 抬高。
        dataset = seq_frames[0].dataset

        try:
            err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
                evaluated_pred_bb, anno_bb, dataset, target_visible
            )
        except Exception:
            valid_sequence[seq_id] = 0
            continue

        score_mask = torch.ones(anno_bb.shape[0], dtype=torch.bool)
        if sparse_convention == SPARSE_CONVENTION_OBSERVATION_ONLY:
            # observation-only 只改变稀疏执行的计分分母；观测帧上输出 absent、
            # parse_error 或 model_error 仍保留在分母中记为失败。
            score_mask = ~_align_mask(non_observation, anno_bb.shape[0])

        valid_scored = valid_frame & score_mask
        if bool(valid_scored.any()):
            avg_overlap_all[seq_id, 0] = err_overlap[valid_scored].mean()

        if exclude_invalid_frames:
            seq_length = valid_scored.long().sum()
        else:
            seq_length = score_mask.long().sum()

        if seq_length <= 0:
            valid_sequence[seq_id] = 0
            continue
        scored_frames += int(seq_length.item())

        ave_success_rate_plot_overlap[seq_id, 0, :] = (
            (
                (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1))
                & score_mask.view(-1, 1)
            )
            .sum(0)
            .float()
            / seq_length
        )
        ave_success_rate_plot_center[seq_id, 0, :] = (
            (
                (err_center.view(-1, 1) <= threshold_set_center.view(1, -1))
                & score_mask.view(-1, 1)
            )
            .sum(0)
            .float()
            / seq_length
        )
        ave_success_rate_plot_center_norm[seq_id, 0, :] = (
            (
                (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1))
                & score_mask.view(-1, 1)
            )
            .sum(0)
            .float()
            / seq_length
        )

    num_valid = valid_sequence.long().sum().item()
    print(f"\n计算完成：{num_valid} / {num_sequences} 条序列有效")

    return {
        "sequences": sequence_names,
        "valid_sequence": valid_sequence.tolist(),
        "ave_success_rate_plot_overlap": ave_success_rate_plot_overlap.tolist(),
        "ave_success_rate_plot_center": ave_success_rate_plot_center.tolist(),
        "ave_success_rate_plot_center_norm": ave_success_rate_plot_center_norm.tolist(),
        "avg_overlap_all": avg_overlap_all.tolist(),
        "threshold_set_overlap": threshold_set_overlap.tolist(),
        "threshold_set_center": threshold_set_center.tolist(),
        "threshold_set_center_norm": threshold_set_center_norm.tolist(),
        "num_valid_sequences": num_valid,
        "num_sequences": num_sequences,
        "sparse_convention": sparse_convention,
        "sparsity": {
            "total_frames": total_frames,
            "observation_frames": observation_frames,
            "unobserved_frames": total_frames - observation_frames,
            "observation_rate": observation_frames / total_frames if total_frames else None,
            "prediction_frames": prediction_frames,
            "prediction_rate": prediction_frames / total_frames if total_frames else None,
            "scored_frames": scored_frames,
        },
    }
