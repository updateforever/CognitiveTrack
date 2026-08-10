"""跟踪、存在性、身份判别与长时恢复指标。

所有函数均接收 :class:`~cogtrack.evaluation.io.CanonicalFrame` 列表并返回
普通字典，既便于单元测试，也便于后续写入 JSON/CSV。指标计算没有模型、
数据集加载器或协议类依赖。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

from .io import CanonicalFrame, is_execution_error

# 与 pytracking 常用绘图口径一致：Success 使用 0:0.05:1，Precision 使用
# 0:1:50 像素，Normalized Precision 使用 0:0.01:0.5。
DEFAULT_SUCCESS_THRESHOLDS: tuple[float, ...] = tuple(index / 20.0 for index in range(21))
DEFAULT_PRECISION_THRESHOLDS: tuple[float, ...] = tuple(float(index) for index in range(51))
DEFAULT_NORMALIZED_PRECISION_THRESHOLDS: tuple[float, ...] = tuple(index / 100.0 for index in range(51))


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    """安全除法；分母为零时返回 ``None`` 而不是 NaN。"""

    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def bbox_iou(
    first: Optional[Sequence[float]],
    second: Optional[Sequence[float]],
) -> float:
    """计算两个像素 ``xywh`` 边界框的 IoU。

    无效框返回 0，评测循环因此无需为 absent/error 帧做特殊数值填充。
    坐标采用连续几何定义，不额外添加 ``+1``。
    """

    if first is None or second is None or len(first) < 4 or len(second) < 4:
        return 0.0
    try:
        ax, ay, aw, ah = (float(first[i]) for i in range(4))
        bx, by, bw, bh = (float(second[i]) for i in range(4))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(v) for v in (ax, ay, aw, ah, bx, by, bw, bh)):
        return 0.0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def center_errors(
    prediction: Optional[Sequence[float]],
    ground_truth: Sequence[float],
) -> tuple[float, float]:
    """返回像素中心误差与 pytracking 风格归一化中心误差。"""

    if prediction is None or len(prediction) < 4 or len(ground_truth) < 4:
        return math.inf, math.inf
    try:
        px, py, pw, ph = (float(prediction[i]) for i in range(4))
        gx, gy, gw, gh = (float(ground_truth[i]) for i in range(4))
    except (TypeError, ValueError, OverflowError):
        return math.inf, math.inf
    if (
        not all(math.isfinite(v) for v in (px, py, pw, ph, gx, gy, gw, gh))
        or pw <= 0
        or ph <= 0
        or gw <= 0
        or gh <= 0
    ):
        return math.inf, math.inf

    # 与 pytracking 一致，偶数尺寸框的中心使用 (size - 1) / 2。
    pred_cx = px + 0.5 * (pw - 1.0)
    pred_cy = py + 0.5 * (ph - 1.0)
    gt_cx = gx + 0.5 * (gw - 1.0)
    gt_cy = gy + 0.5 * (gh - 1.0)
    dx, dy = pred_cx - gt_cx, pred_cy - gt_cy
    pixel_error = math.hypot(dx, dy)
    normalized_error = math.hypot(dx / gw, dy / gh)
    return pixel_error, normalized_error


def _curve_value_at(
    thresholds: Sequence[float],
    curve: Optional[Sequence[float]],
    target: float,
) -> Optional[float]:
    """读取曲线指定阈值处的值；自定义阈值不含目标点时返回 ``None``。"""

    if curve is None:
        return None
    for index, threshold in enumerate(thresholds):
        if math.isclose(float(threshold), target, rel_tol=0.0, abs_tol=1e-12):
            return float(curve[index])
    return None


class _LocalizationAccumulator:
    """流式累计定位曲线，避免百万帧评测保留三份逐帧浮点数组。

    阈值有序，因此每帧只需二分定位一次命中区间，并更新差分数组；最终再
    做一次前缀和即可得到整条曲线。这样每帧是 ``O(log T)``，而不是遍历
    123 个默认阈值。
    """

    def __init__(
        self,
        success_thresholds: Sequence[float],
        precision_thresholds: Sequence[float],
        normalized_precision_thresholds: Sequence[float],
    ) -> None:
        if not success_thresholds or not precision_thresholds or not normalized_precision_thresholds:
            raise ValueError("评测阈值不能为空")
        self.success_thresholds = tuple(float(value) for value in success_thresholds)
        self.precision_thresholds = tuple(float(value) for value in precision_thresholds)
        self.normalized_precision_thresholds = tuple(
            float(value) for value in normalized_precision_thresholds
        )
        for name, values in (
            ("success", self.success_thresholds),
            ("precision", self.precision_thresholds),
            ("normalized_precision", self.normalized_precision_thresholds),
        ):
            if any(current <= previous for previous, current in zip(values, values[1:], strict=False)):
                raise ValueError(f"{name} 阈值必须严格递增")
        self.evaluated = 0
        self.success_difference = [0] * (len(self.success_thresholds) + 1)
        self.precision_difference = [0] * (len(self.precision_thresholds) + 1)
        self.normalized_precision_difference = [0] * (
            len(self.normalized_precision_thresholds) + 1
        )

    def add(self, overlap: float, pixel_error: float, normalized_error: float) -> None:
        """加入一帧；失败帧传 ``overlap=-1``、中心误差 ``Inf``。"""

        self.evaluated += 1
        # Success 的条件是 overlap > threshold，命中曲线前缀 [0, end)。
        success_end = bisect_left(self.success_thresholds, overlap)
        if success_end:
            self.success_difference[0] += 1
            self.success_difference[success_end] -= 1

        # Precision 的条件是 error <= threshold，命中曲线后缀 [start, T)。
        precision_start = bisect_left(self.precision_thresholds, pixel_error)
        if precision_start < len(self.precision_thresholds):
            self.precision_difference[precision_start] += 1
            self.precision_difference[-1] -= 1
        normalized_start = bisect_left(
            self.normalized_precision_thresholds,
            normalized_error,
        )
        if normalized_start < len(self.normalized_precision_thresholds):
            self.normalized_precision_difference[normalized_start] += 1
            self.normalized_precision_difference[-1] -= 1

    @staticmethod
    def _restore_counts(difference: Sequence[int]) -> list[int]:
        """把长度为 ``T + 1`` 的差分数组还原为长度 ``T`` 的计数。"""

        running = 0
        counts: list[int] = []
        for delta in difference[:-1]:
            running += delta
            counts.append(running)
        return counts

    def result(self) -> dict[str, Any]:
        """生成三条曲线及其常用标量读数。"""

        if self.evaluated == 0:
            success_curve = precision_curve = normalized_precision_curve = None
        else:
            success_curve = [
                count / self.evaluated
                for count in self._restore_counts(self.success_difference)
            ]
            precision_curve = [
                count / self.evaluated
                for count in self._restore_counts(self.precision_difference)
            ]
            normalized_precision_curve = [
                count / self.evaluated
                for count in self._restore_counts(self.normalized_precision_difference)
            ]
        return {
            "evaluated_frames": self.evaluated,
            "thresholds": {
                "success": list(self.success_thresholds),
                "precision_pixels": list(self.precision_thresholds),
                "normalized_precision": list(self.normalized_precision_thresholds),
            },
            "success_curve": success_curve,
            "precision_curve": precision_curve,
            "normalized_precision_curve": normalized_precision_curve,
            "success_auc": None if success_curve is None else sum(success_curve) / len(success_curve),
            "success_op50": _curve_value_at(self.success_thresholds, success_curve, 0.5),
            "success_op75": _curve_value_at(self.success_thresholds, success_curve, 0.75),
            "precision_at_20": _curve_value_at(
                self.precision_thresholds,
                precision_curve,
                20.0,
            ),
            "normalized_precision_at_0_2": _curve_value_at(
                self.normalized_precision_thresholds,
                normalized_precision_curve,
                0.2,
            ),
        }


def _benchmark_prediction(frame: CanonicalFrame) -> Optional[Sequence[float]]:
    """返回传统 TXT/pytracking 口径的已发布框。

    benchmark 只读顶层 ``target_bbox``。hybrid 可在身份尚未确认时
    按显式配置发布 SUTrack 框，此时 presence 可为 uncertain，但传统
    定位曲线仍必须与实际 TXT 一致。
    """

    if frame.pred_bbox is not None and not is_execution_error(frame.execution_status):
        return frame.pred_bbox
    return None


def _committed_prediction(frame: CanonicalFrame) -> Optional[Sequence[float]]:
    """返回通过身份/存在门控的认知定位框。"""

    if (
        frame.pred_presence == "present"
        and frame.pred_bbox is not None
        and not is_execution_error(frame.execution_status)
    ):
        return frame.pred_bbox
    return None


def evaluate_benchmark_sequence(
    frames: Iterable[CanonicalFrame],
    *,
    success_thresholds: Optional[Sequence[float]] = None,
    precision_thresholds: Optional[Sequence[float]] = None,
    normalized_precision_thresholds: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """按一个序列计算 pytracking 风格的三条 benchmark 曲线。

    分母是该序列内所有带 ``present/absent`` GT 的帧。GT absent 帧在
    Success、Precision 和 Normalized Precision 三条曲线中均计为定位失败，
    即使 tracker 正确输出 absent 也不会在定位分数中获益；正确拒绝应由
    presence 指标衡量。这与 LaSOT 等长时 benchmark 的常规结果口径兼容。

    此函数只产生单序列曲线。数据集总分必须先分别调用本函数，再对各序列
    曲线做等权宏平均，不能把全部帧拼接后微平均。
    """

    success_values = tuple(success_thresholds or DEFAULT_SUCCESS_THRESHOLDS)
    precision_values = tuple(precision_thresholds or DEFAULT_PRECISION_THRESHOLDS)
    normalized_values = tuple(
        normalized_precision_thresholds or DEFAULT_NORMALIZED_PRECISION_THRESHOLDS
    )
    accumulator = _LocalizationAccumulator(success_values, precision_values, normalized_values)
    present_frames = absent_frames = invalid_present_frames = unknown_frames = 0

    for frame in frames:
        if frame.gt_presence not in {"present", "absent"}:
            unknown_frames += 1
            continue
        if frame.gt_presence == "absent":
            absent_frames += 1
            accumulator.add(-1.0, math.inf, math.inf)
            continue

        present_frames += 1
        if frame.gt_bbox is None:
            # 与标准 benchmark 的全序列分母一致：显式 present 但标注框无效的
            # 帧仍占分母，并在定位曲线中记为失败，同时单独计数便于排查数据。
            invalid_present_frames += 1
            accumulator.add(-1.0, math.inf, math.inf)
            continue
        prediction = _benchmark_prediction(frame)
        overlap = bbox_iou(prediction, frame.gt_bbox)
        pixel_error, normalized_error = center_errors(prediction, frame.gt_bbox)
        accumulator.add(overlap, pixel_error, normalized_error)

    output = accumulator.result()
    output.update(
        {
            "aggregation": "single_sequence",
            "scope": "all_gt_labeled_frames",
            "absent_policy": "count_as_localization_failure",
            "present_frames": present_frames,
            "absent_frames": absent_frames,
            "invalid_present_frames": invalid_present_frames,
            "unknown_gt_frames": unknown_frames,
        }
    )
    return output


def evaluate_cognitive_visible_only(
    frames: Iterable[CanonicalFrame],
    *,
    success_thresholds: Optional[Sequence[float]] = None,
    precision_thresholds: Optional[Sequence[float]] = None,
    normalized_precision_thresholds: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """计算仅 GT 可见帧上的认知定位诊断（帧级微平均）。

    只在 GT 为 present 且 GT 框有效的帧上评测。预测 absent、uncertain、
    跳过或执行错误均视为该 present 帧定位失败（IoU=0、中心误差=Inf），
    从而不会因少做决策而获得虚高分。该组指标用于定位能力诊断，不替代
    benchmark 的按序列宏平均总分。
    """

    success_values = tuple(success_thresholds or DEFAULT_SUCCESS_THRESHOLDS)
    precision_values = tuple(precision_thresholds or DEFAULT_PRECISION_THRESHOLDS)
    normalized_values = tuple(
        normalized_precision_thresholds or DEFAULT_NORMALIZED_PRECISION_THRESHOLDS
    )
    accumulator = _LocalizationAccumulator(success_values, precision_values, normalized_values)
    overlap_sum = 0.0

    for frame in frames:
        if frame.gt_presence != "present" or frame.gt_bbox is None:
            continue
        prediction = _committed_prediction(frame)
        overlap = bbox_iou(prediction, frame.gt_bbox)
        pixel_error, normalized_error = center_errors(prediction, frame.gt_bbox)
        accumulator.add(overlap, pixel_error, normalized_error)
        overlap_sum += overlap

    output = accumulator.result()
    output.update(
        {
            "aggregation": "frame_micro_average",
            "scope": "gt_present_with_valid_bbox",
            "mean_iou": safe_div(overlap_sum, accumulator.evaluated),
        }
    )
    return output


def evaluate_tracking(
    frames: Iterable[CanonicalFrame],
    *,
    success_thresholds: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """兼容旧调用名，等价于 :func:`evaluate_cognitive_visible_only`。"""

    return evaluate_cognitive_visible_only(
        frames,
        success_thresholds=success_thresholds,
    )


def aggregate_benchmark_standard(
    sequence_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """对单序列 benchmark 曲线做等权宏平均。

    每个序列无论长短只贡献一份曲线，符合 pytracking 数据集级报告方式。
    ``evaluated_frames`` 等计数字段仍求和，便于核对输入规模，但绝不拿这些
    帧数作为曲线权重。
    """

    all_results = list(sequence_results)
    valid_results = [
        result
        for result in all_results
        if int(result.get("evaluated_frames", 0)) > 0
        and result.get("success_curve") is not None
        and result.get("precision_curve") is not None
        and result.get("normalized_precision_curve") is not None
    ]
    reference_thresholds: Mapping[str, Sequence[float]] = (
        valid_results[0]["thresholds"]
        if valid_results
        else {
            "success": DEFAULT_SUCCESS_THRESHOLDS,
            "precision_pixels": DEFAULT_PRECISION_THRESHOLDS,
            "normalized_precision": DEFAULT_NORMALIZED_PRECISION_THRESHOLDS,
        }
    )
    thresholds = {
        name: [float(value) for value in values]
        for name, values in reference_thresholds.items()
    }

    curve_specs = {
        "success_curve": "success",
        "precision_curve": "precision_pixels",
        "normalized_precision_curve": "normalized_precision",
    }
    averaged_curves: dict[str, Optional[list[float]]] = {}
    for curve_name, threshold_name in curve_specs.items():
        expected_thresholds = thresholds[threshold_name]
        if not valid_results:
            averaged_curves[curve_name] = None
            continue
        curves: list[Sequence[float]] = []
        for result in valid_results:
            current_thresholds = [float(value) for value in result["thresholds"][threshold_name]]
            if current_thresholds != expected_thresholds:
                raise ValueError(f"序列 {curve_name} 的阈值集合不一致，无法做宏平均")
            curve = result[curve_name]
            if len(curve) != len(expected_thresholds):
                raise ValueError(f"序列 {curve_name} 长度与阈值数量不一致")
            curves.append(curve)
        averaged_curves[curve_name] = [
            sum(float(curve[index]) for curve in curves) / len(curves)
            for index in range(len(expected_thresholds))
        ]

    success_curve = averaged_curves["success_curve"]
    precision_curve = averaged_curves["precision_curve"]
    normalized_precision_curve = averaged_curves["normalized_precision_curve"]
    return {
        "aggregation": "sequence_macro_average",
        "scope": "all_gt_labeled_frames",
        "absent_policy": "count_as_localization_failure",
        "num_sequences": len(all_results),
        "evaluated_sequences": len(valid_results),
        "evaluated_frames": sum(int(result.get("evaluated_frames", 0)) for result in all_results),
        "present_frames": sum(int(result.get("present_frames", 0)) for result in all_results),
        "absent_frames": sum(int(result.get("absent_frames", 0)) for result in all_results),
        "invalid_present_frames": sum(
            int(result.get("invalid_present_frames", 0)) for result in all_results
        ),
        "unknown_gt_frames": sum(int(result.get("unknown_gt_frames", 0)) for result in all_results),
        "thresholds": thresholds,
        **averaged_curves,
        "success_auc": None if success_curve is None else sum(success_curve) / len(success_curve),
        "success_op50": _curve_value_at(thresholds["success"], success_curve, 0.5),
        "success_op75": _curve_value_at(thresholds["success"], success_curve, 0.75),
        "precision_at_20": _curve_value_at(
            thresholds["precision_pixels"],
            precision_curve,
            20.0,
        ),
        "normalized_precision_at_0_2": _curve_value_at(
            thresholds["normalized_precision"],
            normalized_precision_curve,
            0.2,
        ),
    }


def evaluate_presence(frames: Iterable[CanonicalFrame]) -> dict[str, Any]:
    """计算 present/absent 二分类与拒绝预测指标。

    ``uncertain`` 是模型的拒绝决策，而不是第三种 GT，因此不进入二分类
    混淆矩阵。执行错误或缺少预测的帧记为 unavailable。报告同时给出：

    * ``uncertain_coverage``：所有有 GT 帧中 uncertain 的比例；
    * ``decision_coverage``：模型作出 present/absent 明确判断的比例；
    * ``selective_accuracy``：仅在明确判断子集上的准确率。
    """

    tp = fp = tn = fn = 0
    total = uncertain = unavailable = 0
    for frame in frames:
        if frame.gt_presence not in {"present", "absent"}:
            continue
        total += 1
        pred = frame.pred_presence
        if pred == "uncertain":
            uncertain += 1
            continue
        if pred not in {"present", "absent"} or is_execution_error(frame.execution_status):
            unavailable += 1
            continue
        if frame.gt_presence == "present" and pred == "present":
            tp += 1
        elif frame.gt_presence == "present" and pred == "absent":
            fn += 1
        elif frame.gt_presence == "absent" and pred == "present":
            fp += 1
        else:
            tn += 1

    decided = tp + fp + tn + fn
    gt_present = tp + fn
    gt_absent = tn + fp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = safe_div(2.0 * precision * recall, precision + recall)

    return {
        "evaluated_frames": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": {
            "gt_present": {"pred_present": tp, "pred_absent": fn},
            "gt_absent": {"pred_present": fp, "pred_absent": tn},
        },
        "uncertain_frames": uncertain,
        "unavailable_frames": unavailable,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": safe_div(fp, gt_absent),
        "miss_rate": safe_div(fn, gt_present),
        "uncertain_coverage": safe_div(uncertain, total),
        "decision_coverage": safe_div(decided, total),
        "unavailable_rate": safe_div(unavailable, total),
        "selective_accuracy": safe_div(tp + tn, decided),
    }


def evaluate_identity(frames: Iterable[CanonicalFrame]) -> dict[str, Any]:
    """在存在 same/different GT 的数据上计算身份判别指标。

    当前 benchmark 若没有身份标签，此节会明确返回 ``evaluated_frames=0``，
    不会根据 bbox 或 presence 伪造身份 GT。
    """

    tp = fp = tn = fn = uncertain = unavailable = 0
    for frame in frames:
        if frame.gt_identity not in {"same", "different"}:
            continue
        pred = frame.pred_identity
        if pred == "uncertain":
            uncertain += 1
            continue
        if pred not in {"same", "different"} or is_execution_error(frame.execution_status):
            unavailable += 1
            continue
        if frame.gt_identity == "same" and pred == "same":
            tp += 1
        elif frame.gt_identity == "same" and pred == "different":
            fn += 1
        elif frame.gt_identity == "different" and pred == "same":
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn + uncertain + unavailable
    decided = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "evaluated_frames": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "uncertain_frames": uncertain,
        "unavailable_frames": unavailable,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "decision_coverage": safe_div(decided, total),
        "selective_accuracy": safe_div(tp + tn, decided),
    }


def evaluate_execution(frames: Iterable[CanonicalFrame]) -> dict[str, Any]:
    """统计执行状态和真实模型调用覆盖率。"""

    frame_list = frames if isinstance(frames, Sequence) else list(frames)
    counts = Counter(frame.execution_status for frame in frame_list)
    error_count = sum(count for state, count in counts.items() if is_execution_error(state))
    observation_labeled = sum(frame.is_observation_frame is not None for frame in frame_list)
    observations = sum(frame.is_observation_frame is True for frame in frame_list)
    return {
        "total_frames": len(frame_list),
        "status_counts": dict(sorted(counts.items())),
        "error_frames": error_count,
        "error_rate": safe_div(error_count, len(frame_list)),
        "observation_labeled_frames": observation_labeled,
        "observation_frames": observations,
        "observation_rate": safe_div(observations, observation_labeled),
    }


def evaluate_reappearance(
    frames: Iterable[CanonicalFrame],
    *,
    recovery_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """评测目标从 absent 转为 present 后的恢复速度。

    一个 reappearance 事件定义为 GT 从 absent 到 present 的转移。恢复定义为
    当前可见区间内首次同时满足：预测 present、预测框有效、IoU 达到阈值，
    且身份判断没有明确给出 different。重现当帧找回的 delay 为 0。
    """

    ordered = sorted(frames, key=lambda frame: frame.frame_id)
    event_starts: list[int] = []
    for index in range(1, len(ordered)):
        if ordered[index - 1].gt_presence == "absent" and ordered[index].gt_presence == "present":
            event_starts.append(index)

    delays: list[int] = []
    unrecovered = 0
    for start_index in event_starts:
        start_frame = ordered[start_index]
        recovered_delay: Optional[int] = None
        for current in ordered[start_index:]:
            if current.gt_presence != "present":
                break
            if (
                not is_execution_error(current.execution_status)
                and current.pred_presence == "present"
                and current.pred_identity != "different"
                and bbox_iou(current.pred_bbox, current.gt_bbox) >= recovery_iou_threshold
            ):
                recovered_delay = max(0, current.frame_id - start_frame.frame_id)
                break
        if recovered_delay is None:
            unrecovered += 1
        else:
            delays.append(recovered_delay)

    return {
        "iou_threshold": recovery_iou_threshold,
        "events": len(event_starts),
        "recovered_events": len(delays),
        "unrecovered_events": unrecovered,
        "recovery_rate": safe_div(len(delays), len(event_starts)),
        "mean_recovery_delay": safe_div(sum(delays), len(delays)),
        "median_recovery_delay": float(median(delays)) if delays else None,
        "max_recovery_delay": max(delays) if delays else None,
        # 保留事件级 delay，便于跨序列计算真实中位数；CSV 写出时会跳过列表。
        "recovery_delays": delays,
    }
