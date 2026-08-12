#!/usr/bin/env python3
"""从公开跟踪训练集合成 Stage-1 VLM 跟踪与存在性数据。

该入口固定使用 LaSOT、TNL2K、MGIT 的官方 ``train`` 划分，不提供切换到
test/val 的参数。每条样本只监督：

1. 根据完整初始化帧及其文本坐标，判断同一目标在当前帧中是否存在；
2. 目标存在时在当前帧中的官方 Qwen grounding 框。

Stage-1 不使用数据集自然语言描述，不构造 memory、reasoning 或置信度标签。
present/absent 均来自同一训练视频的真实逐帧标注；默认以 case 为单位保持约 7:3。
生成结果包含自包含图片资产、可审计源 JSONL，以及已经按完整序列划分并校验通过
的 ms-swift train/val JSONL。

小规模检查：

    python tracking/synthesize_stage1_dataset.py \
        --datasets tnl2k --limit-sequences-per-dataset 50 \
        --max-samples-per-sequence 4 --absent-ratio 0.3 --context-mode pair \
        --output-dir data/stage1_debug

正式构建（第一轮先用 pair 学习基础跟踪与存在性）：

    python tracking/synthesize_stage1_dataset.py \
        --datasets lasot tnl2k mgit --context-mode pair \
        --max-samples-per-sequence 20 --absent-ratio 0.3 \
        --output-dir data/stage1_tracking_presence_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cogtrack.training import (  # noqa: E402
    QWEN_MODEL_FAMILIES,
    TemporalCaseSamplingPlan,
    TrackingSampleConfig,
    build_tracking_samples,
    export_qwen_grounding_records,
    plan_temporal_presence_cases,
    read_jsonl,
    split_records_by_sequence,
    validate_records,
    write_jsonl,
)
from pytracking.datasets import iter_dataset  # noqa: E402
from pytracking.datasets.mgit import load_split_definition  # noqa: E402
from pytracking.evaluation.data import Sequence  # noqa: E402
from pytracking.evaluation.environment import EnvironmentSettings, load_environment  # noqa: E402

STAGE1_DATASETS = ("lasot", "tnl2k", "mgit")
DATASET_VERSION = "cogtrack.stage1_tracking_presence.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 LaSOT/TNL2K/MGIT 官方训练集构造 Stage-1 VLM 跟踪与存在性数据。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=STAGE1_DATASETS,
        default=list(STAGE1_DATASETS),
        help="参与合成的训练集；默认使用全部三个来源。",
    )
    parser.add_argument("--env-config", help="本机环境 YAML；默认按 CognitiveTrack 规则发现。")
    parser.add_argument("--output-dir", required=True, help="图片、源标注和 ms-swift JSONL 输出根目录。")
    parser.add_argument(
        "--context-mode",
        choices=("pair", "mosaic", "both"),
        default="pair",
        help="首轮建议 pair；both 会额外生成 mosaic，适合后续时序上下文实验。",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="候选当前帧步长。")
    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=20,
        help="每序列最多抽取的候选帧数；both 最多产生约两倍训练样本。",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="仅在 absent-ratio=0 时使用全部合法 present 帧；会覆盖每序列上限。",
    )
    parser.add_argument(
        "--absent-ratio",
        type=float,
        default=0.3,
        help="真实消失帧 case 的全局比例，默认 0.3；设为 0 可复现纯正样本预热。",
    )
    parser.add_argument(
        "--limit-sequences-per-dataset",
        type=int,
        help="每个数据集仅取前 N 个官方训练序列，仅用于 dry-run。",
    )
    parser.add_argument("--history-size", type=int, default=4, help="mosaic 最多包含的过去可信帧数。")
    parser.add_argument("--mosaic-panel-height", type=int, default=240)
    parser.add_argument(
        "--history-corruption-ratio",
        type=float,
        default=0.0,
        help="额外生成带单个错误历史框的 mosaic 比例；默认 0，建议鲁棒性版使用 0.15。",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=648,
        help="导出图片长边上限；默认与当前本地 Qwen 推理配置一致。",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--reuse-existing-assets",
        action="store_true",
        help="允许复用输出目录中已有的同名图片资产；适合通过硬链接共享 Stage-1 图片。",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05, help="从各训练来源中按序列划分验证集。")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--mgit-version",
        choices=("tiny", "full"),
        default="tiny",
        help="MGIT 训练 split 版本；当前公共镜像完整提供 tiny/train。",
    )
    parser.add_argument(
        "--allow-missing-mgit-sequences",
        action="store_true",
        help="显式跳过本地 MGIT split 中无帧文件的序列，并在报告中记录实际序列数。",
    )
    parser.add_argument(
        "--sampling-plan",
        help=(
            "重放既有 sampling_plan.json；提供后不重新抽帧，并严格校验数据集、seed、"
            "正负比例和每序列 case 上限。正式跨服务器复现建议始终使用。"
        ),
    )
    parser.add_argument(
        "--qwen-model-families",
        nargs="+",
        choices=QWEN_MODEL_FAMILIES,
        default=list(QWEN_MODEL_FAMILIES),
        help="生成模型专属训练视图；默认同时导出 Qwen2.5-VL 与 Qwen3-VL。",
    )
    parser.add_argument("--force", action="store_true", help="覆盖同名构建产物；不会删除其他目录。")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只生成并审计 sampling_plan.json，不读取或编码图片。",
    )
    return parser


def _validate_train_sequence(sequence: Sequence, expected_dataset: str) -> None:
    """在读图前执行训练来源审计，阻止测试序列进入合成流程。"""

    source_split = str(sequence.metadata.get("split", "")).lower()
    if sequence.dataset != expected_dataset:
        raise ValueError(
            f"数据集身份不一致：请求 {expected_dataset}，Sequence 标记为 {sequence.dataset}"
        )
    if source_split != "train":
        raise ValueError(
            f"拒绝非训练划分：{sequence.dataset}/{sequence.name} split={source_split!r}"
        )


def _checked_sequences(
    first: Sequence,
    remainder: Iterator[Sequence],
    *,
    dataset_name: str,
) -> Iterator[Sequence]:
    for sequence in chain((first,), remainder):
        _validate_train_sequence(sequence, dataset_name)
        yield sequence


def _prepare_train_sequences(
    dataset_names: Iterable[str],
    *,
    environment: EnvironmentSettings,
    limit_per_dataset: int | None,
    mgit_version: str,
    allow_missing_mgit_sequences: bool,
) -> Iterator[Sequence]:
    """先探测每个来源的首个序列，再开始写文件，尽早暴露未解压数据。"""

    prepared: list[tuple[str, Sequence, Iterator[Sequence]]] = []
    for dataset_name in dataset_names:
        kwargs: dict[str, Any] = {"split": "train"}
        if dataset_name == "mgit":
            kwargs["version"] = mgit_version
            if allow_missing_mgit_sequences:
                root = environment.dataset_root("mgit") / "data" / "train"
                names = load_split_definition(mgit_version, "train")
                names = [
                    name
                    for name in names
                    if (root / name / f"frame_{name}").is_dir()
                    and any((root / name / f"frame_{name}").iterdir())
                ]
                if not names:
                    raise ValueError("MGIT train 没有包含帧文件的可用序列")
                kwargs["sequence_names"] = names
        iterator = iter_dataset(
            dataset_name,
            environment=environment,
            limit=limit_per_dataset,
            **kwargs,
        )
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"{dataset_name} train 中没有可用序列") from exc
        _validate_train_sequence(first, dataset_name)
        prepared.append((dataset_name, first, iterator))

    for dataset_name, first, remainder in prepared:
        yield from _checked_sequences(first, remainder, dataset_name=dataset_name)


def _source_seed(base_seed: int, source: str) -> int:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], byteorder="big")


def _split_by_source(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """先按数据来源分层，再按完整序列划分，避免小来源被随机划空。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source = str(record.get("metadata", {}).get("source_dataset", "unknown"))
        grouped[source].append(record)

    output = {"train": [], "val": []}
    for source in sorted(grouped):
        split = split_records_by_sequence(
            grouped[source],
            val_ratio=val_ratio,
            test_ratio=0.0,
            seed=_source_seed(seed, source),
        )
        output["train"].extend(split["train"])
        output["val"].extend(split["val"])
    return output


def _sequence_count(records: Iterable[dict[str, Any]]) -> int:
    return len(
        {
            (
                str(row.get("metadata", {}).get("source_dataset")),
                str(row.get("metadata", {}).get("source_sequence")),
            )
            for row in records
        }
    )


def _load_sampling_plan(path: str | Path) -> TemporalCaseSamplingPlan:
    plan_path = Path(path).expanduser().resolve()
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("sampling plan 顶层必须是 JSON 对象")
    return TemporalCaseSamplingPlan.from_dict(payload)


def _validate_replayed_plan(
    plan: TemporalCaseSamplingPlan,
    *,
    dataset_names: list[str],
    seed: int,
    absent_ratio: float,
    max_samples_per_sequence: int,
) -> None:
    plan_datasets = {item.dataset for item in plan.sequences}
    if plan_datasets != set(dataset_names):
        raise ValueError(
            f"sampling plan 数据集不匹配：plan={sorted(plan_datasets)} cli={sorted(dataset_names)}"
        )
    if plan.seed != seed:
        raise ValueError(f"sampling plan seed={plan.seed}，与 CLI seed={seed} 不一致")
    if abs(plan.requested_absent_ratio - absent_ratio) > 1e-12:
        raise ValueError(
            "sampling plan requested_absent_ratio="
            f"{plan.requested_absent_ratio}，与 CLI absent_ratio={absent_ratio} 不一致"
        )
    if plan.max_cases_per_sequence != max_samples_per_sequence:
        raise ValueError(
            "sampling plan max_cases_per_sequence="
            f"{plan.max_cases_per_sequence}，与 CLI max_samples_per_sequence="
            f"{max_samples_per_sequence} 不一致"
        )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = Counter(str(row["metadata"]["source_dataset"]) for row in records)
    contexts = Counter(str(row["metadata"]["effective_mode"]) for row in records)
    states = Counter(str(row["target_status"]) for row in records)
    return {
        "samples": len(records),
        "sequences": _sequence_count(records),
        "samples_by_dataset": dict(sorted(datasets.items())),
        "samples_by_context": dict(sorted(contexts.items())),
        "samples_by_state": dict(sorted(states.items())),
    }


def _export_ms_swift(
    source_path: Path,
    *,
    output_root: Path,
    val_ratio: float,
    seed: int,
    requested_absent_ratio: float,
    model_families: list[str],
) -> dict[str, Any]:
    canonical_records = list(read_jsonl(source_path))
    if not canonical_records:
        raise ValueError("源 JSONL 中没有样本")

    # 这里重复检查 Stage-1 的科学边界，防止未来通用构造器改动后静默污染数据。
    for index, record in enumerate(canonical_records):
        metadata = record.get("metadata", {})
        if metadata.get("source_split") != "train":
            raise ValueError(f"样本 {index} 不是 train 来源")
        if metadata.get("used_language_description") is not False:
            raise ValueError(f"样本 {index} 意外使用了语言描述")
        if record.get("target_status") not in {"present", "absent"}:
            raise ValueError(f"样本 {index} 缺少合法 present/absent 标签")

    # 同一 current 可与不同 earlier reference 构成独立 case；鲁棒性变体则与其 clean
    # case 共享 reference/current。比例按唯一 (reference, current) 统计，避免 corruption
    # 数量改变原 sampling plan 的正负样本定义。
    unique_cases: dict[tuple[str, str, int, int], str] = {}
    for record in canonical_records:
        metadata = record["metadata"]
        key = (
            str(metadata["source_dataset"]),
            str(metadata["source_sequence"]),
            int(metadata["reference_frame_id"]),
            int(metadata["frame_id"]),
        )
        state = str(record["target_status"])
        if key in unique_cases and unique_cases[key] != state:
            raise ValueError(f"同一 case 出现冲突标签: {key}")
        unique_cases[key] = state
    absent_cases = sum(state == "absent" for state in unique_cases.values())
    actual_absent_ratio = absent_cases / len(unique_cases)
    ratio_tolerance = 1.0 / len(unique_cases)
    if abs(actual_absent_ratio - requested_absent_ratio) > ratio_tolerance:
        raise ValueError(
            f"唯一 reference/current case 的 absent 比例不符合计划："
            f"actual={actual_absent_ratio:.6f} requested={requested_absent_ratio:.6f}"
        )

    swift_root = output_root / "ms_swift"
    swift_root.mkdir(parents=True, exist_ok=True)
    splits = _split_by_source(canonical_records, val_ratio=val_ratio, seed=seed)
    training_views: dict[str, Any] = {}
    for family in dict.fromkeys(model_families):
        family_root = swift_root / family
        family_stats: dict[str, Any] = {}
        export_total: dict[str, Any] | None = None
        for split_name, rows in splits.items():
            records, export_report = export_qwen_grounding_records(
                rows,
                image_root=output_root,
                model_family=family,
            )
            validation = validate_records(records, image_root=output_root, check_images=True)
            if not validation.ok:
                preview = "; ".join(issue.message for issue in validation.errors[:5])
                raise ValueError(f"{family}/{split_name} ms-swift 样本校验失败：{preview}")
            write_jsonl(family_root / f"{split_name}.jsonl", records)
            (family_root / f"validation_{split_name}.json").write_text(
                json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            family_stats[split_name] = _summarize(records)
            if export_total is None:
                export_total = export_report.to_dict()
            else:
                export_total["sample_count"] += export_report.sample_count
                export_total["present_count"] += export_report.present_count
                export_total["absent_count"] += export_report.absent_count
                export_total["bbox_placeholder_count"] += export_report.bbox_placeholder_count
        training_views[family] = {
            "export": export_total,
            "partitions": family_stats,
        }
        (family_root / "dataset_info.json").write_text(
            json.dumps(
                {
                    "schema_version": "cogtrack.qwen_grounding_dataset.v1",
                    "model_family": family,
                    "qwen_vl_bbox_format": "new",
                    "coordinate_conversion_owner": "ms-swift model template",
                    **training_views[family],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "case_count": len(unique_cases),
        "present_case_count": len(unique_cases) - absent_cases,
        "absent_case_count": absent_cases,
        "absent_case_ratio": actual_absent_ratio,
        "partitions": {name: _summarize(rows) for name, rows in splits.items()},
        "training_views": training_views,
    }
    (swift_root / "dataset_info.json").write_text(
        json.dumps(
            {
                "schema_version": "cogtrack.qwen_grounding_multi_family.v1",
                "source": source_path.relative_to(output_root).as_posix(),
                "dataset_root": ".",
                "split_method": "source_stratified_complete_sequence",
                "qwen_vl_bbox_format": "new",
                "families": {
                    family: training_views[family] for family in dict.fromkeys(model_families)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.limit_sequences_per_dataset is not None and args.limit_sequences_per_dataset <= 0:
            raise ValueError("--limit-sequences-per-dataset 必须为正整数")
        if not 0 <= args.val_ratio < 1:
            raise ValueError("--val-ratio 必须位于 [0,1)")
        if not 0 <= args.absent_ratio < 1:
            raise ValueError("--absent-ratio 必须位于 [0,1)")
        if args.all_frames and args.absent_ratio != 0:
            raise ValueError("--all-frames 只适用于 --absent-ratio 0；比例采样必须设置每序列上限")
        dataset_names = list(dict.fromkeys(args.datasets))
        environment = load_environment(args.env_config)
        output_root = Path(args.output_dir).expanduser().resolve()
        sampling_plan = None
        frame_ids_by_sequence = None
        anchor_frame_ids_by_sequence = None
        reference_frame_ids_by_sequence = None
        if args.sampling_plan:
            if args.absent_ratio <= 0:
                raise ValueError("--sampling-plan 只适用于包含 presence/absence 规划的数据")
            sampling_plan = _load_sampling_plan(args.sampling_plan)
            _validate_replayed_plan(
                sampling_plan,
                dataset_names=dataset_names,
                seed=args.seed,
                absent_ratio=args.absent_ratio,
                max_samples_per_sequence=args.max_samples_per_sequence,
            )
            frame_ids_by_sequence = sampling_plan.frame_ids_by_sequence
            anchor_frame_ids_by_sequence = sampling_plan.anchor_frame_ids_by_sequence
            reference_frame_ids_by_sequence = sampling_plan.reference_frame_ids_by_sequence
        elif args.absent_ratio > 0:
            sampling_plan = plan_temporal_presence_cases(
                _prepare_train_sequences(
                    dataset_names,
                    environment=environment,
                    limit_per_dataset=args.limit_sequences_per_dataset,
                    mgit_version=args.mgit_version,
                    allow_missing_mgit_sequences=args.allow_missing_mgit_sequences,
                ),
                max_cases_per_sequence=args.max_samples_per_sequence,
                absent_ratio=args.absent_ratio,
                frame_stride=args.frame_stride,
                seed=args.seed,
            )
            frame_ids_by_sequence = sampling_plan.frame_ids_by_sequence
            anchor_frame_ids_by_sequence = sampling_plan.anchor_frame_ids_by_sequence
            reference_frame_ids_by_sequence = sampling_plan.reference_frame_ids_by_sequence
        sequences = _prepare_train_sequences(
            dataset_names,
            environment=environment,
            limit_per_dataset=args.limit_sequences_per_dataset,
            mgit_version=args.mgit_version,
            allow_missing_mgit_sequences=args.allow_missing_mgit_sequences,
        )
        if args.plan_only:
            if sampling_plan is None:
                raise ValueError("--plan-only 需要 presence sampling plan；请保持 --absent-ratio > 0")
            output_root.mkdir(parents=True, exist_ok=True)
            plan_path = output_root / "sampling_plan.json"
            if plan_path.exists() and not args.force:
                raise FileExistsError(f"sampling plan 已存在；如需覆盖请启用 --force: {plan_path}")
            plan_path.write_text(
                json.dumps(sampling_plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(sampling_plan.to_dict(include_frame_ids=False), ensure_ascii=False, indent=2))
            print(f"[CognitiveTrack] sampling plan 已生成: {plan_path}")
            return 0
        config = TrackingSampleConfig(
            mode=args.context_mode,
            frame_stride=args.frame_stride,
            max_samples_per_sequence=(
                None if args.all_frames or frame_ids_by_sequence is not None
                else args.max_samples_per_sequence
            ),
            seed=args.seed,
            history_size=args.history_size,
            mosaic_panel_height=args.mosaic_panel_height,
            history_corruption_ratio=args.history_corruption_ratio,
            present_only=args.absent_ratio == 0,
            use_language_description=False,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
            reuse_existing_assets=args.reuse_existing_assets,
        )
        build_report = build_tracking_samples(
            sequences,
            output_root,
            config=config,
            overwrite=args.force,
            frame_ids_by_sequence=frame_ids_by_sequence,
            anchor_frame_ids_by_sequence=anchor_frame_ids_by_sequence,
            reference_frame_ids_by_sequence=reference_frame_ids_by_sequence,
        )
        if args.sampling_plan:
            if build_report.sequence_count != sampling_plan.sequence_count:
                raise ValueError(
                    "原始数据缺少 sampling plan 中的序列："
                    f"expected={sampling_plan.sequence_count} actual={build_report.sequence_count}"
                )
            if args.history_corruption_ratio == 0:
                if build_report.sample_count != sampling_plan.case_count:
                    raise ValueError(
                        "重建 case 数与 sampling plan 不一致："
                        f"expected={sampling_plan.case_count} actual={build_report.sample_count}"
                    )
                if (
                    build_report.present_count != sampling_plan.present_count
                    or build_report.absent_count != sampling_plan.absent_count
                ):
                    raise ValueError(
                        "原始数据的 present/absent 标注已与 sampling plan 发生变化："
                        f"expected={sampling_plan.present_count}/{sampling_plan.absent_count} "
                        f"actual={build_report.present_count}/{build_report.absent_count}"
                    )
            else:
                unique_cases: dict[tuple[str, str, int, int], str] = {}
                for record in read_jsonl(output_root / build_report.source_jsonl):
                    metadata = record["metadata"]
                    key = (
                        str(metadata["source_dataset"]),
                        str(metadata["source_sequence"]),
                        int(metadata["reference_frame_id"]),
                        int(metadata["frame_id"]),
                    )
                    state = str(record["target_status"])
                    if key in unique_cases and unique_cases[key] != state:
                        raise ValueError(f"corrupted 版本出现 case 标签冲突: {key}")
                    unique_cases[key] = state
                unique_present = sum(value == "present" for value in unique_cases.values())
                unique_absent = sum(value == "absent" for value in unique_cases.values())
                if (
                    len(unique_cases) != sampling_plan.case_count
                    or unique_present != sampling_plan.present_count
                    or unique_absent != sampling_plan.absent_count
                ):
                    raise ValueError(
                        "corrupted 版本去重后的 case/状态与 sampling plan 不一致："
                        "expected="
                        f"{sampling_plan.case_count}/{sampling_plan.present_count}/"
                        f"{sampling_plan.absent_count} "
                        f"actual={len(unique_cases)}/{unique_present}/{unique_absent}"
                    )
        if sampling_plan is not None:
            (output_root / "sampling_plan.json").write_text(
                json.dumps(sampling_plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        split_stats = _export_ms_swift(
            output_root / build_report.source_jsonl,
            output_root=output_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
            requested_absent_ratio=args.absent_ratio,
            model_families=list(dict.fromkeys(args.qwen_model_families)),
        )
        dataset_info = {
            "schema_version": DATASET_VERSION,
            "task": "stage1_tracking_presence",
            "source_datasets": dataset_names,
            "source_split": "train",
            "mgit_version": args.mgit_version if "mgit" in dataset_names else None,
            "allow_missing_mgit_sequences": args.allow_missing_mgit_sequences,
            "target_status": ["present", "absent"] if args.absent_ratio > 0 else ["present"],
            "canonical_bbox_format": "norm1000_xyxy",
            "reference_mode": "full_frame_bbox_text",
            "reference_canonical_bbox_format": "norm1000_xyxy",
            "training_coordinate_protocols": {
                "qwen2_5_vl": "processor-resized absolute pixel xyxy",
                "qwen3_vl": "relative 0-to-1000 xyxy",
            },
            "coordinate_conversion_owner": "ms-swift model template",
            "qwen_vl_bbox_format": "new",
            "uses_language_description": False,
            "supervises_memory": False,
            "supervises_confidence": False,
            "context_mode": args.context_mode,
            "sequence_split": "source-stratified complete-sequence split",
            "negative_source": "same-sequence ground-truth absent frames only",
            "requested_absent_case_ratio": args.absent_ratio,
            "seed": args.seed,
            "build": build_report.to_dict(),
            "sampling_plan": (
                sampling_plan.to_dict(include_frame_ids=False) if sampling_plan is not None else None
            ),
            "splits": split_stats,
        }
        (output_root / "dataset_info.json").write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    print("[CognitiveTrack] Stage-1 跟踪与存在性数据构建完成")
    print(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    print(f"[训练] DATASET_ROOT={output_root}")
    for family in dict.fromkeys(args.qwen_model_families):
        print(f"[训练:{family}] TRAIN_DATA={output_root / 'ms_swift' / family / 'train.jsonl'}")
        print(f"[训练:{family}] VAL_DATA={output_root / 'ms_swift' / family / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
