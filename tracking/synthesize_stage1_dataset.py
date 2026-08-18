#!/usr/bin/env python3
"""VLT-v6.4 合成器的内部 CLI 实现。

正式用户入口只有 ``tracking/synthesize_vlt_v6_dataset.py``。本文件保留旧文件名是为了
避免在 8×4090 迁移前进行近千行机械重命名；它不是可独立选择旧 Stage-1 协议的入口。

该入口固定使用 LaSOT、TNL2K、MGIT 的官方 ``train`` 划分，不提供切换到
test/val 的参数。每条样本只监督：

1. 根据完整过去参考图中的目标指代，判断同一目标在当前帧中是否存在；
2. 目标存在时在当前帧中的官方 Qwen grounding 框。

实现内部仍保留少量 legacy profile 供归档测试重放，但当前命令只能通过 v6.4 wrapper
进入。VLT-v6.4 固定“带框初始化图 + 历史轨迹 mosaic + 当前搜索图”，加入初始化文本
和最近状态记忆，并使用三字段输出；``tracking_sft`` 只 mask 未知的
``memory_update`` 值。当前 profile 不构造 reasoning、置信度或旧六分类标签。
present/absent 均来自同一训练视频的真实逐帧标注；默认以 case 为单位保持约 7:3。
生成结果包含自包含图片资产、可审计源 JSONL，以及已经按完整序列划分并校验通过
的 ms-swift train/val JSONL。

小规模检查：

    python tracking/synthesize_vlt_v6_dataset.py \
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

from cogtrack.context import (  # noqa: E402
    PROMPT_PROFILE_VISUAL_V5,
    PROMPT_PROFILE_VLT_V6,
    REFERENCE_MODE_BBOX_TEXT,
    REFERENCE_MODE_VISUAL_BOX,
    REFERENCE_MODES,
)
from cogtrack.training import (  # noqa: E402
    MEMORY_SUPERVISION_DISABLED,
    MEMORY_SUPERVISION_EXPLICIT,
    MEMORY_SUPERVISION_LABELLED_MODES,
    MEMORY_SUPERVISION_MASKED_NULL,
    MEMORY_SUPERVISION_MODES,
    MEMORY_SUPERVISION_THREE_STATE,
    QWEN_MODEL_FAMILIES,
    REFERENCE_POLICY_FIXED_ANCHOR,
    REFERENCE_POLICY_SAMPLED_PRIOR,
    SFT_SUPERVISION_FULL,
    SFT_SUPERVISION_PROFILES,
    SFT_SUPERVISION_TRACKING_SFT,
    TemporalCaseSamplingPlan,
    TrackingSampleConfig,
    build_tracking_samples,
    export_qwen_grounding_records,
    load_memory_labels_jsonl,
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
VISUAL_V5_DATASET_VERSION = "cogtrack.visual_tracking_probe.v5"
VLT_V6_DATASET_VERSION = "cogtrack.vlt_tracking_sft.v6.4"
SYNTHESIS_PROFILES = ("legacy_stage1", "visual_v5", "vlt_v6")


def _parser(profile: str = "legacy_stage1") -> argparse.ArgumentParser:
    if profile not in SYNTHESIS_PROFILES:
        raise ValueError(f"profile 必须是 {SYNTHESIS_PROFILES} 之一")
    is_visual_v5 = profile == "visual_v5"
    is_vlt_v6 = profile == "vlt_v6"
    is_visual_reference = is_visual_v5 or is_vlt_v6
    parser = argparse.ArgumentParser(
        description=(
            "从 LaSOT/TNL2K/MGIT 官方训练集构造 VLT-v6 tracking SFT 数据。"
            if is_vlt_v6
            else "从 LaSOT/TNL2K/MGIT 官方训练集构造 visual-v5 三字段跟踪数据。"
            if is_visual_v5
            else "从 LaSOT/TNL2K/MGIT 官方训练集构造旧 Stage-1 跟踪与存在性数据。"
        ),
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
        default="mosaic" if is_vlt_v6 else "both" if is_visual_v5 else "pair",
        help=(
            "VLT-v6 固定 mosaic 三图；其他 profile 可选择 pair/mosaic/both。"
            if is_vlt_v6
            else "首轮建议 pair；both 会额外生成 mosaic，适合后续时序上下文实验。"
        ),
    )
    parser.add_argument(
        "--reference-mode",
        choices=tuple(sorted(REFERENCE_MODES)),
        default=REFERENCE_MODE_VISUAL_BOX if is_visual_reference else REFERENCE_MODE_BBOX_TEXT,
        help="visual_v5/vlt_v6 必须使用 visual_box；legacy_stage1 必须使用 bbox_text。",
    )
    parser.add_argument(
        "--memory-supervision",
        choices=tuple(sorted(MEMORY_SUPERVISION_MODES)),
        default=(
            MEMORY_SUPERVISION_MASKED_NULL
            if is_vlt_v6
            else MEMORY_SUPERVISION_EXPLICIT
            if is_visual_v5
            else MEMORY_SUPERVISION_DISABLED
        ),
        help=(
            "visual_v5 正式构建使用 explicit；vlt_v6 state_update_sft 使用 three_state（标签允许"
            "部分覆盖：absent -> hard_null，present+标签 -> verified_update，其余 -> "
            "masked_unknown）；大规模 tracking_sft 固定使用 masked_null。"
        ),
    )
    parser.add_argument(
        "--memory-labels",
        help=(
            "explicit/three_state 模式逐帧标签 JSONL；需包含 dataset/sequence/frame_id/"
            "memory_update/source。three_state 下允许部分覆盖，由 "
            "MGIT 分段状态标签工具生成。"
        ),
    )
    parser.add_argument(
        "--sft-supervision-profile",
        choices=("auto", *sorted(SFT_SUPERVISION_PROFILES)),
        default="auto",
        help=(
            "训练视图监督档位；auto 按 memory_supervision 推断。"
            "state_update_sft 只允许全部标签已验证的状态更新数据。"
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="候选当前帧步长。")
    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=20,
        help="每序列最多抽取的候选帧数；both 最多产生约两倍训练样本。",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        action="append",
        default=None,
        metavar="DATASET=N",
        help=(
            "按数据源覆盖每序列上限，可重复，例如 --max-samples-per-dataset mgit=200。"
            "长序列数据源（MGIT 单条 7k-15k 帧）是 verified_update 的唯一来源，需要更高"
            "上限；短序列跟着抬只会重复采同一条视频。absent 配额仍逐源独立强制。"
        ),
    )
    parser.add_argument(
        "--reference-policy",
        choices=("auto", REFERENCE_POLICY_SAMPLED_PRIOR, REFERENCE_POLICY_FIXED_ANCHOR),
        default="auto",
        help=(
            "模板（Image 1）帧的选取策略。auto 按 profile 取默认值；"
            f"{REFERENCE_POLICY_FIXED_ANCHOR} 把所有 case 的模板锁死在序列首个 present 帧，"
            f"{REFERENCE_POLICY_SAMPLED_PRIOR} 为每个 case 在更早的 present 帧里确定性随机取。"
            "重放已发布 plan 时必须显式指定与该 plan 相同的策略。"
        ),
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
    parser.add_argument(
        "--history-size",
        type=int,
        default=3 if is_vlt_v6 else 4,
        help=(
            "VLT-v6 固定读取最近三次可信观测；其他 profile 为 mosaic 最大历史帧数。"
            if is_vlt_v6
            else "mosaic 最多包含的过去可信帧数。"
        ),
    )
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
        default=["qwen3_vl"] if is_visual_reference else list(QWEN_MODEL_FAMILIES),
        help=(
            "生成模型专属训练视图；视觉指代 profile 默认只导出 Qwen3-VL。"
        ),
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parse_dataset_caps(raw_values: list[str] | None) -> dict[str, int]:
    """把 ``DATASET=N`` 形式的重复参数解析成小写键 dict。"""

    caps: dict[str, int] = {}
    for raw in raw_values or []:
        text = str(raw).strip()
        if text.count("=") != 1:
            raise ValueError(f"--max-samples-per-dataset 必须是 DATASET=N 形式，实际为 {raw!r}")
        name, _, value = text.partition("=")
        key = name.strip().lower()
        if not key:
            raise ValueError(f"--max-samples-per-dataset 的 dataset 名不能为空: {raw!r}")
        try:
            cap = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"--max-samples-per-dataset 的 N 必须是整数: {raw!r}") from exc
        if cap <= 0:
            raise ValueError(f"--max-samples-per-dataset[{key}] 必须是正整数，实际为 {cap}")
        if key in caps:
            raise ValueError(f"--max-samples-per-dataset 重复指定了 {key}")
        caps[key] = cap
    return caps


def _validate_replayed_plan(
    plan: TemporalCaseSamplingPlan,
    *,
    dataset_names: list[str],
    seed: int,
    absent_ratio: float,
    max_samples_per_sequence: int,
    max_cases_by_dataset: dict[str, int],
    reference_policy: str,
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
    if plan.reference_policy != reference_policy:
        raise ValueError(
            "sampling plan reference_policy="
            f"{plan.reference_policy!r}，与当前请求的 {reference_policy!r} 不一致"
        )
    # 分数据源上限直接决定每条序列采几个 case，重放时不一致就会静默产出另一份数据。
    if plan.resolved_max_cases_by_dataset != max_cases_by_dataset:
        raise ValueError(
            "sampling plan max_cases_by_dataset="
            f"{plan.resolved_max_cases_by_dataset}，与 CLI --max-samples-per-dataset="
            f"{max_cases_by_dataset} 不一致"
        )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = Counter(str(row["metadata"]["source_dataset"]) for row in records)
    contexts = Counter(str(row["metadata"]["effective_mode"]) for row in records)
    states = Counter(str(row["target_status"]) for row in records)
    corruption = Counter(
        str(row["metadata"].get("history_corruption") or "clean") for row in records
    )
    memory_sources = Counter(
        str(source)
        for row in records
        if (source := row["metadata"].get("memory_label_source")) is not None
    )
    memory_review = Counter(
        "reviewed" if bool(row["metadata"].get("memory_label_reviewed")) else "unreviewed"
        for row in records
        if row["metadata"].get("memory_label_source") is not None
    )
    return {
        "samples": len(records),
        "sequences": _sequence_count(records),
        "samples_by_dataset": dict(sorted(datasets.items())),
        "samples_by_context": dict(sorted(contexts.items())),
        "samples_by_state": dict(sorted(states.items())),
        "samples_by_history_corruption": dict(sorted(corruption.items())),
        "samples_by_memory_label_source": dict(sorted(memory_sources.items())),
        "memory_review_status": dict(sorted(memory_review.items())),
        "memory_non_null_samples": sum(row.get("memory_update") is not None for row in records),
    }


def _export_ms_swift(
    source_path: Path,
    *,
    output_root: Path,
    val_ratio: float,
    seed: int,
    requested_absent_ratio: float,
    model_families: list[str],
    profile: str,
) -> dict[str, Any]:
    canonical_records = list(read_jsonl(source_path))
    if not canonical_records:
        raise ValueError("源 JSONL 中没有样本")

    # 这里重复检查 Stage-1 的科学边界，防止未来通用构造器改动后静默污染数据。
    for index, record in enumerate(canonical_records):
        metadata = record.get("metadata", {})
        if metadata.get("source_split") != "train":
            raise ValueError(f"样本 {index} 不是 train 来源")
        if profile == "vlt_v6":
            if metadata.get("prompt_profile") != PROMPT_PROFILE_VLT_V6:
                raise ValueError(f"样本 {index} 未使用 vlt_v6 Prompt")
            if not str(metadata.get("initial_target_text") or "").strip():
                raise ValueError(f"样本 {index} 缺少初始化目标文本或视觉锚点回退描述")
            if len(record.get("images", [])) != 3:
                raise ValueError(f"样本 {index} 不符合 vlt_v6 固定三图协议")
        elif metadata.get("used_language_description") is not False:
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


def main(*, profile: str = "legacy_stage1") -> int:
    args = _parser(profile).parse_args()
    try:
        if args.limit_sequences_per_dataset is not None and args.limit_sequences_per_dataset <= 0:
            raise ValueError("--limit-sequences-per-dataset 必须为正整数")
        if not 0 <= args.val_ratio < 1:
            raise ValueError("--val-ratio 必须位于 [0,1)")
        if not 0 <= args.absent_ratio < 1:
            raise ValueError("--absent-ratio 必须位于 [0,1)")
        if args.all_frames and args.absent_ratio != 0:
            raise ValueError("--all-frames 只适用于 --absent-ratio 0；比例采样必须设置每序列上限")
        if profile == "legacy_stage1":
            if args.reference_mode != REFERENCE_MODE_BBOX_TEXT:
                raise ValueError("legacy_stage1 profile 只允许 --reference-mode bbox_text")
            if args.memory_supervision != MEMORY_SUPERVISION_DISABLED:
                raise ValueError("legacy_stage1 profile 不监督 memory_update")
        elif profile == "visual_v5":
            if args.reference_mode != REFERENCE_MODE_VISUAL_BOX:
                raise ValueError("visual_v5 profile 只允许 --reference-mode visual_box")
            if not args.plan_only and args.memory_supervision == MEMORY_SUPERVISION_DISABLED:
                raise ValueError("visual_v5 数据必须使用三字段 memory_update 监督")
        else:
            if args.reference_mode != REFERENCE_MODE_VISUAL_BOX:
                raise ValueError("vlt_v6 profile 只允许 --reference-mode visual_box")
            if args.context_mode != "mosaic":
                raise ValueError("vlt_v6 正式协议固定 --context-mode mosaic")
            if args.memory_supervision not in {
                MEMORY_SUPERVISION_MASKED_NULL,
                MEMORY_SUPERVISION_EXPLICIT,
                MEMORY_SUPERVISION_THREE_STATE,
            }:
                raise ValueError(
                    "vlt_v6 只允许 masked_null / three_state 核心 SFT 或 explicit 记忆监督"
                )
            if args.history_size != 3:
                raise ValueError("vlt_v6 正式协议固定 --history-size 3")
        labelled = args.memory_supervision in MEMORY_SUPERVISION_LABELLED_MODES
        if labelled and not args.memory_labels:
            if not args.plan_only:
                raise ValueError(
                    f"--memory-supervision {args.memory_supervision} 必须同时提供 --memory-labels"
                )
        elif not labelled and args.memory_labels:
            raise ValueError(
                "--memory-labels 只能与 --memory-supervision explicit/three_state 同时使用"
            )
        memory_labels = (
            load_memory_labels_jsonl(args.memory_labels) if args.memory_labels else None
        )
        if args.sft_supervision_profile == "auto":
            sft_supervision_profile = (
                SFT_SUPERVISION_TRACKING_SFT
                if args.memory_supervision
                in {MEMORY_SUPERVISION_MASKED_NULL, MEMORY_SUPERVISION_THREE_STATE}
                else SFT_SUPERVISION_FULL
            )
        else:
            sft_supervision_profile = args.sft_supervision_profile
        dataset_names = list(dict.fromkeys(args.datasets))
        dataset_caps = _parse_dataset_caps(args.max_samples_per_dataset)
        if args.reference_policy != "auto":
            reference_policy = args.reference_policy
        else:
            # visual_v5 是已发布 release，默认必须保持固定锚点才能字节级重放。
            # vlt_v6 默认改为逐 case 随机模板：固定锚点让全部 case 的 Image 1 都是第 0
            # 帧，模型只见过"从视频开头初始化"这一种情形，而真实使用可以从任意帧起跟。
            # 随机模板同时把 (reference, current) 间隔从单一值摊成短中长分布。
            reference_policy = (
                REFERENCE_POLICY_FIXED_ANCHOR
                if profile == "visual_v5"
                else REFERENCE_POLICY_SAMPLED_PRIOR
            )
        environment = load_environment(args.env_config)
        output_root = Path(args.output_dir).expanduser().resolve()
        sampling_plan = None
        frame_ids_by_sequence = None
        anchor_frame_ids_by_sequence = None
        reference_frame_ids_by_sequence = None
        if args.sampling_plan:
            sampling_plan = _load_sampling_plan(args.sampling_plan)
            _validate_replayed_plan(
                sampling_plan,
                dataset_names=dataset_names,
                seed=args.seed,
                absent_ratio=args.absent_ratio,
                max_samples_per_sequence=args.max_samples_per_sequence,
                max_cases_by_dataset=dataset_caps,
                reference_policy=reference_policy,
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
                max_cases_by_dataset=dataset_caps,
                absent_ratio=args.absent_ratio,
                frame_stride=args.frame_stride,
                seed=args.seed,
                reference_policy=reference_policy,
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
            print(f"[SHA256] {_file_sha256(plan_path)}  {plan_path.name}")
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
            use_language_description=profile == "vlt_v6",
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
            reuse_existing_assets=args.reuse_existing_assets,
            reference_mode=args.reference_mode,
            memory_supervision=args.memory_supervision,
            prompt_profile=(
                PROMPT_PROFILE_VLT_V6
                if profile == "vlt_v6"
                else PROMPT_PROFILE_VISUAL_V5
            ),
            force_history_image=profile == "vlt_v6",
            sft_supervision_profile=sft_supervision_profile,
        )
        build_report = build_tracking_samples(
            sequences,
            output_root,
            config=config,
            overwrite=args.force,
            frame_ids_by_sequence=frame_ids_by_sequence,
            anchor_frame_ids_by_sequence=anchor_frame_ids_by_sequence,
            reference_frame_ids_by_sequence=reference_frame_ids_by_sequence,
            memory_labels_by_sequence=memory_labels,
        )
        if args.sampling_plan:
            if build_report.sequence_count != sampling_plan.sequence_count:
                raise ValueError(
                    "原始数据缺少 sampling plan 中的序列："
                    f"expected={sampling_plan.sequence_count} actual={build_report.sequence_count}"
                )
            # both 或 corrupted-history 会让同一 reference/current case 产生多个训练
            # 视图，因此必须按 case key 去重后与 plan 对账，不能直接比较样本行数。
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
                    raise ValueError(f"同一 case 的多个上下文视图出现标签冲突: {key}")
                unique_cases[key] = state
            unique_present = sum(value == "present" for value in unique_cases.values())
            unique_absent = sum(value == "absent" for value in unique_cases.values())
            if (
                len(unique_cases) != sampling_plan.case_count
                or unique_present != sampling_plan.present_count
                or unique_absent != sampling_plan.absent_count
            ):
                raise ValueError(
                    "构建结果去重后的 case/状态与 sampling plan 不一致："
                    "expected="
                    f"{sampling_plan.case_count}/{sampling_plan.present_count}/"
                    f"{sampling_plan.absent_count} "
                    f"actual={len(unique_cases)}/{unique_present}/{unique_absent}"
                )
        if sampling_plan is not None:
            output_plan_path = output_root / "sampling_plan.json"
            output_plan_path.write_text(
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
            profile=profile,
        )
        # tracking_sft 表示"数据集里存在需要 loss mask 的 masked_unknown 行"，因此
        # 训练必须挂上 --loss_scale cogtrack_tracking_sft 插件。three_state 里
        # hard_null 与 verified_update 行通过 per-message loss_scale=1.0 旁路插件，
        # 两者在同一 batch 内共存。
        dataset_info = {
            "schema_version": (
                VLT_V6_DATASET_VERSION
                if profile == "vlt_v6"
                else VISUAL_V5_DATASET_VERSION
                if profile == "visual_v5"
                else DATASET_VERSION
            ),
            "task": (
                "vlt_v6_tracking_sft"
                if profile == "vlt_v6"
                else "visual_tracking_v5_probe"
                if profile == "visual_v5"
                else "stage1_tracking_presence"
            ),
            "synthesis_profile": profile,
            "source_datasets": dataset_names,
            "source_split": "train",
            "mgit_version": args.mgit_version if "mgit" in dataset_names else None,
            "allow_missing_mgit_sequences": args.allow_missing_mgit_sequences,
            "target_status": ["present", "absent"] if args.absent_ratio > 0 else ["present"],
            "canonical_bbox_format": "norm1000_xyxy",
            "reference_mode": args.reference_mode,
            "prompt_profile": build_report.prompt_profile,
            "force_history_image": build_report.force_history_image,
            "history_layout_version": build_report.history_layout_version,
            "reference_policy": reference_policy,
            "reference_canonical_bbox_format": "norm1000_xyxy",
            "training_coordinate_protocols": {
                "qwen2_5_vl": "processor-resized absolute pixel xyxy",
                "qwen3_vl": "relative 0-to-1000 xyxy",
            },
            "coordinate_conversion_owner": "ms-swift model template",
            "qwen_vl_bbox_format": "new",
            "uses_initial_target_text": profile == "vlt_v6",
            "uses_dataset_language_where_online_safe": profile == "vlt_v6",
            "uses_semantic_memory_input": build_report.semantic_memory_input_count > 0,
            "memory_supervision": args.memory_supervision,
            # three_state 同样监督记忆，只是逐行决定该行的记忆值是否参与 loss。
            "supervises_memory": args.memory_supervision in MEMORY_SUPERVISION_LABELLED_MODES,
            # 数据集级"是否存在被 mask 的记忆值"。逐行的真值在
            # metadata.memory_supervision_state 与 messages[assistant].loss_scale。
            "memory_loss_masked": args.memory_supervision
            in {MEMORY_SUPERVISION_MASKED_NULL, MEMORY_SUPERVISION_THREE_STATE},
            "sft_supervision_profile": sft_supervision_profile,
            "data_tier": (
                "legacy_baseline"
                if profile == "legacy_stage1"
                else "state_update_sft"
                if sft_supervision_profile == "state_update_sft"
                else "tracking_sft"
                if sft_supervision_profile == SFT_SUPERVISION_TRACKING_SFT
                else (
                    "sft_probe"
                    if args.memory_supervision == MEMORY_SUPERVISION_EXPLICIT
                    else "pipeline_feasibility"
                )
            ),
            "sft_eligible": (
                profile == "legacy_stage1"
                or args.memory_supervision
                in {
                    MEMORY_SUPERVISION_EXPLICIT,
                    MEMORY_SUPERVISION_MASKED_NULL,
                    MEMORY_SUPERVISION_THREE_STATE,
                }
            ),
            "paper_full_eligible": profile == "legacy_stage1",
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
            "sampling_plan_sha256": (
                _file_sha256(output_plan_path) if sampling_plan is not None else None
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

    print(f"[CognitiveTrack] {profile} 数据构建完成")
    print(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    print(f"[训练] DATASET_ROOT={output_root}")
    for family in dict.fromkeys(args.qwen_model_families):
        print(f"[训练:{family}] TRAIN_DATA={output_root / 'ms_swift' / family / 'train.jsonl'}")
        print(f"[训练:{family}] VAL_DATA={output_root / 'ms_swift' / family / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
