#!/usr/bin/env python3
"""Build a self-contained mixed tracking/state-update SFT release.

The published view keeps every source row once.  The training view keeps every
tracking row once and deterministically resamples the two verified state-label
classes to the requested 90/4/6 mixture.  Sequence splits are harmonised before
sampling so that a sequence can never occur in both train and validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TRACKING_PROFILE = "tracking_sft"
STATE_PROFILE = "state_update_sft"
MASKED_UNKNOWN = "masked_unknown"
VERIFIED_UPDATE = "verified_update"
VERIFIED_HARD_NULL = "verified_hard_null"
DEFAULT_RATIOS = {
    MASKED_UNKNOWN: 0.90,
    VERIFIED_UPDATE: 0.04,
    VERIFIED_HARD_NULL: 0.06,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_no}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def seed_loss_scale_arrow_schema(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Put one explicit per-message loss scale in the first JSONL row.

    HuggingFace ``datasets`` infers nested JSON struct fields from the first
    Arrow chunk.  A mixed file can therefore fail late when many tracking rows
    (which intentionally omit ``loss_scale``) precede the first fully supervised
    state-update row.  Moving one such row to the front makes the optional
    ``messages[].loss_scale`` field part of the inferred schema without changing
    sample membership, supervision, or multiplicity.
    """

    ordered = list(rows)
    explicit_index = next(
        (
            index
            for index, row in enumerate(ordered)
            if any(
                isinstance(message, Mapping) and message.get("loss_scale") is not None
                for message in row.get("messages", [])
            )
        ),
        None,
    )
    if explicit_index in {None, 0}:
        return ordered
    return [ordered[explicit_index], *ordered[:explicit_index], *ordered[explicit_index + 1 :]]


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"row {row.get('id')!r} has no metadata object")
    return metadata


def _profile(row: Mapping[str, Any]) -> str:
    value = str(_metadata(row).get("sft_supervision_profile", ""))
    if value not in {TRACKING_PROFILE, STATE_PROFILE}:
        raise ValueError(f"unsupported SFT profile {value!r} in row {row.get('id')!r}")
    return value


def _state(row: Mapping[str, Any]) -> str:
    value = str(_metadata(row).get("memory_supervision_state", ""))
    expected = {
        TRACKING_PROFILE: {MASKED_UNKNOWN},
        STATE_PROFILE: {VERIFIED_UPDATE, VERIFIED_HARD_NULL},
    }[_profile(row)]
    if value not in expected:
        raise ValueError(
            f"profile/state mismatch for {row.get('id')!r}: {_profile(row)!r}/{value!r}"
        )
    return value


def _sequence_key(row: Mapping[str, Any]) -> str:
    metadata = _metadata(row)
    dataset = str(metadata.get("source_dataset") or metadata.get("dataset") or "").lower()
    sequence = str(metadata.get("source_sequence") or metadata.get("sequence") or "")
    if not dataset or not sequence:
        raise ValueError(f"row {row.get('id')!r} has no dataset/sequence")
    return f"{dataset}::{sequence}"


def _load_release(root: Path, source_name: str) -> list[tuple[dict[str, Any], str, str]]:
    result: list[tuple[dict[str, Any], str, str]] = []
    for split in ("train", "val"):
        path = root / "ms_swift" / "qwen3_vl" / f"{split}.jsonl"
        for row in _read_jsonl(path):
            expected = TRACKING_PROFILE if source_name == "tracking" else STATE_PROFILE
            if _profile(row) != expected:
                raise ValueError(f"{path} contains {_profile(row)!r}, expected {expected!r}")
            _state(row)
            result.append((row, split, source_name))
    return result


def harmonise_splits(
    tracking: Sequence[tuple[dict[str, Any], str, str]],
    state: Sequence[tuple[dict[str, Any], str, str]],
) -> tuple[dict[str, list[tuple[dict[str, Any], str]]], dict[str, Any]]:
    """Use tracking splits as canonical for shared sequences."""

    tracking_splits: dict[str, str] = {}
    state_splits: dict[str, str] = {}
    for rows, mapping in ((tracking, tracking_splits), (state, state_splits)):
        for row, split, _source in rows:
            key = _sequence_key(row)
            previous = mapping.setdefault(key, split)
            if previous != split:
                raise ValueError(f"source release leaks sequence across splits: {key}")

    shared = set(tracking_splits).intersection(state_splits)
    conflicts = {key for key in shared if tracking_splits[key] != state_splits[key]}
    output: dict[str, list[tuple[dict[str, Any], str]]] = {"train": [], "val": []}
    moved_rows = 0
    for row, split, source in [*tracking, *state]:
        key = _sequence_key(row)
        resolved = tracking_splits.get(key, split)
        if resolved != split:
            moved_rows += 1
        cloned = copy.deepcopy(row)
        metadata = dict(_metadata(cloned))
        metadata.update(
            {
                "mixed_release_source": source,
                "mixed_original_split": split,
                "mixed_resolved_split": resolved,
                "mixed_split_policy": "tracking_canonical_then_state_v1",
            }
        )
        cloned["metadata"] = metadata
        output[resolved].append((cloned, source))

    train_keys = {_sequence_key(row) for row, _source in output["train"]}
    val_keys = {_sequence_key(row) for row, _source in output["val"]}
    overlap = train_keys.intersection(val_keys)
    if overlap:
        raise AssertionError(f"harmonised split still leaks sequences: {sorted(overlap)[:3]}")
    report = {
        "policy": "tracking_canonical_then_state_v1",
        "tracking_sequences": len(tracking_splits),
        "state_sequences": len(state_splits),
        "shared_sequences": len(shared),
        "conflicting_sequences": len(conflicts),
        "conflicting_sequence_keys": sorted(conflicts),
        "moved_state_rows": moved_rows,
        "train_sequences": len(train_keys),
        "val_sequences": len(val_keys),
        "train_val_sequence_overlap": 0,
    }
    return output, report


def _balanced_resample(
    rows: Sequence[dict[str, Any]], target: int, rng: random.Random
) -> list[dict[str, Any]]:
    if not rows or target <= 0:
        raise ValueError("resampling requires non-empty rows and a positive target")
    sampled: list[dict[str, Any]] = []
    while len(sampled) < target:
        cycle = list(rows)
        rng.shuffle(cycle)
        sampled.extend(cycle[: target - len(sampled)])
    return sampled


def build_weighted_train(
    unique_train: Sequence[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_train:
        groups[_state(row)].append(row)
    if set(groups) != set(DEFAULT_RATIOS):
        raise ValueError(f"training rows do not cover all supervision states: {sorted(groups)}")

    tracking_count = len(groups[MASKED_UNKNOWN])
    total = round(tracking_count / DEFAULT_RATIOS[MASKED_UNKNOWN])
    update_target = round(total * DEFAULT_RATIOS[VERIFIED_UPDATE])
    null_target = total - tracking_count - update_target
    targets = {
        MASKED_UNKNOWN: tracking_count,
        VERIFIED_UPDATE: update_target,
        VERIFIED_HARD_NULL: null_target,
    }

    rng = random.Random(seed)
    selected: list[tuple[dict[str, Any], str]] = [
        (row, MASKED_UNKNOWN) for row in groups[MASKED_UNKNOWN]
    ]
    for state in (VERIFIED_UPDATE, VERIFIED_HARD_NULL):
        selected.extend(
            (row, state) for row in _balanced_resample(groups[state], targets[state], rng)
        )
    rng.shuffle(selected)

    occurrences: Counter[tuple[str, str]] = Counter()
    output: list[dict[str, Any]] = []
    for row, state in selected:
        cloned = copy.deepcopy(row)
        source = str(_metadata(cloned)["mixed_release_source"])
        occurrence_key = (source, str(cloned.get("id", "")))
        occurrences[occurrence_key] += 1
        metadata = dict(_metadata(cloned))
        metadata.update(
            {
                "mixed_training_view": "weighted_90_4_6_v1",
                "mixed_sampling_category": state,
                "mixed_sampling_occurrence": occurrences[occurrence_key],
            }
        )
        cloned["metadata"] = metadata
        output.append(cloned)

    counts = Counter(_state(row) for row in output)
    report = {
        "policy": "weighted_90_4_6_v1",
        "seed": seed,
        "target_ratios": DEFAULT_RATIOS,
        "samples": len(output),
        "counts": dict(sorted(counts.items())),
        "actual_ratios": {
            key: counts[key] / len(output) for key in sorted(DEFAULT_RATIOS)
        },
        "unique_source_rows": {
            key: len(groups[key]) for key in sorted(DEFAULT_RATIOS)
        },
        "maximum_row_occurrence": max(occurrences.values()),
    }
    return output, report


def _rewrite_images_and_link(
    rows: Iterable[tuple[dict[str, Any], str]],
    *,
    source_roots: Mapping[str, Path],
    output_root: Path,
) -> tuple[int, int]:
    linked: set[str] = set()
    total_bytes = 0
    for row, source in rows:
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"row {row.get('id')!r} must reference exactly three images")
        rewritten: list[str] = []
        for raw in images:
            relative = Path(str(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe image path in {row.get('id')!r}: {relative}")
            source_path = source_roots[source] / relative
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            suffix = relative.parts[1:] if relative.parts[:1] == ("images",) else relative.parts
            destination_relative = Path("images") / source / Path(*suffix)
            destination = output_root / destination_relative
            key = destination_relative.as_posix()
            if key not in linked:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source_path, destination)
                linked.add(key)
                total_bytes += source_path.stat().st_size
            rewritten.append(destination_relative.as_posix())
        row["images"] = rewritten
    return len(linked), total_bytes


def _link_preview(preview_root: Path, output_root: Path) -> int:
    source_assets = preview_root / "assets"
    count = 0
    for source in sorted(source_assets.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_assets)
        destination = output_root / "assets" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        count += 1
    return count


def _release_readme(report: Mapping[str, Any], preview_readme: str) -> str:
    unique = report["unique_view"]
    weighted = report["weighted_training_view"]
    prefix = f"""# CognitiveTrack VLT-v6.4 Mixed SFT Dataset

这是可直接发布和训练的自包含 Qwen3-VL-4B SFT release。输入固定为三张图片，输出为：

```json
{{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}}
```

## 文件与规模

- `train.jsonl`：正式 90/4/6 加权训练视图，共 {weighted['samples']:,} 行；
- `train_unique.jsonl`：训练划分的全部源行各保留一次，共 {unique['train_samples']:,} 行；
- `val.jsonl`：按完整序列隔离的验证视图，共 {unique['val_samples']:,} 行；
- `images/`：两类 release 的完整三图素材，所有 JSONL 路径均相对于本目录；
- `assets/`：下方 README 案例的展示图片；
- `manifest.json`、`build_report.json`、`checksums.sha256`：来源、采样和完整性审计。

正式训练比例是 `90% masked_unknown tracking / 4% verified update / 6% verified hard-null`。
`tracking_sft` 只屏蔽未知 `memory_update` 的值；`state_update_sft` 的更新与 hard-null 全监督。
`train_unique.jsonl` 不做采样重复，但按设计保留两种数据中共享视觉 case 的不同监督行。

## ms-swift 训练入口

```bash
export DATASET_ROOT=/path/to/cogtrack_v640_mixed_sft_full_v1
export MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
export TRAIN_DATA="$DATASET_ROOT/train.jsonl"
export VAL_DATA="$DATASET_ROOT/val.jsonl"
export SFT_SUPERVISION_PROFILE=mixed_sft
export TUNER_TYPE=full
export FREEZE_VIT=true FREEZE_LLM=false FREEZE_ALIGNER=false
export DEEPSPEED=zero2

bash scripts/train_qwen3vl_4b_tracking_sft.sh
```

训练使用 `QWENVL_BBOX_FORMAT=new`，Qwen3-VL bbox 是 Image 3 上 `[0,1000] xyxy`。

## 发布许可说明

代码与 CognitiveTrack 生成的标注遵循项目许可；`images/` 中的原始视频帧仍分别受
LaSOT、TNL2K 和 MGIT 的上游许可与再分发条款约束。公开上传完整图片前必须逐项确认
这些数据集允许二次分发；若不允许，应发布 JSONL/manifest 和构建脚本，并要求用户从
官方来源准备原始数据，而不能因为本目录技术上自包含就默认拥有图像再分发权。

---

"""
    preview = preview_readme
    if preview.startswith("# "):
        preview = "## 数据 Case 可视化\n" + "\n".join(preview.splitlines()[1:]).lstrip()
    return prefix + preview.rstrip() + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_root: Path, *, workers: int = 16) -> int:
    if workers <= 0:
        raise ValueError("checksum workers must be positive")
    checksum_path = output_root / "checksums.sha256"
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        checksum_path.open("w", encoding="utf-8") as handle,
    ):
        for path, digest in zip(files, executor.map(_sha256, files), strict=True):
            handle.write(f"{digest}  {path.relative_to(output_root).as_posix()}\n")
    return len(files)


def build_release(
    *,
    tracking_root: Path,
    state_root: Path,
    preview_root: Path,
    output_root: Path,
    seed: int,
    include_checksums: bool = True,
    checksum_workers: int = 16,
) -> dict[str, Any]:
    roots = {
        "tracking": tracking_root.resolve(),
        "state_update": state_root.resolve(),
    }
    for root in [*roots.values(), preview_root.resolve()]:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)

    tracking = _load_release(roots["tracking"], "tracking")
    state = _load_release(roots["state_update"], "state_update")
    splits, split_report = harmonise_splits(tracking, state)

    # Images are rewritten once on the unique view; weighted rows are cloned afterwards.
    all_unique_pairs = [*splits["train"], *splits["val"]]
    image_count, image_bytes = _rewrite_images_and_link(
        all_unique_pairs, source_roots=roots, output_root=output_root
    )
    unique_train = seed_loss_scale_arrow_schema(
        [row for row, _source in splits["train"]]
    )
    unique_val = seed_loss_scale_arrow_schema([row for row, _source in splits["val"]])
    weighted_train, sampling_report = build_weighted_train(unique_train, seed=seed)
    weighted_train = seed_loss_scale_arrow_schema(weighted_train)

    _write_jsonl(output_root / "train.jsonl", weighted_train)
    _write_jsonl(output_root / "train_unique.jsonl", unique_train)
    _write_jsonl(output_root / "val.jsonl", unique_val)
    nested = output_root / "ms_swift" / "qwen3_vl"
    nested.mkdir(parents=True)
    for name in ("train.jsonl", "train_unique.jsonl", "val.jsonl"):
        os.link(output_root / name, nested / name)

    unique_counts = Counter(_state(row) for row in [*unique_train, *unique_val])
    report: dict[str, Any] = {
        "schema_version": "cogtrack.mixed_sft_release.v1",
        "release_name": output_root.name,
        "prompt_version": "6.4.0",
        "model_family": "qwen3_vl",
        "bbox_protocol": "norm1000_xyxy",
        "qwen_vl_bbox_format": "new",
        "self_contained": True,
        "seed": seed,
        "source_releases": {
            "tracking_sft": tracking_root.name,
            "state_update_sft": state_root.name,
        },
        "split": split_report,
        "unique_view": {
            "train_samples": len(unique_train),
            "val_samples": len(unique_val),
            "total_samples": len(unique_train) + len(unique_val),
            "counts": dict(sorted(unique_counts.items())),
        },
        "weighted_training_view": sampling_report,
        "images": {
            "files": image_count,
            "logical_bytes": image_bytes,
        },
    }
    preview_count = _link_preview(preview_root.resolve(), output_root)
    report["readme_assets"] = preview_count
    preview_readme = (preview_root / "README.md").read_text(encoding="utf-8")
    (output_root / "README.md").write_text(
        _release_readme(report, preview_readme), encoding="utf-8"
    )
    _write_json(
        output_root / "dataset_info.json",
        {
            "schema_version": "cogtrack.mixed_sft_release.v1",
            "dataset_name": output_root.name,
            "train": "train.jsonl",
            "train_unique": "train_unique.jsonl",
            "val": "val.jsonl",
            "dataset_root": ".",
            "sft_supervision_profile": "mixed_sft",
            "model_family": "qwen3_vl",
        },
    )
    if include_checksums:
        report["checksum_files"] = sum(
            1 for path in output_root.rglob("*") if path.is_file()
        ) + 2  # manifest.json and build_report.json are written immediately below.
    _write_json(output_root / "manifest.json", report)
    _write_json(output_root / "build_report.json", report)
    if include_checksums:
        actual_checksum_files = write_checksums(output_root, workers=checksum_workers)
        if actual_checksum_files != report["checksum_files"]:
            raise AssertionError(
                f"checksum inventory changed: {actual_checksum_files} != "
                f"{report['checksum_files']}"
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--checksum-workers", type=int, default=16)
    args = parser.parse_args()
    try:
        report = build_release(
            tracking_root=args.tracking_root,
            state_root=args.state_root,
            preview_root=args.preview_root,
            output_root=args.output_root,
            seed=args.seed,
            include_checksums=not args.skip_checksums,
            checksum_workers=args.checksum_workers,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
