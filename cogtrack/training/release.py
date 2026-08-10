"""把自包含跟踪数据集打包为 Hugging Face/ModelScope 友好的发布目录。

发布包保留 ms-swift JSONL 和审计元数据，但将大量小图片按完整序列打入独立 TAR
分卷。每个序列只进入一个分卷，因此参考图和当前帧不会被拆散。TAR 不再压缩 JPEG，
避免无收益的 CPU 开销；SHA-256 用于跨服务器传输校验。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReleaseShard:
    """单个图片分卷的可审计信息。"""

    path: str
    size_bytes: int
    sha256: str
    sequence_count: int
    file_count: int


@dataclass(frozen=True)
class DatasetReleaseReport:
    """发布目录的结构化报告。"""

    schema_version: str
    source_root: str
    release_root: str
    max_shard_bytes: int
    image_sequence_count: int
    image_file_count: int
    image_size_bytes: int
    shards: tuple[ReleaseShard, ...]
    metadata_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shards"] = [asdict(shard) for shard in self.shards]
        payload["metadata_files"] = list(self.metadata_files)
        return payload


_REQUIRED_FILES = (
    "dataset_info.json",
    "build_report.json",
    "source_samples.jsonl",
    "ms_swift/dataset_info.json",
    "ms_swift/qwen2_5_vl/train.jsonl",
    "ms_swift/qwen2_5_vl/val.jsonl",
    "ms_swift/qwen2_5_vl/dataset_info.json",
    "ms_swift/qwen2_5_vl/validation_train.json",
    "ms_swift/qwen2_5_vl/validation_val.json",
    "ms_swift/qwen3_vl/train.jsonl",
    "ms_swift/qwen3_vl/val.jsonl",
    "ms_swift/qwen3_vl/dataset_info.json",
    "ms_swift/qwen3_vl/validation_train.json",
    "ms_swift/qwen3_vl/validation_val.json",
)
_OPTIONAL_FILES = ("sampling_plan.json",)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_directories(images_root: Path) -> list[Path]:
    directories = sorted(
        sequence_dir
        for dataset_dir in images_root.iterdir()
        if dataset_dir.is_dir()
        for sequence_dir in dataset_dir.iterdir()
        if sequence_dir.is_dir()
    )
    if not directories:
        raise ValueError(f"images 下没有 dataset/sequence 目录: {images_root}")
    return directories


def _directory_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _group_sequences(
    sequence_dirs: Iterable[Path],
    *,
    max_shard_bytes: int,
) -> list[list[tuple[Path, list[Path], int]]]:
    groups: list[list[tuple[Path, list[Path], int]]] = []
    current: list[tuple[Path, list[Path], int]] = []
    current_size = 0
    for directory in sequence_dirs:
        files = _directory_files(directory)
        if not files:
            continue
        size = sum(path.stat().st_size for path in files)
        if current and current_size + size > max_shard_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append((directory, files, size))
        current_size += size
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("没有可打包的图片文件")
    return groups


def _add_reproducible_file(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    """清除机器相关属主和时间戳，使相同输入产生稳定 TAR。"""

    info = archive.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def _write_shard(
    path: Path,
    group: list[tuple[Path, list[Path], int]],
    *,
    source_root: Path,
) -> ReleaseShard:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for _, files, _ in group:
            for file_path in files:
                _add_reproducible_file(
                    archive,
                    file_path,
                    file_path.relative_to(source_root).as_posix(),
                )
    return ReleaseShard(
        path=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        sequence_count=len(group),
        file_count=sum(len(files) for _, files, _ in group),
    )


def _dataset_card(dataset_info: dict[str, Any], release: DatasetReleaseReport) -> str:
    splits = dataset_info.get("splits", {})
    case_count = splits.get("case_count", dataset_info.get("build", {}).get("sample_count", "unknown"))
    sources = ", ".join(dataset_info.get("source_datasets", [])) or "unknown"
    return f"""---
language:
- en
task_categories:
- image-to-text
tags:
- visual-object-tracking
- multimodal
- qwen-vl
- ms-swift
pretty_name: CognitiveTrack Stage-1 Tracking Presence
size_categories:
- 10K<n<100K
license: other
---

# CognitiveTrack Stage-1 Tracking Presence

Portable multimodal SFT data derived from the official **training splits** of {sources}.
Each case contains an unmodified full initialization frame, its target coordinates in the prompt,
and an unannotated current full frame. Coordinates use the official model-family convention:
Qwen2.5-VL uses absolute coordinates on the processor-resized image, while Qwen3-VL uses relative
0-to-1000 coordinates. Both views are generated from the same canonical annotation through
ms-swift's `<bbox> + objects.bbox + image_id` grounding interface.

The assistant output has exactly two semantic fields: `target_status` and the family-specific
bbox field. Real same-sequence present/absent cases are sampled at approximately 70:30; no
cross-sequence or synthetic negatives are used.

## Summary

- Cases: {case_count}
- Sources: {sources}
- Images: {release.image_file_count}
- Image shards: {len(release.shards)}
- Image paths: relative and portable
- Training format: ms-swift multimodal `messages + images + objects` JSONL
- Grounding format: `QWENVL_BBOX_FORMAT=new`
- Memory/confidence/reasoning supervision: none

## Files

- `ms_swift/qwen2_5_vl/{{train,val}}.jsonl`: Qwen2.5-VL official absolute-pixel view.
- `ms_swift/qwen3_vl/{{train,val}}.jsonl`: Qwen3-VL official relative-1000 view.
- `image_shards/images-*.tar`: sequence-aligned uncompressed JPEG TAR shards.
- `source_samples.jsonl`, `sampling_plan.json`: provenance and sampling audit.
- `dataset_info.json`, `build_report.json`, `release_manifest.json`: version and statistics.
- `SHA256SUMS`: transfer integrity checks.

## Prepare on another server

From the downloaded dataset repository root:

```bash
sha256sum -c SHA256SUMS
for shard in image_shards/images-*.tar; do tar -xf "$shard"; done
```

This creates `images/...` beside the JSONL files. Then run:

```bash
DATASET_ROOT=$(pwd) \\
TRAIN_DATA=$(pwd)/ms_swift/qwen2_5_vl/train.jsonl \\
VAL_DATA=$(pwd)/ms_swift/qwen2_5_vl/val.jsonl \\
MODEL_PATH=/path/to/Qwen2.5-VL \\
bash /path/to/CognitiveTrack/scripts/train_sft.sh
```

For Qwen3-VL, change all three paths to the Qwen3-VL model and the `qwen3_vl` data directory.
Never train one model family with the other family's JSONL view.

## Redistribution notice

This package contains frames derived from upstream tracking datasets. Before publishing a
**public** Hugging Face or ModelScope repository, verify and comply with the redistribution
terms of every upstream dataset. A private dataset repository does not remove those obligations.
The generated annotations and CognitiveTrack tooling do not relicense upstream images.
"""


def _copy_metadata(source_root: Path, release_root: Path) -> tuple[str, ...]:
    copied: list[str] = []
    for relative in (*_REQUIRED_FILES, *_OPTIONAL_FILES):
        source = source_root / relative
        if not source.is_file():
            if relative in _REQUIRED_FILES:
                raise FileNotFoundError(f"发布包缺少必需文件: {source}")
            continue
        target = release_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    return tuple(copied)


def package_dataset_release(
    source_root: str | Path,
    release_root: str | Path,
    *,
    max_shard_bytes: int = 2 * 1024**3,
    overwrite: bool = False,
) -> DatasetReleaseReport:
    """创建不含绝对路径、可校验、可分卷上传的数据发布目录。"""

    source = Path(source_root).expanduser().resolve()
    release = Path(release_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"源数据目录不存在: {source}")
    if release == source or source in release.parents:
        raise ValueError("release_root 不能等于或位于 source_root 内部")
    if isinstance(max_shard_bytes, bool) or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes 必须为正整数")
    if release.exists() and any(release.iterdir()) and not overwrite:
        raise FileExistsError(f"发布目录非空；如需覆盖请启用 overwrite: {release}")
    release.mkdir(parents=True, exist_ok=True)

    metadata_files = _copy_metadata(source, release)
    sequence_dirs = _sequence_directories(source / "images")
    groups = _group_sequences(sequence_dirs, max_shard_bytes=max_shard_bytes)
    shards: list[ReleaseShard] = []
    shard_root = release / "image_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups):
        shard_path = shard_root / f"images-{index:05d}-of-{len(groups):05d}.tar"
        if shard_path.exists() and not overwrite:
            raise FileExistsError(f"图片分卷已存在: {shard_path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{shard_path.name}.",
            suffix=".partial",
            dir=shard_root,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shard = _write_shard(temporary, group, source_root=source)
            os.replace(temporary, shard_path)
            shards.append(
                ReleaseShard(
                    path=f"image_shards/{shard_path.name}",
                    size_bytes=shard_path.stat().st_size,
                    sha256=shard.sha256,
                    sequence_count=shard.sequence_count,
                    file_count=shard.file_count,
                )
            )
        finally:
            temporary.unlink(missing_ok=True)

    image_files = sum(shard.file_count for shard in shards)
    image_bytes = sum(size for group in groups for _, _, size in group)
    report = DatasetReleaseReport(
        schema_version="cogtrack.dataset.release.v1",
        source_root=str(source),
        release_root=str(release),
        max_shard_bytes=max_shard_bytes,
        image_sequence_count=len(sequence_dirs),
        image_file_count=image_files,
        image_size_bytes=image_bytes,
        shards=tuple(shards),
        metadata_files=metadata_files,
    )
    manifest_path = release / "release_manifest.json"
    portable_manifest = report.to_dict()
    # 构建机绝对路径只用于 CLI 回显，不进入待上传发布包。
    portable_manifest["source_root"] = "<omitted-build-host-path>"
    portable_manifest["release_root"] = "."
    manifest_path.write_text(
        json.dumps(portable_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    dataset_info = json.loads((release / "dataset_info.json").read_text(encoding="utf-8"))
    (release / "README.md").write_text(_dataset_card(dataset_info, report), encoding="utf-8")

    checksum_files = [
        *(release / relative for relative in metadata_files),
        *(release / shard.path for shard in shards),
        manifest_path,
        release / "README.md",
    ]
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(release).as_posix()}"
        for path in sorted(checksum_files)
    ]
    (release / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return report


__all__ = ["DatasetReleaseReport", "ReleaseShard", "package_dataset_release"]
