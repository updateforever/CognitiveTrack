"""实验 manifest 的可复现元数据收集。

本模块只读取文件哈希、Python/包版本与 Git HEAD，不导入 torch、
transformers 或模型权重，因此不会让轻量 dummy 实验意外占用显存。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    """返回文件路径、字节数和 SHA-256；缺失文件返回 ``None``。"""

    if path is None:
        return None
    file_path = Path(path).expanduser().resolve(strict=False)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _referenced_configs(tracker_config: Path | None) -> list[dict[str, Any]]:
    """递归收集 tracker YAML 引用的模型配置与 checkpoint 指纹。"""

    if tracker_config is None or not tracker_config.is_file():
        return []
    try:
        with tracker_config.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(payload, dict):
        return []

    references: list[tuple[str, str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"model_config", "checkpoint"} and isinstance(item, str):
                    references.append((str(key), item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)

    results: list[dict[str, Any]] = []
    observed: set[Path] = set()
    for kind, value in references:
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "$" in expanded:
            continue
        path = Path(expanded)
        if not path.is_absolute():
            path = tracker_config.parent / path
        path = path.resolve(strict=False)
        if path in observed:
            continue
        fingerprint = file_fingerprint(path)
        if fingerprint is not None:
            fingerprint["kind"] = kind
            results.append(fingerprint)
            observed.add(path)
    return results


def _git_directory(project_root: Path) -> Path | None:
    """向上寻找普通仓库或 worktree 的 Git 元数据目录。"""

    for root in (project_root, *project_root.parents):
        marker = root / ".git"
        if marker.is_dir():
            return marker
        if marker.is_file():
            try:
                text = marker.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text.startswith("gitdir:"):
                value = Path(text.split(":", 1)[1].strip())
                return (root / value).resolve(strict=False) if not value.is_absolute() else value
    return None


def git_head(project_root: str | Path) -> str | None:
    """不调用 shell 地解析 Git HEAD，非 Git 安装包返回 ``None``。"""

    git_dir = _git_directory(Path(project_root).expanduser().resolve(strict=False))
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None

    ref_name = head.split(":", 1)[1].strip()
    try:
        value = (git_dir / ref_name).read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    try:
        lines = (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith(("#", "^")):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref_name:
            return parts[0]
    return None


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_tree_fingerprint(project_root: str | Path) -> dict[str, Any]:
    """对可执行源码/配置做稳定树哈希，覆盖未提交研究代码。"""

    root = Path(project_root).expanduser().resolve(strict=False)
    suffixes = {".py", ".yaml", ".yml", ".toml", ".sh"}
    candidates: list[Path] = []
    for directory_name in ("cogtrack", "pytracking", "tracking", "configs", "scripts"):
        directory = root / directory_name
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in suffixes
                and "__pycache__" not in path.parts
            )
    for file_name in ("pyproject.toml", "environment.yml"):
        path = root / file_name
        if path.is_file():
            candidates.append(path)

    digest = hashlib.sha256()
    unique_paths = sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())
    for path in unique_paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"file_count": len(unique_paths), "sha256": digest.hexdigest()}


def collect_reproducibility_metadata(
    *,
    project_root: str | Path,
    tracker_config: str | Path | None,
    environment_config: str | Path | None,
) -> dict[str, Any]:
    """收集一次实验的稳定元数据，供每序列 manifest 共用。"""

    root = Path(project_root).expanduser().resolve(strict=False)
    tracker_path = (
        Path(tracker_config).expanduser().resolve(strict=False) if tracker_config is not None else None
    )
    return {
        "framework": "CognitiveTrack",
        "framework_version": _package_versions(("cognitive-track",))["cognitive-track"] or "0.1.0",
        "git_commit": git_head(root),
        "source_tree": source_tree_fingerprint(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": list(sys.argv),
        "packages": _package_versions(
            ("numpy", "opencv-python", "Pillow", "PyYAML", "torch", "transformers", "ms-swift")
        ),
        "tracker_config": file_fingerprint(tracker_path),
        "referenced_configs": _referenced_configs(tracker_path),
        "environment_config": file_fingerprint(environment_config),
    }
