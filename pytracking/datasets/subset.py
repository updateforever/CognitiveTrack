"""读取冻结的 benchmark 序列子集。

子集文件只保存完整序列名，不复制图片、标注或关键帧。这样 Tiny 与 Full 共享同一
份冻结真值和 loader，唯一差异是参与评测的序列集合。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def read_sequence_subset(path: str | Path) -> tuple[str, ...]:
    """读取每行一个序列名的 UTF-8 文本，并执行严格完整性检查。"""

    subset_path = Path(path).expanduser().resolve()
    if not subset_path.is_file():
        raise FileNotFoundError(f"序列子集文件不存在: {subset_path}")

    names: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        subset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        name = raw_line.strip()
        if not name or name.startswith("#"):
            continue
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(
                f"序列子集 {subset_path}:{line_number} 包含不安全名称: {name!r}"
            )
        if name in seen:
            raise ValueError(
                f"序列子集 {subset_path}:{line_number} 包含重复名称: {name!r}"
            )
        seen.add(name)
        names.append(name)

    if not names:
        raise ValueError(f"序列子集为空: {subset_path}")
    return tuple(names)


def sequence_subset_from_config(
    payload: Mapping[str, Any],
    config_path: str | Path,
) -> tuple[str, ...] | None:
    """解析数据集 YAML 中的 ``sequences`` 或 ``sequences_file``。

    相对 ``sequences_file`` 始终以其数据集 YAML 所在目录为基准，避免命令从不同
    工作目录启动时选到不同子集。
    """

    inline = payload.get("sequences")
    file_value = payload.get("sequences_file")
    if inline is not None and file_value is not None:
        raise ValueError("dataset config 的 sequences 与 sequences_file 不能同时设置")
    if inline is not None:
        if isinstance(inline, (str, bytes)) or not isinstance(inline, (list, tuple)):
            raise TypeError("dataset config 的 sequences 必须是序列名列表")
        names = tuple(str(name).strip() for name in inline)
        if not names or any(not name for name in names):
            raise ValueError("dataset config 的 sequences 不能为空或包含空名称")
        if len(names) != len(set(names)):
            raise ValueError("dataset config 的 sequences 包含重复名称")
        return names
    if file_value is None:
        return None

    owner = Path(config_path).expanduser().resolve()
    subset_path = Path(str(file_value)).expanduser()
    if not subset_path.is_absolute():
        subset_path = owner.parent / subset_path
    return read_sequence_subset(subset_path)
