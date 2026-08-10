"""轻量、确定性的文本和图像路径读取工具。

这里刻意不引入 pandas。大型 benchmark 初始化时，numpy 与 pathlib 足以完成
标注读取，且启动开销和隐式类型转换都更可控。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
_NATURAL_TOKEN = re.compile(r"(\d+)")


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    """返回适合帧文件名的自然排序键，例如 2.jpg 会排在 10.jpg 前。"""

    text = Path(value).name
    return tuple(int(token) if token.isdigit() else token.lower() for token in _NATURAL_TOKEN.split(text))


def list_image_files(directory: str | Path) -> list[str]:
    """列出目录下的图像文件并进行自然排序。

    只扫描当前目录，不递归；跟踪数据集通常每个序列都有明确的帧目录，递归
    搜索容易意外混入可视化图或缩略图。
    """

    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"帧目录不存在: {path}")
    frames = [entry for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES]
    frames.sort(key=natural_sort_key)
    if not frames:
        raise FileNotFoundError(f"帧目录中没有支持的图像文件: {path}")
    return [str(frame) for frame in frames]


def load_numeric_table(
    path: str | Path,
    *,
    dtype: np.dtype | type = np.float64,
    columns: int | None = None,
) -> np.ndarray:
    """读取逗号、制表符或空格分隔的数值标注。

    部分公开数据集的标注文件在不同镜像中使用不同分隔符。本函数按首个非空
    行判断格式，避免上层 loader 重复实现兼容逻辑。
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"标注文件不存在: {file_path}")

    first_line = ""
    with file_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                first_line = line
                break
    if not first_line:
        raise ValueError(f"标注文件为空: {file_path}")

    delimiter = "," if "," in first_line else None
    values = np.loadtxt(file_path, delimiter=delimiter, dtype=dtype, ndmin=1)
    if columns is not None:
        if values.size % columns != 0:
            raise ValueError(f"{file_path} 含 {values.size} 个数值，无法整理为 {columns} 列")
        values = values.reshape(-1, columns)
    return np.asarray(values, dtype=dtype)


def read_text(path: str | Path, *, required: bool = False) -> str | None:
    """读取 UTF-8 文本；可选文件缺失或内容为空时返回 ``None``。"""

    file_path = Path(path)
    if not file_path.is_file():
        if required:
            raise FileNotFoundError(f"文本文件不存在: {file_path}")
        return None
    text = file_path.read_text(encoding="utf-8-sig").strip()
    return text or None


def validate_parallel_lengths(name: str, expected: int, **items: Sequence[object] | np.ndarray | None) -> None:
    """检查帧、标注和状态数组是否逐帧对齐。"""

    for item_name, item in items.items():
        if item is not None and len(item) != expected:
            raise ValueError(f"序列 {name}: {item_name} 长度 {len(item)} != 帧数 {expected}")


def read_index_file(path: str | Path) -> list[int]:
    """读取一行一个索引的纯文本文件，忽略空行和 ``#`` 注释。"""

    indices: list[int] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                indices.append(int(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法整数: {line!r}") from exc
    return indices


def ensure_unique(values: Iterable[str], *, label: str) -> list[str]:
    """保留顺序地去重，并对重复项给出清晰错误。"""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{label} 中存在重复项: {value}")
        seen.add(value)
        result.append(value)
    return result
