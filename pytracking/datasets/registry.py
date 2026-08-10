"""数据集注册与延迟导入。

注册表只包含公开跟踪数据，确保新框架不会隐式加载其他研究子项目。
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pytracking.evaluation.data import BaseDataset, SequenceList
from pytracking.evaluation.environment import EnvironmentSettings, load_environment


@dataclass(frozen=True)
class DatasetSpec:
    module: str
    class_name: str
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)


_MGIT = "pytracking.datasets.mgit"

_DATASETS: dict[str, DatasetSpec] = {
    "cognitivebench": DatasetSpec("pytracking.datasets.cognitivebench", "CognitiveBenchDataset", {"split": "test"}),
    "lasot": DatasetSpec("pytracking.datasets.lasot", "LaSOTDataset", {"split": "test"}),
    "tnl2k": DatasetSpec("pytracking.datasets.tnl2k", "TNL2KDataset", {"split": "test"}),
    # MGIT / VideoCube。保留旧仓库的 videocube_* 名称，避免既有实验命令失效。
    "mgit": DatasetSpec(_MGIT, "MGITDataset", {"split": "test", "version": "full"}),
    "mgit_test": DatasetSpec(_MGIT, "MGITDataset", {"split": "test", "version": "full"}),
    "mgit_val": DatasetSpec(_MGIT, "MGITDataset", {"split": "val", "version": "full"}),
    "videocube_test": DatasetSpec(_MGIT, "MGITDataset", {"split": "test", "version": "full"}),
    "videocube_val": DatasetSpec(_MGIT, "MGITDataset", {"split": "val", "version": "full"}),
    "videocube_test_tiny": DatasetSpec(_MGIT, "MGITDataset", {"split": "test", "version": "tiny"}),
    "videocube_val_tiny": DatasetSpec(_MGIT, "MGITDataset", {"split": "val", "version": "tiny"}),
}


def list_datasets() -> tuple[str, ...]:
    """返回稳定排序的数据集注册名。"""

    return tuple(sorted(_DATASETS))


def load_dataset(
    name: str,
    *,
    environment: EnvironmentSettings | None = None,
    sequence_names: Sequence[str] | None = None,
    limit: int | None = None,
    **kwargs: Any,
) -> SequenceList:
    """按注册名加载单个数据集并返回标准 ``SequenceList``。"""

    return SequenceList(
        iter_dataset(
            name,
            environment=environment,
            sequence_names=sequence_names,
            limit=limit,
            **kwargs,
        )
    )


def iter_dataset(
    name: str,
    *,
    environment: EnvironmentSettings | None = None,
    sequence_names: Sequence[str] | None = None,
    limit: int | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """逐序列构造数据，供大规模训练数据合成时控制内存占用。

    ``load_dataset`` 为评测兼容性仍返回完整 ``SequenceList``；本接口只在消费到
    某个序列时展开其全部帧路径，因此适合 LaSOT/TNL2K/MGIT 训练集。
    """

    key = name.lower()
    spec = _DATASETS.get(key)
    if spec is None:
        raise KeyError(f"未知数据集 {name!r}；可选: {', '.join(list_datasets())}")
    module = importlib.import_module(spec.module)
    dataset_class = getattr(module, spec.class_name)
    options = {**spec.default_kwargs, **kwargs}
    dataset: BaseDataset = dataset_class(environment=environment or load_environment(), **options)
    if sequence_names is not None:
        selected_names = list(sequence_names)
        if limit is not None:
            if limit < 0:
                raise ValueError("dataset limit 不能为负数")
            selected_names = selected_names[:limit]
        yield from (dataset.get_sequence(sequence_name) for sequence_name in selected_names)
        return
    if limit is not None:
        if limit < 0:
            raise ValueError("dataset limit 不能为负数")
        available_names = getattr(dataset, "sequence_names", None)
        if available_names is None:
            yield from dataset.get_sequence_list()[:limit]
            return
        yield from (dataset.get_sequence(sequence_name) for sequence_name in available_names[:limit])
        return
    available_names = getattr(dataset, "sequence_names", None)
    if available_names is None:
        yield from dataset.get_sequence_list()
        return
    yield from (dataset.get_sequence(sequence_name) for sequence_name in available_names)


def get_dataset(
    *names: str,
    environment: EnvironmentSettings | None = None,
    dataset_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> SequenceList:
    """合并多个数据集；各数据集可通过 ``dataset_kwargs`` 单独传参。"""

    result = SequenceList()
    shared_environment = environment or load_environment()
    for name in names:
        kwargs = dict((dataset_kwargs or {}).get(name, {}))
        result.extend(load_dataset(name, environment=shared_environment, **kwargs))
    return result
