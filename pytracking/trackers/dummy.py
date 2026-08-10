"""用于验证数据—runner—结果闭环的零依赖参考 tracker。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTracker


class DummyTracker(BaseTracker):
    """始终返回初始化框，不读取 GT，也不执行模型推理。

    它不是有意义的算法 baseline，仅用于安装检查、CI 和新数据集 loader 调试。
    """

    def initialize(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        del image
        bbox = info.get("init_bbox")
        if bbox is None:
            raise ValueError("DummyTracker 需要 init_bbox")
        self.state = [float(value) for value in bbox]
        return {"target_bbox": self.state.copy(), "diagnostics": {"backend": "dummy"}}

    def track(self, image: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        del image
        return {
            "target_bbox": self.state.copy(),
            "diagnostics": {
                "backend": "dummy",
                "is_observation_frame": bool(info.get("is_observation_frame", True)),
            },
        }


def get_tracker_class() -> type[DummyTracker]:
    """保持 pytracking 的动态构建约定。"""

    return DummyTracker
