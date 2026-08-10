"""CognitiveTrack 的独立 pytracking 风格运行框架。

本包只提供跟踪任务通用的底层能力：数据集、跟踪器生命周期、逐帧运行与
结果落盘。模型、认知记忆和训练代码位于更高层模块，不能反向耦合到这里。
"""

from .evaluation.data import BaseDataset, Sequence, SequenceList
from .trackers.base import BaseTracker

__all__ = ["BaseDataset", "BaseTracker", "Sequence", "SequenceList"]
