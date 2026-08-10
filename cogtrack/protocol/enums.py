"""CognitiveTrack v4 使用的有限状态集合。

主跟踪任务只监督 ``present/absent``。``uncertain`` 保留给工程错误或未来的
选择性预测研究，但 v4 Prompt 不要求模型生成它。
"""

from enum import Enum


class StringEnum(str, Enum):
    """兼容 Python 3.8+ 的字符串枚举基类。"""

    def __str__(self) -> str:
        return self.value


class GroundTruthPresence(StringEnum):
    """数据集可提供的目标存在性真值。"""

    PRESENT = "present"
    ABSENT = "absent"


class TargetPresence(StringEnum):
    """模型对目标存在性的预测；允许证据不足时拒答。"""

    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class IdentityMatch(StringEnum):
    """当前候选与初始化目标的实例身份关系。"""

    SAME = "same"
    DIFFERENT = "different"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


class Localizability(StringEnum):
    """目标是否能够输出可靠边界框。"""

    LOCALIZABLE = "localizable"
    UNLOCALIZABLE = "unlocalizable"
    NOT_APPLICABLE = "not_applicable"


class ExecutionStatus(StringEnum):
    """一次帧级执行的工程状态，与目标是否存在完全独立。"""

    OK = "ok"
    SKIPPED = "skipped"
    IMAGE_ERROR = "image_error"
    MODEL_ERROR = "model_error"
    API_ERROR = "api_error"
    PARSE_ERROR = "parse_error"
    INTERNAL_ERROR = "internal_error"
