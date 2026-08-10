"""CognitiveTrack 协议层异常。

协议校验、坐标转换和模型输出解析都使用明确的异常类型。调用方应把这些
异常映射为相应的 :class:`ExecutionStatus`，而不是把异常伪装成 ``absent``。
"""


class CognitiveTrackError(Exception):
    """新框架内部可预期异常的基类。"""


class ProtocolValidationError(CognitiveTrackError, ValueError):
    """结构化协议字段缺失、类型错误或字段之间语义冲突。"""


class BoundingBoxError(ProtocolValidationError):
    """边界框格式、数值或坐标范围不合法。"""


class ModelOutputParseError(CognitiveTrackError, ValueError):
    """VLM 文本不能严格解析成 CognitiveTrack v4 输出。"""
