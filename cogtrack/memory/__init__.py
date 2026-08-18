"""身份锚点、动态记忆库和可信更新门控。"""

from .bank import MemoryBank, MemoryBankConfig
from .records import IdentityAnchor, MemoryKind, MemoryRecord, MemorySource
from .update_policy import (
    SEMANTIC_EVENT_CONTINUED_ABSENCE,
    SEMANTIC_EVENT_CONTINUOUS_PRESENT,
    SEMANTIC_EVENT_DISAPPEARANCE,
    SEMANTIC_EVENT_REAPPEARANCE,
    GatedMemoryUpdatePolicy,
    MemoryCandidate,
    MemoryUpdateDecision,
    MemoryUpdatePolicyConfig,
)

__all__ = [
    "GatedMemoryUpdatePolicy",
    "IdentityAnchor",
    "MemoryBank",
    "MemoryBankConfig",
    "MemoryCandidate",
    "MemoryKind",
    "MemoryRecord",
    "MemorySource",
    "MemoryUpdateDecision",
    "MemoryUpdatePolicyConfig",
    "SEMANTIC_EVENT_CONTINUED_ABSENCE",
    "SEMANTIC_EVENT_CONTINUOUS_PRESENT",
    "SEMANTIC_EVENT_DISAPPEARANCE",
    "SEMANTIC_EVENT_REAPPEARANCE",
]
