"""单次强教师状态标签的解析和确定性质量门。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from cogtrack.prompts.state_api_teacher import STATE_API_CHANGE_ELEMENTS
from cogtrack.training.state_teacher_labels import (
    is_vacuous_update,
    normalize_state_sentence,
)

API_ANNOTATION_POLICY = "single_pass_frontier_api_v1"
API_LABEL_SOURCE = "frontier_api_teacher_v1"
API_HARD_NULL_SOURCE = "frontier_api_teacher_hard_null_v1"

_DECISIONS = frozenset({"update", "keep", "uncertain"})
_EVIDENCE = frozenset({"sufficient", "insufficient"})


@dataclass(frozen=True)
class ApiTeacherDecision:
    decision: str
    changed_elements: tuple[str, ...]
    memory_update: str | None
    confidence: float
    evidence_sufficiency: str
    significant_change: bool
    identity_consistent: bool
    standalone_complete: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(raw: Any) -> Mapping[str, Any] | None:
    """容忍围栏和少量包裹文本，但拒绝多对象猜测。"""

    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, Mapping) else None


def _strict_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def parse_api_teacher_response(raw: Any) -> tuple[ApiTeacherDecision | None, str]:
    """解析 API 教师响应；返回 ``(decision, rejection_reason)``。"""

    payload = extract_json_object(raw)
    if payload is None:
        return None, "unparseable_json"
    required = {
        "decision",
        "changed_elements",
        "memory_update",
        "confidence",
        "evidence_sufficiency",
        "significant_change",
        "identity_consistent",
        "standalone_complete",
        "evidence",
    }
    if set(payload) != required:
        return None, "schema_keys_mismatch"
    decision = str(payload.get("decision") or "").strip().lower()
    evidence_sufficiency = str(payload.get("evidence_sufficiency") or "").strip().lower()
    if decision not in _DECISIONS:
        return None, "invalid_decision"
    raw_elements = payload.get("changed_elements")
    if not isinstance(raw_elements, list) or any(not isinstance(value, str) for value in raw_elements):
        return None, "invalid_changed_elements"
    changed_elements = tuple(dict.fromkeys(value.strip().lower() for value in raw_elements))
    if any(value not in STATE_API_CHANGE_ELEMENTS for value in changed_elements):
        return None, "invalid_changed_elements"
    if evidence_sufficiency not in _EVIDENCE:
        return None, "invalid_evidence_sufficiency"
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None, "invalid_confidence"
    if not 0.0 <= confidence <= 1.0:
        return None, "invalid_confidence"
    significant = _strict_bool(payload, "significant_change")
    identity = _strict_bool(payload, "identity_consistent")
    standalone = _strict_bool(payload, "standalone_complete")
    if significant is None or identity is None or standalone is None:
        return None, "non_boolean_audit_field"
    evidence = normalize_state_sentence(str(payload.get("evidence") or ""))
    if not evidence:
        return None, "empty_evidence"
    if not evidence.lower().startswith("image 3 shows"):
        return None, "invalid_current_evidence"
    raw_update = payload.get("memory_update")
    if raw_update is not None and not isinstance(raw_update, str):
        return None, "invalid_memory_update_type"
    update = normalize_state_sentence(raw_update or "") or None

    if decision == "uncertain":
        if update is not None or changed_elements or evidence_sufficiency != "insufficient":
            return None, "inconsistent_uncertain"
    elif decision == "keep":
        if update is not None or changed_elements:
            return None, "inconsistent_keep"
        if evidence_sufficiency != "sufficient" or significant:
            return None, "unverified_keep"
    else:
        if update is None or not changed_elements:
            return None, "inconsistent_update"
        if evidence_sufficiency != "sufficient" or not significant or not identity or not standalone:
            return None, "unverified_update"
        words = update.split()
        if len(words) < 5 or len(words) > 30 or len(update) > 256:
            return None, "invalid_update_length"

    return (
        ApiTeacherDecision(
            decision=decision,
            changed_elements=changed_elements,
            memory_update=update,
            confidence=confidence,
            evidence_sufficiency=evidence_sufficiency,
            significant_change=significant,
            identity_consistent=identity,
            standalone_complete=standalone,
            evidence=evidence,
        ),
        "",
    )


def quality_gate_api_decision(
    decision: ApiTeacherDecision,
    *,
    input_state: str,
    confidence_threshold: float,
) -> tuple[bool, str]:
    """应用不依赖模型自述的确定性门槛。"""

    if decision.decision == "uncertain":
        return False, "model_uncertain"
    if decision.confidence < confidence_threshold:
        return False, "low_confidence"
    if not decision.identity_consistent:
        return False, "identity_inconsistent"
    if decision.decision == "keep":
        return True, "verified_hard_null"
    assert decision.memory_update is not None
    if is_vacuous_update(decision.memory_update, input_state):
        return False, "vacuous_update"
    return True, "verified_update"


__all__ = [
    "API_ANNOTATION_POLICY",
    "API_HARD_NULL_SOURCE",
    "API_LABEL_SOURCE",
    "ApiTeacherDecision",
    "extract_json_object",
    "parse_api_teacher_response",
    "quality_gate_api_decision",
]
