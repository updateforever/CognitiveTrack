from __future__ import annotations

import json

from cogtrack.prompts.state_api_teacher import state_api_prompt_contract
from cogtrack.training.state_api_labels import (
    parse_api_teacher_response,
    quality_gate_api_decision,
)
from tracking.annotate_state_update_openai_api import _parse_and_gate


def _response(**overrides) -> str:
    payload = {
        "decision": "update",
        "changed_elements": ["viewpoint", "appearance"],
        "memory_update": "the tracked airplane shows its silver underside from a side viewpoint",
        "confidence": 0.94,
        "evidence_sufficiency": "sufficient",
        "significant_change": True,
        "identity_consistent": True,
        "standalone_complete": True,
        "evidence": "Image 3 shows the boxed airplane consistently from the side",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_prompt_treats_memory_as_dynamic_referring_expression() -> None:
    contract = state_api_prompt_contract()
    assert contract["expected_image_count"] == 3
    assert contract["version"] == "2.2.1"
    assert "referring expression" in str(contract["system_prompt"])
    assert "viewpoint" in contract["change_elements"]


def test_prompt_allows_large_correction_without_changing_visual_identity() -> None:
    contract = state_api_prompt_contract()
    system_prompt = str(contract["system_prompt"])
    user_prompt = str(contract["user_prompt_template"])
    assert "sole permanent identity anchor" in system_prompt
    assert "description error, not identity drift" in system_prompt
    assert "including a coarse or inaccurate object" in system_prompt
    assert "Target 12 to 24 English words and never\n   exceed 30" in system_prompt
    assert "may be coarse or inaccurate" in user_prompt


def test_prompt_uses_only_image_three_as_current_and_avoids_transient_updates() -> None:
    system_prompt = str(state_api_prompt_contract()["system_prompt"])
    assert "Only the image labeled\nIMAGE 3 is current" in system_prompt
    assert "Do not\n   update solely because of left/right travel direction" in system_prompt
    assert 'Begin the evidence field with "Image 3 shows"' in system_prompt


def test_viewpoint_description_is_not_removed_by_hardcoded_filter() -> None:
    decision, reason = parse_api_teacher_response(_response())
    assert reason == ""
    assert decision is not None
    accepted, outcome = quality_gate_api_decision(
        decision,
        input_state="an airplane flying in the sky",
        confidence_threshold=0.85,
    )
    assert accepted is True
    assert outcome == "verified_update"


def test_uncertain_and_vacuous_outputs_are_rejected() -> None:
    uncertain, _ = parse_api_teacher_response(
        _response(
            decision="uncertain",
            changed_elements=[],
            memory_update=None,
            confidence=0.9,
            evidence_sufficiency="insufficient",
            significant_change=False,
        )
    )
    assert uncertain is not None
    assert quality_gate_api_decision(
        uncertain,
        input_state="an airplane flying in the sky",
        confidence_threshold=0.85,
    ) == (False, "model_uncertain")

    vacuous, _ = parse_api_teacher_response(
        _response(memory_update="an airplane flying in the blue sky above the ground")
    )
    assert vacuous is not None
    assert quality_gate_api_decision(
        vacuous,
        input_state="an airplane flying in the blue sky above the ground",
        confidence_threshold=0.85,
    ) == (False, "vacuous_update")


def test_evidence_must_ground_the_decision_in_image_three() -> None:
    decision, reason = parse_api_teacher_response(
        _response(evidence="Image 2 shows an airplane on the runway")
    )
    assert decision is None
    assert reason == "invalid_current_evidence"

    parsed, reason = _parse_and_gate(
        _response(evidence="The current image appears to show an airplane"),
        input_state="an airplane on the runway",
        confidence_threshold=0.85,
        require_update=False,
    )
    assert parsed is None
    assert reason == "invalid_current_evidence"


def test_reappearance_requires_an_update_in_portable_runner() -> None:
    parsed, reason = _parse_and_gate(
        _response(
            decision="keep",
            changed_elements=[],
            memory_update=None,
            evidence_sufficiency="sufficient",
            significant_change=False,
        ),
        input_state="The target has disappeared and is currently not visible in the search frame.",
        confidence_threshold=0.85,
        require_update=True,
    )
    assert parsed is None
    assert reason == "reappearance_requires_update"
