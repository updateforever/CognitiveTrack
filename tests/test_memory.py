import numpy as np

from cogtrack.memory import (
    GatedMemoryUpdatePolicy,
    IdentityAnchor,
    MemoryBank,
    MemoryCandidate,
    MemoryKind,
    MemorySource,
    MemoryUpdatePolicyConfig,
)
from cogtrack.protocol import ExecutionStatus, IdentityMatch, TargetPresence


def _candidate(frame_id):
    return MemoryCandidate(
        kind=MemoryKind.POSITIVE,
        frame_id=frame_id,
        source=MemorySource.VLM_PREDICTION,
        execution_status=ExecutionStatus.OK,
        target_presence=TargetPresence.PRESENT,
        identity_match=IdentityMatch.SAME,
        bbox_xywh=(10, 10, 20, 20),
        image=np.zeros((40, 40, 3), dtype=np.uint8),
    )


def test_positive_memory_requires_two_confirmations():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy()
    first = policy.process(bank, _candidate(1))
    second = policy.process(bank, _candidate(2))
    assert not first.accepted
    assert second.accepted
    assert len(bank.select_positive(4)) == 1


def test_execution_error_cannot_enter_memory():
    candidate = _candidate(1)
    candidate = MemoryCandidate(
        **{**candidate.__dict__, "execution_status": ExecutionStatus.MODEL_ERROR}
    )
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    decision = GatedMemoryUpdatePolicy().process(MemoryBank(anchor), candidate)
    assert not decision.accepted


def _semantic_candidate(frame_id: int, text: str) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.SEMANTIC,
        frame_id=frame_id,
        source=MemorySource.VLM_PREDICTION,
        execution_status=ExecutionStatus.OK,
        target_presence=TargetPresence.PRESENT,
        identity_match=IdentityMatch.SAME,
        bbox_xywh=(10, 10, 20, 20),
        text=text,
    )


def test_semantic_memory_has_deduplication_and_update_cooldown():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy(
        MemoryUpdatePolicyConfig(min_semantic_frame_gap=30)
    )

    first = policy.process(bank, _semantic_candidate(1, "Rear view reveals two white stripes."))
    too_soon = policy.process(bank, _semantic_candidate(10, "The target now carries a red bag."))
    duplicate = policy.process(bank, _semantic_candidate(40, " rear VIEW reveals  two WHITE stripes. "))
    accepted_later = policy.process(bank, _semantic_candidate(40, "The target now carries a red bag."))

    assert first.accepted is True
    assert too_soon.accepted is False
    assert "过近" in too_soon.reason
    assert duplicate.accepted is False
    assert "重复" in duplicate.reason
    assert accepted_later.accepted is True
    assert [record.text for record in bank.records(MemoryKind.SEMANTIC)] == [
        "Rear view reveals two white stripes.",
        "The target now carries a red bag.",
    ]


def test_no_change_text_is_treated_as_null_instead_of_polluting_memory():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy()

    for frame_id, text in enumerate(
        (
            "no change in appearance or configuration",
            "The target is unchanged.",
            "没有明显外观变化",
            "无需更新",
        ),
        start=1,
    ):
        decision = policy.process(bank, _semantic_candidate(frame_id, text))
        assert decision.accepted is False
        assert "memory_update=null" in decision.reason
    assert bank.records(MemoryKind.SEMANTIC) == ()
