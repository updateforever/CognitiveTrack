import numpy as np

from cogtrack.memory import (
    SEMANTIC_EVENT_CONTINUED_ABSENCE,
    SEMANTIC_EVENT_DISAPPEARANCE,
    SEMANTIC_EVENT_REAPPEARANCE,
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


def _semantic_candidate(
    frame_id: int,
    text: str,
    *,
    presence: TargetPresence = TargetPresence.PRESENT,
    temporal_event: str | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.SEMANTIC,
        frame_id=frame_id,
        source=MemorySource.VLM_PREDICTION,
        execution_status=ExecutionStatus.OK,
        target_presence=presence,
        identity_match=(
            IdentityMatch.SAME
            if presence is TargetPresence.PRESENT
            else IdentityMatch.NOT_APPLICABLE
        ),
        bbox_xywh=(10, 10, 20, 20) if presence is TargetPresence.PRESENT else None,
        text=text,
        metadata={"temporal_event": temporal_event} if temporal_event else {},
    )


def test_semantic_memory_has_deduplication_and_update_cooldown():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy(
        MemoryUpdatePolicyConfig(min_semantic_frame_gap=30, semantic_confirmations=2)
    )

    first_pending = policy.process(bank, _semantic_candidate(1, "Rear view reveals two white stripes."))
    first = policy.process(bank, _semantic_candidate(2, "Rear view reveals two stable white stripes."))
    too_soon = policy.process(bank, _semantic_candidate(10, "The target now carries a red bag."))
    duplicate = policy.process(bank, _semantic_candidate(40, " rear VIEW reveals  two STABLE WHITE stripes. "))
    later_pending = policy.process(bank, _semantic_candidate(40, "The target now carries a red bag."))
    accepted_later = policy.process(bank, _semantic_candidate(41, "Target now carries the same red bag."))

    assert first_pending.accepted is False
    assert "等待跨帧" in first_pending.reason
    assert first.accepted is True
    assert too_soon.accepted is False
    assert "过近" in too_soon.reason
    assert duplicate.accepted is False
    assert "重复" in duplicate.reason
    assert later_pending.accepted is False
    assert accepted_later.accepted is True
    assert [record.text for record in bank.records(MemoryKind.SEMANTIC)] == [
        "Rear view reveals two stable white stripes.",
        "Target now carries the same red bag.",
    ]


def test_semantic_confirmation_rejects_unrelated_second_proposal():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy(MemoryUpdatePolicyConfig(semantic_confirmations=2))

    first = policy.process(bank, _semantic_candidate(1, "Rear view reveals two white stripes."))
    unrelated = policy.process(bank, _semantic_candidate(2, "The target now carries a red bag."))

    assert first.accepted is False
    assert unrelated.accepted is False
    assert unrelated.confirmations == 1
    assert bank.records(MemoryKind.SEMANTIC) == ()


def test_semantic_confirmation_uses_a_separate_sparse_frame_window():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy(
        MemoryUpdatePolicyConfig(
            max_confirmation_gap=30,
            semantic_confirmations=2,
            max_semantic_confirmation_gap=300,
        )
    )

    first = policy.process(bank, _semantic_candidate(10, "Rear view reveals two white stripes."))
    second = policy.process(
        bank,
        _semantic_candidate(100, "Rear view reveals two stable white stripes."),
    )

    assert first.accepted is False
    assert second.accepted is True
    assert second.confirmations == 2


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


def test_semantic_presence_transitions_are_immediate_and_bypass_cooldown():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy(
        MemoryUpdatePolicyConfig(min_semantic_frame_gap=30, semantic_confirmations=2)
    )

    disappeared = policy.process(
        bank,
        _semantic_candidate(
            1,
            "The initialized target is currently absent.",
            presence=TargetPresence.ABSENT,
            temporal_event=SEMANTIC_EVENT_DISAPPEARANCE,
        ),
    )
    reappeared = policy.process(
        bank,
        _semantic_candidate(
            2,
            "The initialized target has reappeared with its rear stripes visible.",
            temporal_event=SEMANTIC_EVENT_REAPPEARANCE,
        ),
    )

    assert disappeared.accepted is True
    assert reappeared.accepted is True
    assert [record.metadata["target_presence"] for record in bank.records(MemoryKind.SEMANTIC)] == [
        "absent",
        "present",
    ]


def test_continued_absence_semantic_proposal_is_rejected():
    anchor = IdentityAnchor(0, (10, 10, 20, 20), image=np.zeros((40, 40, 3), dtype=np.uint8))
    bank = MemoryBank(anchor)
    policy = GatedMemoryUpdatePolicy()

    decision = policy.process(
        bank,
        _semantic_candidate(
            2,
            "The initialized target remains absent.",
            presence=TargetPresence.ABSENT,
            temporal_event=SEMANTIC_EVENT_CONTINUED_ABSENCE,
        ),
    )

    assert decision.accepted is False
    assert "首次消失" in decision.reason
    assert bank.records(MemoryKind.SEMANTIC) == ()
