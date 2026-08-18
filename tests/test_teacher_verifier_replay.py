from __future__ import annotations

from types import SimpleNamespace

from cogtrack.training.state_teacher_labels import RejectionLog
from tools.archive.state_teacher_v1 import build_teacher_state_update_labels as builder


class _Verifier:
    def __init__(self) -> None:
        self.users: list[str] = []

    def generate(self, requests, *, seed: int, temperature: float):
        assert seed == 0
        assert temperature == 0.0
        outputs = []
        for system, user, _images in requests:
            self.users.append(user)
            if "proposed no-update decision" in system:
                outputs.append(
                    '{"accept_keep": true, "failure_mode": "none", '
                    '"justification": "state remains adequate"}'
                )
            else:
                outputs.append(
                    '{"accept": false, "failure_mode": "not_visible", '
                    '"justification": "candidate unsupported"}'
                )
        return outputs


def test_verifier_replays_state_after_rejected_update(monkeypatch) -> None:
    monkeypatch.setattr(builder, "_marked_frame", lambda *args, **kwargs: object())
    walk = SimpleNamespace(
        dataset="lasot",
        name="sequence",
        identity="the original target identity",
        sequence=object(),
        anchor_frame=0,
    )
    rows = [
        {
            "dataset": "lasot",
            "sequence": "sequence",
            "frame_id": 100,
            "target_status": "present",
            "memory_update": "an unsupported replacement state",
            "verified_null": False,
            "source": "teacher",
            "reviewed": False,
            "input_state": walk.identity,
        },
        {
            "dataset": "lasot",
            "sequence": "sequence",
            "frame_id": 200,
            "target_status": "present",
            "memory_update": None,
            "verified_null": True,
            "source": "teacher_hard_null",
            "reviewed": False,
            # 教师链曾把 frame 100 当成已接受；verifier 拒绝后必须重放回 identity。
            "input_state": "an unsupported replacement state",
        },
    ]
    verifier = _Verifier()
    rejections = RejectionLog()
    kept = builder._apply_verifier_verdicts(
        rows,
        [walk],
        verifier,
        max_side=128,
        batch_size=8,
        rejections=rejections,
    )
    assert len(kept) == 1
    assert kept[0]["frame_id"] == 200
    assert kept[0]["input_state"] == walk.identity
    assert kept[0]["reviewed"] is True
    assert walk.identity in verifier.users[-1]
    assert "an unsupported replacement state" not in verifier.users[-1]
    assert rejections.to_dict() == {"not_visible": 1}
