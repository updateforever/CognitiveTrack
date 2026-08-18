import json
from pathlib import Path

import pytest

from cogtrack.training.loss_mask import (
    MEMORY_STATE_MASKED_UNKNOWN,
    MEMORY_STATE_VERIFIED_HARD_NULL,
    MEMORY_STATE_VERIFIED_UPDATE,
    assistant_loss_scale_for_state,
    decide_memory_supervision_state,
    split_tracking_core_response,
)
from tracking.validate_sft_supervision import validate_dataset


def test_tracking_core_masks_only_memory_value() -> None:
    response = (
        '{"bbox_2d":<bbox>,"status":"present",'
        '"memory_update":null}'
    )

    parts, weights = split_tracking_core_response(
        response,
        require_memory_field=True,
    )

    assert "".join(parts) == response
    assert weights == [1.0, 0.0, 1.0]
    assert parts[0].endswith('"memory_update":')
    assert parts[1] == "null"
    assert parts[2] == "}"
    assert "<bbox>" in parts[0]


def test_tracking_core_rejects_noncanonical_memory_boundary() -> None:
    response = (
        '{"bbox_2d":null,"status":"absent",'
        '"memory_update" : null}'
    )
    with pytest.raises(ValueError, match="canonical memory_update"):
        split_tracking_core_response(response, require_memory_field=True)


def test_tracking_core_rejects_extra_field_after_memory() -> None:
    response = (
        '{"bbox_2d":null,"status":"absent",'
        '"memory_update":null,"extra":1}'
    )
    with pytest.raises(ValueError, match="最后一个"):
        split_tracking_core_response(response, require_memory_field=True)


def test_two_field_history_remains_full_loss_in_non_strict_mode() -> None:
    response = '{"bbox_2d":null,"status":"absent"}'
    assert split_tracking_core_response(response) == ([response], [1.0])


def _row(
    *,
    state: str,
    temporal_case: str,
    memory_value: str = "null",
    bbox: str = "null",
    loss_scale: float | None = None,
    include_loss_scale: bool = True,
    verified_null: bool | None = None,
) -> dict:
    """构造一条三态监督 SFT 行。"""

    assistant: dict = {
        "role": "assistant",
        "content": (
            f'{{"bbox_2d":{bbox},"status":"{temporal_case}",'
            f'"memory_update":{memory_value}}}'
        ),
    }
    if include_loss_scale and loss_scale is not None:
        assistant["loss_scale"] = loss_scale
    row = {
        "messages": [
            {"role": "system", "content": "track"},
            {"role": "user", "content": "<image><image><image>"},
            assistant,
        ],
        "images": ["a.jpg", "b.jpg", "c.jpg"],
        "metadata": {
            "prompt_profile": "vlt_v6",
            "sft_supervision_profile": "tracking_core",
            "memory_supervision_state": state,
            "memory_loss_masked": state == "masked_unknown",
            "temporal_case": temporal_case,
        },
    }
    if verified_null is not None:
        row["metadata"]["memory_verified_null"] = verified_null
    return row


def _write(tmp_path: Path, row: dict) -> Path:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return dataset


def test_masked_unknown_row_carries_no_loss_scale(tmp_path: Path) -> None:
    """present 无可靠标签：不写 loss_scale，交给插件 mask memory 值。"""

    row = _row(state="masked_unknown", temporal_case="present", bbox="<bbox>")
    assert "loss_scale" not in row["messages"][2]
    assert validate_dataset(_write(tmp_path, row), profile="tracking_core") == 1


def test_verified_hard_null_row_requires_loss_scale_one(tmp_path: Path) -> None:
    """absent：null 必须参与 loss，因此必须显式 loss_scale=1.0。"""

    row = _row(
        state="verified_hard_null", temporal_case="absent", loss_scale=1.0
    )
    assert validate_dataset(_write(tmp_path, row), profile="tracking_core") == 1


def test_verified_update_row_requires_nonempty_text(tmp_path: Path) -> None:
    row = _row(
        state="verified_update",
        temporal_case="present",
        bbox="<bbox>",
        memory_value='"Rear view now shows two white stripes"',
        loss_scale=1.0,
    )
    assert validate_dataset(_write(tmp_path, row), profile="tracking_core") == 1


def test_state_and_loss_scale_must_agree(tmp_path: Path) -> None:
    """metadata 被 ms-swift 丢弃，只有 loss_scale 驱动 loss；不一致必须报错。

    这是本项目最危险的静默失败模式：审计报告说 hard-null 已全监督，实际训练
    仍在 mask，而训练日志看不出任何异常。
    """

    # 声明 hard-null 却漏写 loss_scale -> 实际仍被 mask
    row = _row(
        state="verified_hard_null",
        temporal_case="absent",
        loss_scale=1.0,
        include_loss_scale=False,
    )
    with pytest.raises(ValueError, match="metadata 不驱动 loss"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")

    # 声明 masked_unknown 却写了 loss_scale -> 占位 null 被当成真标签监督
    row = _row(state="masked_unknown", temporal_case="present", bbox="<bbox>")
    row["messages"][2]["loss_scale"] = 1.0
    with pytest.raises(ValueError, match="metadata 不驱动 loss"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")


def test_non_binary_loss_scale_is_rejected(tmp_path: Path) -> None:
    """LossScale.is_binary=True，非 0/1 会被 labels 路径静默吞掉。"""

    row = _row(state="verified_hard_null", temporal_case="absent", loss_scale=0.5)
    with pytest.raises(ValueError, match="0.0 或 1.0|loss_scale"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")


def test_hard_null_is_rejected_on_present_rows(tmp_path: Path) -> None:
    """present 行没有“确定不该更新”的证据，不能当负标签全监督。"""

    row = _row(
        state="verified_hard_null",
        temporal_case="present",
        bbox="<bbox>",
        loss_scale=1.0,
    )
    with pytest.raises(ValueError, match="只允许 absent"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")


def test_verified_update_cannot_be_null(tmp_path: Path) -> None:
    row = _row(
        state="verified_update", temporal_case="present", bbox="<bbox>", loss_scale=1.0
    )
    with pytest.raises(ValueError, match="非空状态文本"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")


def test_missing_memory_state_is_rejected(tmp_path: Path) -> None:
    row = _row(state="masked_unknown", temporal_case="present", bbox="<bbox>")
    del row["metadata"]["memory_supervision_state"]
    with pytest.raises(ValueError, match="memory_supervision_state"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")


def test_unknown_memory_state_is_rejected(tmp_path: Path) -> None:
    row = _row(state="masked_unknown", temporal_case="present", bbox="<bbox>")
    row["metadata"]["memory_supervision_state"] = "verified_hardnull"
    with pytest.raises(ValueError, match="memory_supervision_state 必须是"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")


def test_supervision_preflight_rejects_wrong_profile(tmp_path: Path) -> None:
    row = _row(state="verified_hard_null", temporal_case="absent", loss_scale=1.0)
    dataset = _write(tmp_path, row)
    with pytest.raises(ValueError, match="训练请求"):
        validate_dataset(dataset, profile="full")

    row["metadata"].update(
        sft_supervision_profile="full",
        memory_loss_masked=False,
        memory_supervision="feasibility_null",
    )
    with pytest.raises(ValueError, match="禁止用于 SFT"):
        validate_dataset(_write(tmp_path, row), profile="full")


def test_mixed_profile_accepts_tracking_and_state_rows(tmp_path: Path) -> None:
    tracking = _row(state="masked_unknown", temporal_case="present", bbox="<bbox>")
    tracking["metadata"]["sft_supervision_profile"] = "tracking_sft"
    state = _row(
        state="verified_hard_null",
        temporal_case="absent",
        loss_scale=1.0,
    )
    state["metadata"]["sft_supervision_profile"] = "state_update_sft"
    dataset = tmp_path / "mixed.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row) for row in (tracking, state)) + "\n",
        encoding="utf-8",
    )

    assert validate_dataset(dataset, profile="mixed_sft") == 2


def test_decide_state_absent_placeholder_is_masked_unknown() -> None:
    """tracking_sft 无状态标签时，absent null 也只是未知占位。"""
    assert (
        decide_memory_supervision_state(status="absent", memory_update=None)
        == MEMORY_STATE_MASKED_UNKNOWN
    )


def test_decide_state_present_null_is_masked_unknown() -> None:
    """占位 null 不是负标签，否则会把模型教成“永不更新”。"""
    assert (
        decide_memory_supervision_state(status="present", memory_update=None)
        == MEMORY_STATE_MASKED_UNKNOWN
    )


def test_decide_state_present_text_is_verified_update() -> None:
    assert (
        decide_memory_supervision_state(
            status="present", memory_update="the skateboarder is now airborne"
        )
        == MEMORY_STATE_VERIFIED_UPDATE
    )


def test_decide_state_accepts_absent_disappearance_text() -> None:
    assert (
        decide_memory_supervision_state(
            status="absent", memory_update="The target has disappeared from the search frame."
        )
        == MEMORY_STATE_VERIFIED_UPDATE
    )


def test_decide_state_rejects_blank_update_text() -> None:
    """空串是标签生产 bug，不能静默变成“已验证要写空 memory”。"""
    with pytest.raises(ValueError, match="不能为空"):
        decide_memory_supervision_state(status="present", memory_update="   ")


def test_decide_state_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status 必须是 present/absent"):
        decide_memory_supervision_state(status="occluded", memory_update=None)


def test_source_and_training_views_agree_on_state() -> None:
    """源层与训练视图必须同源判定，否则审计链会自相矛盾。

    这是 metadata 层的等价物：源 JSONL 说 absent 行被 mask，训练视图说全量监督，
    而两边都不报错。
    """
    for status, update in (
        ("absent", None),
        ("present", None),
        ("present", "target has turned around"),
    ):
        state = decide_memory_supervision_state(status=status, memory_update=update)
        scale = assistant_loss_scale_for_state(state)
        masked = state == MEMORY_STATE_MASKED_UNKNOWN
        # mask 与不写 loss_scale 必须一一对应。
        assert masked == (scale is None)


def test_decide_state_present_verified_null_is_hard_null() -> None:
    """present + 已证明无需更新 -> hard-null，null 参与 loss。

    没有这一态，被监督的 present 行只剩 verified_update，模型可以用“present 就
    输出文本”通过训练，而 hard-null 与 target_status=absent 完全共线。
    """
    assert (
        decide_memory_supervision_state(
            status="present", memory_update=None, verified_null=True
        )
        == MEMORY_STATE_VERIFIED_HARD_NULL
    )


def test_decide_state_verified_null_defaults_off() -> None:
    """默认必须是 masked：缺证据时不能把占位 null 当负标签。"""
    assert (
        decide_memory_supervision_state(status="present", memory_update=None)
        == MEMORY_STATE_MASKED_UNKNOWN
    )


def test_decide_state_rejects_verified_null_with_text() -> None:
    with pytest.raises(ValueError, match="verified_null=True 时 memory_update 必须为 null"):
        decide_memory_supervision_state(
            status="present", memory_update="moved left", verified_null=True
        )


def test_present_verified_null_is_fully_supervised() -> None:
    """hard-null 的 null 必须参与 loss，present 与 absent 一视同仁。"""
    state = decide_memory_supervision_state(
        status="present", memory_update=None, verified_null=True
    )
    assert assistant_loss_scale_for_state(state) == 1.0


def test_auditor_accepts_present_hard_null_only_when_declared(tmp_path: Path) -> None:
    """present + hard-null 必须显式声明证据；否则审计必须拒绝。

    这是"缺标签不能变负标签"这条不变量在审计层的最后一道闸门。
    """
    declared = _row(
        state="verified_hard_null",
        temporal_case="present",
        bbox="<bbox>",
        loss_scale=1.0,
        verified_null=True,
    )
    assert validate_dataset(_write(tmp_path, declared), profile="tracking_core") == 1

    undeclared = _row(
        state="verified_hard_null",
        temporal_case="present",
        bbox="<bbox>",
        loss_scale=1.0,
    )
    with pytest.raises(ValueError, match="verified_hard_null 只允许 absent 行"):
        validate_dataset(_write(tmp_path, undeclared), profile="tracking_core")


def test_auditor_rejects_declared_claim_that_fell_back_to_masked(tmp_path: Path) -> None:
    """声明了负标签却被 mask：审计说监督、训练实际不监督，必须报错。"""
    row = _row(
        state="masked_unknown",
        temporal_case="present",
        bbox="<bbox>",
        verified_null=True,
    )
    with pytest.raises(ValueError, match="已声明的负标签"):
        validate_dataset(_write(tmp_path, row), profile="tracking_core")
