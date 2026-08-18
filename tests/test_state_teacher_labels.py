"""教师输出解析与双次一致性裁决的聚焦测试。

这里覆盖的是标签质量闸门本身：解析失败、schema 自相矛盾、幻觉措辞、双次分歧。
每条都对应一种真实会从未微调教师那里收到的输出形态。
"""

from __future__ import annotations

import pytest

from cogtrack.training.state_teacher_labels import (
    MIN_SELF_AGREEMENT,
    RejectionLog,
    TeacherDecision,
    extract_json_object,
    is_vacuous_update,
    normalize_state_sentence,
    parse_keep_verifier_response,
    parse_teacher_response,
    parse_verifier_response,
    reconcile_dual_pass,
    state_agreement,
)


def _update_payload(state: str, reason: str = "action_changed") -> str:
    return (
        '{"state_changed": true, "reason_code": "%s", "new_state": "%s", '
        '"evidence": "the man is now on a bicycle"}' % (reason, state)
    )


class TestJsonExtraction:
    def test_strips_markdown_fence_and_surrounding_prose(self) -> None:
        raw = 'Sure! Here is the result:\n```json\n{"state_changed": false}\n```\nHope it helps.'
        assert extract_json_object(raw) == {"state_changed": False}

    def test_keeps_newlines_inside_string_values(self) -> None:
        raw = '{"state_changed": false, "evidence": "line one\\nline two"}'
        assert extract_json_object(raw)["evidence"] == "line one\nline two"

    @pytest.mark.parametrize("raw", ["", "   ", "no json at all", "{broken", None, 42])
    def test_returns_none_for_unusable_input(self, raw) -> None:
        assert extract_json_object(raw) is None

    def test_recovers_object_wrapped_in_a_single_element_array(self) -> None:
        # 未微调教师偶尔会把唯一对象包进数组。这是格式滑落而非语义歧义，恢复它零风险。
        assert extract_json_object('[{"state_changed": true}]') == {"state_changed": True}

    def test_rejects_multi_object_array_instead_of_guessing_which_one(self) -> None:
        # 贪婪匹配会跨越两个对象得到非法 JSON。这正是想要的结果：有多个候选判定时
        # 无从知道哪个才是答案，宁可整条丢弃。
        assert extract_json_object('[{"state_changed": true}, {"state_changed": false}]') is None


class TestTeacherParsing:
    def test_accepts_a_well_formed_update(self) -> None:
        decision, reason = parse_teacher_response(
            _update_payload("a man in a red jacket riding a bicycle on a city road")
        )
        assert reason == ""
        assert decision is not None
        assert decision.state_changed is True
        assert decision.reason_code == "action_changed"
        assert decision.new_state.startswith("a man in a red jacket")

    def test_keep_decision_is_a_valid_label_not_a_failure(self) -> None:
        decision, reason = parse_teacher_response(
            '{"state_changed": false, "reason_code": "no_significant_change", '
            '"new_state": "", "evidence": "the target has barely moved"}'
        )
        assert reason == ""
        assert decision is not None
        assert decision.state_changed is False
        assert decision.new_state == ""

    def test_accepts_stringified_booleans(self) -> None:
        decision, _ = parse_teacher_response(
            '{"state_changed": "false", "reason_code": "no_significant_change", '
            '"new_state": "", "evidence": "static"}'
        )
        assert decision is not None and decision.state_changed is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("not json", "unparseable_json"),
            ('{"reason_code": "action_changed"}', "missing_state_changed"),
            ('{"state_changed": "maybe", "reason_code": "action_changed"}',
             "non_boolean_state_changed"),
            ('{"state_changed": true, "reason_code": "vibes", "new_state": "a man rides a bike"}',
             "unknown_reason_code"),
            ('{"state_changed": true, "reason_code": "no_significant_change", '
             '"new_state": "a man rides a bike here"}', "update_with_keep_reason"),
            ('{"state_changed": false, "reason_code": "action_changed", "new_state": ""}',
             "keep_with_change_reason"),
            ('{"state_changed": true, "reason_code": "action_changed", "new_state": ""}',
             "empty_new_state"),
        ],
    )
    def test_rejects_malformed_or_contradictory_payloads(self, raw: str, expected: str) -> None:
        decision, reason = parse_teacher_response(raw)
        assert decision is None
        assert reason == expected

    def test_rejects_state_that_is_too_short_to_stand_alone(self) -> None:
        decision, reason = parse_teacher_response(_update_payload("walking"))
        assert decision is None
        assert reason == "state_too_short"

    def test_rejects_state_that_swallowed_the_whole_reasoning(self) -> None:
        decision, reason = parse_teacher_response(_update_payload(" ".join(["word"] * 60)))
        assert decision is None
        assert reason == "state_too_long"

    @pytest.mark.parametrize(
        "state",
        [
            "the man now also carries a backpack down the street",
            "the target is unchanged from the earlier observation here",
            "the same as before but closer to the camera now",
            "a man who probably wants to cross the street soon",
            "the person inside the red box walking on a road",
            "the target visible in Image 2 walking along a road",
        ],
    )
    def test_rejects_relative_or_ungrounded_phrasing(self, state: str) -> None:
        decision, reason = parse_teacher_response(_update_payload(state))
        assert decision is None, f"应当拒绝：{state!r}"
        assert reason == "ungrounded_or_relative_phrasing"

    @pytest.mark.parametrize(
        "state",
        [
            # 实测教师产出的原句：描述观测方式而非目标状态。
            "yellow airplane flying in the air, viewed from below",
            "brown airplane in flight, seen from below",
            "green airplane flying in the air from a side perspective",
            "yellow airplane flying in a different orientation",
            "whole body of the man in white, now standing with a rear view",
            "a white car driving along the road with motion blur",
            "the man walking down the street, partially occluded",
        ],
    )
    def test_rejects_observation_descriptions(self, state: str) -> None:
        # 相机视角、尺度、清晰度在当前帧直接可见，写进跨帧文本记忆只会制造 churn。
        decision, reason = parse_teacher_response(_update_payload(state))
        assert decision is None, f"应当拒绝：{state!r}"
        assert reason == "observation_not_state"

    @pytest.mark.parametrize(
        "state",
        [
            "a man in a red jacket riding a bicycle on a city road",
            "the man in white climbing a wooden structure",
            "yellow airplane landing on a runway",
            "the man standing upright in a garden",
        ],
    )
    def test_keeps_genuine_state_descriptions(self, state: str) -> None:
        # 反向保护：观测措辞过滤器不能误伤真正描述目标状态的句子。
        decision, reason = parse_teacher_response(_update_payload(state))
        assert decision is not None, f"不该拒绝：{state!r}（reason={reason}）"


class TestVerifierParsing:
    def test_parses_acceptance(self) -> None:
        accept, mode = parse_verifier_response('{"accept": true, "failure_mode": "none"}')
        assert accept is True
        assert mode == "none"

    def test_parses_rejection_with_failure_mode(self) -> None:
        accept, mode = parse_verifier_response(
            '{"accept": false, "failure_mode": "not_visible", "justification": "no bike"}'
        )
        assert accept is False
        assert mode == "not_visible"

    def test_rejects_self_contradictory_acceptance(self) -> None:
        accept, mode = parse_verifier_response('{"accept": true, "failure_mode": "too_vague"}')
        assert accept is None
        assert mode == "accept_with_failure_mode"

    @pytest.mark.parametrize("raw", ["garbage", '{"failure_mode": "none"}', '{"accept": "yes"}'])
    def test_rejects_unusable_verifier_output(self, raw: str) -> None:
        accept, _ = parse_verifier_response(raw)
        assert accept is None

    def test_accepts_verified_keep(self) -> None:
        accept, mode = parse_keep_verifier_response(
            '{"accept_keep": true, "failure_mode": "none", "justification": "same state"}'
        )
        assert accept is True
        assert mode == "none"

    def test_rejects_uncertain_keep(self) -> None:
        accept, mode = parse_keep_verifier_response(
            '{"accept_keep": false, "failure_mode": "insufficient_evidence", '
            '"justification": "target too small"}'
        )
        assert accept is False
        assert mode == "insufficient_evidence"

    @pytest.mark.parametrize(
        "raw",
        [
            '{"accept_keep": true, "failure_mode": "clear_state_change"}',
            '{"accept_keep": false, "failure_mode": "none"}',
            '{"accept_keep": false, "failure_mode": "unknown"}',
        ],
    )
    def test_rejects_inconsistent_keep_verdict(self, raw: str) -> None:
        accept, _ = parse_keep_verifier_response(raw)
        assert accept is None


class TestAgreement:
    def test_word_order_changes_do_not_count_as_disagreement(self) -> None:
        score = state_agreement(
            "a man riding a bicycle on the road",
            "on the road, a man rides a bicycle",
        )
        assert score >= MIN_SELF_AGREEMENT

    def test_genuinely_different_descriptions_disagree(self) -> None:
        score = state_agreement(
            "a man riding a bicycle on the road",
            "a white airplane parked at an airport gate",
        )
        assert score < MIN_SELF_AGREEMENT

    def test_identical_text_scores_one(self) -> None:
        assert state_agreement("a man on a bike", "a man on a bike") == 1.0

    def test_empty_against_nonempty_scores_zero(self) -> None:
        assert state_agreement("", "a man on a bike") == 0.0


class TestDualPassReconciliation:
    def _update(self, state: str) -> TeacherDecision:
        return TeacherDecision(True, "action_changed", state, "evidence")

    def _keep(self) -> TeacherDecision:
        return TeacherDecision(False, "no_significant_change", "", "evidence")

    def test_agreeing_updates_produce_a_label(self) -> None:
        decision, reason = reconcile_dual_pass(
            self._update("a man riding a bicycle on the road"),
            self._update("a man rides a bicycle along the road"),
        )
        assert reason == "dual_pass_agreed"
        assert decision is not None and decision.state_changed

    def test_agreeing_updates_prefer_the_shorter_wording(self) -> None:
        short = "a man riding a bicycle on the road"
        long = "a man riding a bicycle on the road near some parked cars"
        decision, _ = reconcile_dual_pass(self._update(long), self._update(short))
        assert decision is not None and decision.new_state == short

    def test_agreeing_keeps_produce_a_keep_label(self) -> None:
        decision, reason = reconcile_dual_pass(self._keep(), self._keep())
        assert reason == "dual_pass_keep"
        assert decision is not None and decision.state_changed is False

    def test_insufficient_evidence_does_not_become_hard_null(self) -> None:
        uncertain = TeacherDecision(False, "insufficient_evidence", "", "too blurred")
        decision, reason = reconcile_dual_pass(uncertain, uncertain)
        assert decision is None
        assert reason == "dual_pass_insufficient_evidence"

    def test_decision_conflict_is_discarded_not_voted(self) -> None:
        decision, reason = reconcile_dual_pass(
            self._update("a man riding a bicycle on the road"), self._keep()
        )
        assert decision is None
        assert reason == "dual_pass_decision_conflict"

    def test_wording_conflict_is_discarded(self) -> None:
        decision, reason = reconcile_dual_pass(
            self._update("a man riding a bicycle on the road"),
            self._update("a white airplane parked at the airport gate"),
        )
        assert decision is None
        assert reason == "dual_pass_wording_conflict"

    @pytest.mark.parametrize(
        ("first", "second"),
        [(None, None), (None, "update"), ("update", None)],
    )
    def test_parse_failure_in_either_pass_discards_the_frame(self, first, second) -> None:
        a = self._update("a man riding a bicycle on the road") if first else None
        b = self._update("a man riding a bicycle on the road") if second else None
        decision, reason = reconcile_dual_pass(a, b)
        assert decision is None
        assert reason == "dual_pass_parse_failure"


class TestVacuousUpdates:
    """教师声称"变了"但文本等价于输入状态——实测确实发生的失效模式。"""

    def test_identical_text_is_vacuous(self) -> None:
        state = "yellow airplane flying in the air"
        assert is_vacuous_update(state, state) is True

    def test_case_and_period_only_difference_is_vacuous(self) -> None:
        assert is_vacuous_update(
            "Yellow airplane flying in the air.", "yellow airplane flying in the air"
        ) is True

    def test_reworded_but_equivalent_text_is_vacuous(self) -> None:
        assert is_vacuous_update(
            "a yellow airplane flying in the air",
            "yellow airplane flying in the air",
        ) is True

    def test_genuine_change_is_not_vacuous(self) -> None:
        assert is_vacuous_update(
            "yellow airplane landing on a runway",
            "yellow airplane flying in the air",
        ) is False

    def test_empty_candidate_is_vacuous(self) -> None:
        assert is_vacuous_update("", "yellow airplane flying in the air") is True

    def test_empty_baseline_never_makes_a_candidate_vacuous(self) -> None:
        # 序列没有身份文本时输入状态可能为空，此时任何描述都是新增信息。
        assert is_vacuous_update("a yellow airplane in the air", "") is False


class TestInvariants:
    def test_decision_rejects_update_without_text(self) -> None:
        with pytest.raises(ValueError, match="new_state 不能为空"):
            TeacherDecision(True, "action_changed", "", "evidence")

    def test_decision_rejects_keep_carrying_text(self) -> None:
        with pytest.raises(ValueError, match="new_state 必须为空"):
            TeacherDecision(False, "no_significant_change", "a man on a bike", "evidence")

    def test_normalization_flattens_whitespace_without_restyling(self) -> None:
        assert normalize_state_sentence("  a man   riding\na bike  ") == "a man riding a bike"

    def test_normalization_preserves_leading_case(self) -> None:
        # 归一化必须留给 normalize_state_text 统一处理，否则教师文本和 MGIT 文本风格分叉。
        assert normalize_state_sentence("the man walks") == "the man walks"

    def test_rejection_log_accumulates_by_reason(self) -> None:
        log = RejectionLog()
        log.add("unparseable_json")
        log.add("unparseable_json")
        log.add("state_too_short")
        assert log.total() == 3
        assert log.to_dict() == {"state_too_short": 1, "unparseable_json": 2}
