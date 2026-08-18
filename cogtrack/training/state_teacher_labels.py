"""教师/verifier 输出的解析、校验与双次生成一致性判定。

单独成模块的原因：这里是标签质量真正被决定的地方，必须能在没有 GPU、没有模型权重的
情况下被单元测试完整覆盖。历史本地 teacher 推理入口已归档到
``tools/archive/state_teacher_v1/build_teacher_state_update_labels.py``；当前正式标签走
portable OpenAI-compatible API 流程。

流水线的判定顺序是"先结构、再语义、再一致性、最后裁决"，任何一步失败都记录具体原因而
不是静默丢弃——被拒的样本本身是评估教师可靠性的数据。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from cogtrack.prompts.state_teacher import STATE_TEACHER_REASON_CODES

#: 一句合格状态描述的长度边界（词数）。下限挡住 "walking" 这类无法独立成立的碎片，
#: 上限挡住教师把整段推理写进 new_state。
MIN_STATE_WORDS = 4
MAX_STATE_WORDS = 40

#: 教师常见的越界表达。它们要么把不可见信息写成事实，要么依赖上文，
#: 都违反"整体替换快照"语义。
_UNGROUNDED_PATTERNS = (
    r"\bnow also\b",
    r"\bstill\b.{0,20}\bas before\b",
    r"\bsame as\b",
    r"\bpreviously\b",
    r"\bunchanged\b",
    r"\bappears to be (?:thinking|planning|trying|about to)\b",
    r"\bprobably\b",
    r"\bmight be\b",
    r"\bseems to want\b",
    r"\bimage \d\b",
    r"\bred box\b",
    r"\bbounding box\b",
    r"\bthe frame\b",
)
_UNGROUNDED_RE = re.compile("|".join(_UNGROUNDED_PATTERNS), re.IGNORECASE)

#: 描述"如何被观测"而非"目标在做什么"的措辞。这些信息当前帧直接可见，写进跨帧文本记忆
#: 只会制造无意义的反复重写。prompt 里已经禁止，这里再挡一道：prompt 是建议，parser 是闸门。
_OBSERVATION_PATTERNS = (
    r"\bview(?:ed|ing)? from\b",
    r"\bseen from\b",
    r"\bfrom (?:below|behind|above|the side|a side|the front)\b",
    r"\b(?:side|front|rear|top|bottom|aerial|low|high)[- ]angle\b",
    r"\bcamera (?:angle|view|perspective)\b",
    r"\b(?:side|frontal|rear|profile) (?:view|perspective|orientation)\b",
    r"\bdifferent (?:orientation|perspective|angle)\b",
    r"\bcloser to the camera\b",
    r"\bfarther (?:from|away)\b",
    r"\bappears (?:larger|smaller|closer|farther)\b",
    r"\b(?:motion )?blur(?:red|ry)?\b",
    r"\bout of focus\b",
    r"\blow resolution\b",
    r"\bpartially (?:visible|occluded|hidden)\b",
)
_OBSERVATION_RE = re.compile("|".join(_OBSERVATION_PATTERNS), re.IGNORECASE)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class TeacherDecision:
    """一次教师判定的结构化结果。"""

    state_changed: bool
    reason_code: str
    new_state: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.state_changed, bool):
            raise TypeError("state_changed 必须是 bool")
        if self.state_changed and not self.new_state:
            raise ValueError("state_changed 为真时 new_state 不能为空")
        if not self.state_changed and self.new_state:
            raise ValueError("state_changed 为假时 new_state 必须为空")


@dataclass
class RejectionLog:
    """按原因累计被拒样本，用于事后评估教师可靠性。"""

    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        key = str(reason).strip() or "unknown"
        self.counts[key] = self.counts.get(key, 0) + 1

    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def extract_json_object(raw: str) -> dict | None:
    """从模型自由文本里取出唯一的 JSON 对象。

    未微调的教师经常在 JSON 前后带解释或 Markdown 围栏，直接 ``json.loads`` 会全量失败。
    这里做贪婪花括号匹配而不是逐行扫描：``evidence`` 字段里可能含有换行。
    """

    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_state_sentence(text: str) -> str:
    """压平空白并去掉围栏残留，不改写措辞。

    刻意不做首字母大写或补句号之类的美化：教师文本要和 MGIT 人工文本走同一条
    ``normalize_state_text`` 归一化路径，在这里提前加工会让两个来源的风格产生系统差异。
    """

    if not isinstance(text, str):
        return ""
    cleaned = text.replace("`", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(' "\'')


def parse_teacher_response(raw: str) -> tuple[TeacherDecision | None, str]:
    """解析教师输出，返回 ``(decision, rejection_reason)``。

    ``decision`` 为 ``None`` 时 ``rejection_reason`` 说明具体哪一步失败。keep 判定同样
    返回有效 decision——"不需要更新"是真实标签（``verified_hard_null``），不是失败。
    """

    payload = extract_json_object(raw)
    if payload is None:
        return None, "unparseable_json"

    if "state_changed" not in payload:
        return None, "missing_state_changed"
    changed = payload.get("state_changed")
    if isinstance(changed, str):
        lowered = changed.strip().lower()
        if lowered in {"true", "false"}:
            changed = lowered == "true"
    if not isinstance(changed, bool):
        return None, "non_boolean_state_changed"

    reason_code = str(payload.get("reason_code") or "").strip().lower()
    if reason_code not in STATE_TEACHER_REASON_CODES:
        return None, "unknown_reason_code"

    evidence = normalize_state_sentence(str(payload.get("evidence") or ""))
    new_state = normalize_state_sentence(str(payload.get("new_state") or ""))

    if not changed:
        if reason_code not in {"no_significant_change", "insufficient_evidence"}:
            return None, "keep_with_change_reason"
        return TeacherDecision(False, reason_code, "", evidence), ""

    if reason_code in {"no_significant_change", "insufficient_evidence"}:
        return None, "update_with_keep_reason"
    if not new_state:
        return None, "empty_new_state"

    words = new_state.split()
    if len(words) < MIN_STATE_WORDS:
        return None, "state_too_short"
    if len(words) > MAX_STATE_WORDS:
        return None, "state_too_long"
    if _UNGROUNDED_RE.search(new_state):
        return None, "ungrounded_or_relative_phrasing"
    if _OBSERVATION_RE.search(new_state):
        return None, "observation_not_state"

    return TeacherDecision(True, reason_code, new_state, evidence), ""


def parse_verifier_response(raw: str) -> tuple[bool | None, str]:
    """解析 verifier 输出，返回 ``(accept, failure_mode_or_reason)``。"""

    payload = extract_json_object(raw)
    if payload is None:
        return None, "unparseable_json"
    accept = payload.get("accept")
    if isinstance(accept, str):
        lowered = accept.strip().lower()
        if lowered in {"true", "false"}:
            accept = lowered == "true"
    if not isinstance(accept, bool):
        return None, "non_boolean_accept"
    failure_mode = str(payload.get("failure_mode") or "").strip().lower() or "unspecified"
    if accept and failure_mode != "none":
        # 自相矛盾的输出不可信：接受却同时报失败模式，说明它没真正做判定。
        return None, "accept_with_failure_mode"
    return accept, "none" if accept else failure_mode


def parse_keep_verifier_response(raw: str) -> tuple[bool | None, str]:
    """解析 hard-null verifier 输出，返回 ``(accept_keep, failure_mode)``。"""

    payload = extract_json_object(raw)
    if payload is None:
        return None, "unparseable_json"
    accept = payload.get("accept_keep")
    if isinstance(accept, str):
        lowered = accept.strip().lower()
        if lowered in {"true", "false"}:
            accept = lowered == "true"
    if not isinstance(accept, bool):
        return None, "non_boolean_accept_keep"
    failure_mode = str(payload.get("failure_mode") or "").strip().lower() or "unspecified"
    allowed = {
        "none",
        "clear_state_change",
        "previous_state_unsupported",
        "insufficient_evidence",
    }
    if failure_mode not in allowed:
        return None, "unknown_keep_failure_mode"
    if accept and failure_mode != "none":
        return None, "accept_keep_with_failure_mode"
    if not accept and failure_mode == "none":
        return None, "reject_keep_without_failure_mode"
    return accept, "none" if accept else failure_mode


def _token_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def state_agreement(first: str, second: str) -> float:
    """两句状态描述的 Jaccard 词重叠，用于双次生成一致性。

    用词集合而不是编辑距离：教师两次生成常常语序不同但语义相同（"a man riding a bike
    on the road" / "on the road, a man rides a bike"），编辑距离会把这种情形判为不一致。
    """

    a, b = _token_set(first), _token_set(second)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


#: 双次生成被认为语义一致的最低词重叠。0.5 是经验折中：过高会把同义改写全判为分歧，
#: 过低会让两句真正不同的描述蒙混过关。取任一版本进最终标签前必须先过这道闸。
MIN_SELF_AGREEMENT = 0.5

#: 更新文本与输入状态的词重叠上限。超过它说明教师只是把原句换了个说法，
#: 而"换个说法"不是状态变化——放进训练集会教模型无意义地反复重写记忆。
MAX_INPUT_OVERLAP = 0.9


def is_vacuous_update(new_state: str, input_state: str) -> bool:
    """判断一条更新是否等价于它自己的输入状态。

    实测的教师失效模式：声称 ``state_changed: true``，却把输入状态原样抄回，或只改了
    大小写和标点。对 MGIT 这种确定性分段标签，"更新等于输入"是代码 bug，构建期断言直接
    崩掉是对的；教师是随机采样，同一现象属于预期失效，必须在这里滤掉并计数。

    归一化后完全相等，或词重叠超过 ``MAX_INPUT_OVERLAP``，都算空更新。
    """

    candidate = normalize_state_sentence(new_state)
    baseline = normalize_state_sentence(input_state)
    if not candidate:
        return True
    if not baseline:
        return False
    if candidate.casefold().rstrip(".") == baseline.casefold().rstrip("."):
        return True
    return state_agreement(candidate, baseline) > MAX_INPUT_OVERLAP


MAX_REVISIT_OVERLAP = 0.85


def is_state_revisit(new_state: str, history: Sequence[str]) -> bool:
    """判断一条更新是不是回到了这条链上曾出现过的状态。

    替换快照记忆只有一个槽位，因此它只能承载**单调**变化：换了衣服不会换回来，进了室内
    不会自己回到室外。周期量则相反——走/停、弯腰/直立、骑车/推车每隔几秒循环一次。

    实测 TNL2K 90 序列：``appearance_changed`` 与 ``scene_or_background_changed`` 的状态
    回归率是 0%，而 ``action_changed`` 42%、``pose_changed`` 44%。最极端的
    ``CM_walkingnight_done`` 在 "walking" 与 "standing" 之间来回翻转 14 次。把这种链喂进
    训练集，学到的是"每隔几帧把记忆在两个短语间改写一次"，而正确行为是绝大多数帧输出
    null。这与之前删掉的视角措辞是同一类泄漏：当前帧本来就看得见的瞬时属性，不该占用
    跨帧记忆的唯一槽位。

    判据是回归而非"是否属于动作类"：动作只要单调（``standing`` → ``riding a bicycle``
    且不再回头）就是合法记忆；反过来外观若真的来回变化，同样该拦。用行为定性，不靠
    reason_code 自证。
    """

    candidate = normalize_state_sentence(new_state)
    if not candidate:
        return False
    key = candidate.casefold().rstrip(".")
    for earlier in history:
        previous = normalize_state_sentence(earlier)
        if not previous:
            continue
        if key == previous.casefold().rstrip("."):
            return True
        if state_agreement(candidate, previous) > MAX_REVISIT_OVERLAP:
            return True
    return False


def reconcile_dual_pass(
    first: TeacherDecision | None,
    second: TeacherDecision | None,
) -> tuple[TeacherDecision | None, str]:
    """裁决两次不同 seed 的教师生成，返回 ``(decision, reason)``。

    只在两次都判 update 且措辞语义一致时才产出 update 标签；两次都判 keep 时产出 keep。
    一次 update 一次 keep 说明这一帧本身处在判定边界上——这种样本进训练集会教模型在
    模糊处强行下注，因此直接丢弃，不做投票也不取其一。
    """

    if first is None or second is None:
        return None, "dual_pass_parse_failure"
    if first.state_changed != second.state_changed:
        return None, "dual_pass_decision_conflict"
    if not first.state_changed:
        if (
            first.reason_code == "insufficient_evidence"
            or second.reason_code == "insufficient_evidence"
        ):
            # “看不清所以不更新”是在线安全回退，不是已证明无需更新。把它做成
            # hard-null 全监督会把不确定性错误压成负标签。
            return None, "dual_pass_insufficient_evidence"
        return first, "dual_pass_keep"
    score = state_agreement(first.new_state, second.new_state)
    if score < MIN_SELF_AGREEMENT:
        return None, "dual_pass_wording_conflict"
    # 两版一致时取较短的一句：同等信息量下更短的描述更少冗余修饰，也更少幻觉空间。
    chosen = first if len(first.new_state) <= len(second.new_state) else second
    return chosen, "dual_pass_agreed"


__all__ = [
    "MAX_INPUT_OVERLAP",
    "MAX_STATE_WORDS",
    "MIN_SELF_AGREEMENT",
    "MIN_STATE_WORDS",
    "RejectionLog",
    "TeacherDecision",
    "extract_json_object",
    "is_vacuous_update",
    "normalize_state_sentence",
    "parse_keep_verifier_response",
    "parse_teacher_response",
    "parse_verifier_response",
    "reconcile_dual_pass",
    "state_agreement",
]
