"""记忆标注教师 Prompt。

和主链路 ``vlt_tracking`` 的关键区别：教师是**未微调的外部模型**，没有通过 SFT 内化
输出协议，所以判定标准和 JSON schema 必须写全。同时教师拿到的是已用 GT 框标注好的两
帧，它不需要定位，只需回答"从维护状态到当前帧，目标状态是否发生了值得写进记忆的变化"。

设计约束来自 ``docs/protocol.md``：
- ``current_target_state`` 是**整体替换快照**，不是增量日志。因此教师给出的必须是一句
  可以独立成立的完整描述，而不是"现在还多了一顶帽子"这种依赖上文的片段。
- ``initial_identity_description`` 永久不可变。教师不得改写身份，只能描述状态。
- 默认应当是 keep。记忆每更新一次就有一次漂移和污染的机会，宁可漏记也不能乱记。
"""

from __future__ import annotations

from .common import PromptSpec

STATE_TEACHER_PROMPT_NAME = "memory_state_teacher"
STATE_TEACHER_PROMPT_VERSION = "1.0.0"

#: 教师被允许写进 ``reason_codes`` 的封闭集合。开放式理由无法做一致性统计，也让两次
#: 生成之间无法比较；固定枚举才能算 agreement 并在审计时分层抽样。
STATE_TEACHER_REASON_CODES: tuple[str, ...] = (
    "no_significant_change",
    "action_changed",
    # 只指目标自身的姿态（站/坐/躺/攀爬），不含相机视角。实测把两者合成一个
    # ``pose_or_viewpoint_changed`` 时，41% 的更新在描述"从下方看""从侧面看"——那是观测
    # 角度，当前帧本来就摆在模型眼前，写进文本记忆只会制造 churn。文本记忆的价值在于
    # 承载图像通道跨几千帧承载不了的东西。
    "pose_changed",
    "appearance_changed",
    "scene_or_background_changed",
    "interaction_changed",
    "occlusion_recovered",
    "insufficient_evidence",
)

_SYSTEM_PROMPT = """You maintain a single-sentence state memory for a long-term visual object \
tracker. You are given the tracker's initialization frame and a later frame from the same \
video. In both frames the tracked target is already marked with a red box, so you never need \
to search for or localize the target.

Your only job: decide whether the target's state has changed enough that a future tracker \
would benefit from an updated state description.

Rules you must follow:
1. The initial identity is permanent and must never be contradicted or rewritten. If the \
initial identity says "a white airplane", the target is still that airplane even if it is now \
airborne, dirty, or partly hidden.
2. When you do update, output one complete standalone sentence that fully replaces the old \
state. It must remain true and understandable on its own, without reference to the previous \
state. Never write incremental fragments such as "now also wearing a hat".
3. Describe only what is visible in the marked region of the current frame. Never guess names, \
intentions, emotions, or offscreen context.
4. Default to keeping the current state. Update only for changes that persist beyond a few \
frames and that would help re-identify the target later: a different action, a different body \
pose, a changed outfit or surface appearance, a new interaction with another object or person, \
or a clearly new surrounding scene.
5. Never update for how the target is being observed rather than what it is doing. The camera \
angle, the viewing direction, how large the target appears, its distance, motion blur, and \
image quality are all already visible in the current frame and must never be written into the \
state. Phrases like "viewed from below", "seen from behind", "from a side view", "closer to \
the camera", or "appears larger" are always wrong here.
6. If the current frame is too blurred, too small, or too occluded to judge, keep the state \
and say so.

Return exactly one JSON object, no Markdown fences and no extra keys."""

_REASON_CODE_BLOCK = "\n".join(f"  - {code}" for code in STATE_TEACHER_REASON_CODES)


def build_state_teacher_prompt(
    *,
    initial_identity: str,
    current_state: str,
    frame_gap: int,
) -> PromptSpec:
    """构造一次记忆更新判定。

    ``frame_gap`` 只作为时间跨度提示。给它是因为"变化是否持久"无法从两张静态图判断：
    间隔 5 帧的姿态差极可能是瞬时动作，间隔 2000 帧的同样差异则大概率是真的状态改变。
    不把它写进 prompt，教师会把短间隔抖动误判成状态变化。
    """

    identity = str(initial_identity).strip()
    if not identity:
        raise ValueError("initial_identity 不能为空：教师必须知道不可变身份才能避免改写它")
    state = str(current_state).strip() or identity
    if isinstance(frame_gap, bool) or not isinstance(frame_gap, int) or frame_gap < 0:
        raise ValueError("frame_gap 必须是非负整数")

    user_prompt = f"""Image 1: the initialization frame. The red box marks the target.
Image 2: a later frame from the same video, {frame_gap} frames after Image 1. The red box \
marks the same target.

Permanent initial identity (never change this): {identity}
Currently maintained state: {state}

Decide whether the maintained state should be replaced.

Return exactly:
{{
  "state_changed": true or false,
  "reason_code": one of
{_REASON_CODE_BLOCK},
  "new_state": "one complete standalone sentence describing the target now; empty string when \
state_changed is false",
  "evidence": "what you see in the red box of Image 2 that supports your decision"
}}
When state_changed is false, new_state must be an empty string and reason_code must be \
no_significant_change or insufficient_evidence."""

    return PromptSpec(
        name=STATE_TEACHER_PROMPT_NAME,
        version=STATE_TEACHER_PROMPT_VERSION,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=2,
        include_memory_update=False,
    )


STATE_VERIFIER_PROMPT_NAME = "memory_state_verifier"
STATE_VERIFIER_PROMPT_VERSION = "1.0.0"

_VERIFIER_SYSTEM_PROMPT = """You audit proposed state-memory updates for a visual object \
tracker. You are shown a video's initialization frame and a later frame, with the tracked \
target marked by a red box in both, plus a candidate state sentence someone proposed for the \
later frame.

Judge only whether that candidate sentence is acceptable. You do not write your own \
description and you do not need to localize anything.

Reject the candidate if any of these hold:
- It states something not visible in the red box of the later frame (hallucination).
- It contradicts or rewrites the permanent initial identity.
- It is not a standalone sentence, or it only makes sense as a continuation of the old state.
- It describes a momentary motion blur or a trivial change rather than a persistent state.
- It is so vague that it would not help re-identify the target later.

Accept only when the sentence is visually supported, self-contained, consistent with the \
initial identity, and genuinely more informative than the old state.

Return exactly one JSON object, no Markdown fences and no extra keys."""


def build_state_verifier_prompt(
    *,
    initial_identity: str,
    previous_state: str,
    candidate_state: str,
    frame_gap: int,
) -> PromptSpec:
    """构造一次对候选记忆的独立裁决。

    verifier 只做二分接受判定，**不**给自己的改写版本。让它改写会把它变成第二个教师，
    最终标签又回到单模型自证；只判接受/拒绝才能把它的错误和教师的错误解耦。
    """

    identity = str(initial_identity).strip()
    if not identity:
        raise ValueError("initial_identity 不能为空")
    candidate = str(candidate_state).strip()
    if not candidate:
        raise ValueError("candidate_state 不能为空：没有候选就没有需要裁决的东西")
    previous = str(previous_state).strip() or identity
    if isinstance(frame_gap, bool) or not isinstance(frame_gap, int) or frame_gap < 0:
        raise ValueError("frame_gap 必须是非负整数")

    user_prompt = f"""Image 1: the initialization frame. The red box marks the target.
Image 2: a later frame from the same video, {frame_gap} frames after Image 1. The red box \
marks the same target.

Permanent initial identity: {identity}
Previously maintained state: {previous}
Candidate replacement state: {candidate}

Return exactly:
{{
  "accept": true or false,
  "failure_mode": "none | not_visible | contradicts_identity | not_standalone | \
trivial_change | too_vague",
  "justification": "what in Image 2 supports or refutes the candidate"
}}
When accept is true, failure_mode must be none."""

    return PromptSpec(
        name=STATE_VERIFIER_PROMPT_NAME,
        version=STATE_VERIFIER_PROMPT_VERSION,
        system_prompt=_VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=2,
        include_memory_update=False,
    )


STATE_KEEP_VERIFIER_PROMPT_NAME = "state_keep_verifier"
STATE_KEEP_VERIFIER_PROMPT_VERSION = "1.0.0"

_KEEP_VERIFIER_SYSTEM_PROMPT = """You audit a proposed no-update decision for the state memory of \
a visual object tracker. You are shown the initialization frame and a later frame, with the same \
target marked by a red box in both. You also receive the currently maintained state sentence.

Accept the no-update decision only when the later target remains adequately described by the \
maintained state and there is no clear, stable, future-useful change in the target's action, body \
pose, outfit or surface appearance, object interaction, or surrounding scene. Ordinary movement, \
scale, camera viewpoint, blur, and image quality are not state changes.

Reject when a clear state change makes the maintained state inadequate. If the later target is too \
small, blurred, or occluded to decide, reject as insufficient evidence: uncertainty must not be \
converted into a supervised no-update label.

Return exactly one JSON object, no Markdown fences and no extra keys."""


def build_state_keep_verifier_prompt(
    *,
    initial_identity: str,
    previous_state: str,
    frame_gap: int,
) -> PromptSpec:
    """构造一次 hard-null/no-update 的独立裁决。"""

    identity = str(initial_identity).strip()
    if not identity:
        raise ValueError("initial_identity 不能为空")
    previous = str(previous_state).strip() or identity
    if isinstance(frame_gap, bool) or not isinstance(frame_gap, int) or frame_gap < 0:
        raise ValueError("frame_gap 必须是非负整数")
    user_prompt = f"""Image 1: the initialization frame. The red box marks the target.
Image 2: a later frame from the same video, {frame_gap} frames after Image 1. The red box marks the \
same target.

Permanent initial identity: {identity}
Currently maintained state: {previous}
Proposed decision: keep the maintained state; emit memory_update=null.

Return exactly:
{{
  "accept_keep": true or false,
  "failure_mode": "none | clear_state_change | previous_state_unsupported | insufficient_evidence",
  "justification": "what in Image 2 supports or refutes keeping the state"
}}
When accept_keep is true, failure_mode must be none."""
    return PromptSpec(
        name=STATE_KEEP_VERIFIER_PROMPT_NAME,
        version=STATE_KEEP_VERIFIER_PROMPT_VERSION,
        system_prompt=_KEEP_VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=2,
        include_memory_update=False,
    )


__all__ = [
    "STATE_TEACHER_PROMPT_NAME",
    "STATE_TEACHER_PROMPT_VERSION",
    "STATE_TEACHER_REASON_CODES",
    "STATE_KEEP_VERIFIER_PROMPT_NAME",
    "STATE_KEEP_VERIFIER_PROMPT_VERSION",
    "STATE_VERIFIER_PROMPT_NAME",
    "STATE_VERIFIER_PROMPT_VERSION",
    "build_state_teacher_prompt",
    "build_state_keep_verifier_prompt",
    "build_state_verifier_prompt",
]
