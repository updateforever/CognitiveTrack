"""远程强教师的单次动态指代表达标注 Prompt。

这个 Prompt 会随可搬运 annotation bundle 一起发布，因此远端只需要 OpenAI-compatible
API 和 ``openai`` Python 包，不需要安装 CognitiveTrack。教师看到与学生相同的身份锚点
和历史轨迹；唯一额外信息是 Image 3 上用于离线标注的 GT 红框。
"""

from __future__ import annotations

STATE_API_TEACHER_PROMPT_NAME = "state_update_api_teacher"
STATE_API_TEACHER_PROMPT_VERSION = "2.2.1"

STATE_API_CHANGE_ELEMENTS: tuple[str, ...] = (
    "action",
    "pose",
    "appearance",
    "viewpoint",
    "scale",
    "visibility",
    "interaction",
    "scene",
    "other",
)

STATE_API_SYSTEM_PROMPT = """You create high-precision dynamic referring-expression labels for a long-term
single-object tracker.

Image 1 is the permanent identity anchor. Its red box marks the tracked target.
Image 2 is a chronological left-to-right strip of three earlier trusted observations. Every red
box marks the same target. Repeated panels are padding, not additional events.
Image 3 is the current frame. Its red box is an offline annotation aid and will be removed from the
student model's input.

The API message places an explicit role label immediately before each image. Only the image labeled
IMAGE 3 is current. IMAGE 1 and IMAGE 2 are earlier evidence and must never be described as the
current frame.

Decide whether the maintained target description should be replaced now. This memory is a compact
referring expression for recognizing the target in later frames under the permanent visual
identity, not a monotonic story summary. The supplied initial text is only the first dynamic
description: it may be coarse, use a broad dataset category, or be visually inaccurate. Keep the
text exactly as supplied in the input, but freely replace it with a substantially different and
more accurate description when the boxed visual evidence supports doing so. Accuracy is more
important than label yield. Use "uncertain" whenever the evidence is insufficient.

Rules:
1. Image 1, not the supplied initial text, is the sole permanent identity anchor. Never switch to a
   different physical target. A mismatch between a textual category and the boxed target is a
   description error, not identity drift. Correct that text with an update when the images are
   clear. Set identity_consistent according to visual continuity with the boxed target in Image 1,
   never according to whether the wording or category name stayed unchanged.
2. An update is a complete standalone replacement referring expression, never an appended delta.
   It may substantially rewrite every dynamic detail, including a coarse or inaccurate object
   category, while preserving the visual target identity. Target 12 to 24 English words and never
   exceed 30. Prioritize target appearance, distinctive markings, recognition-relevant viewpoint,
   scale, visibility, and interaction. Omit ordinary travel direction and incidental background
   whenever they would not help recognize the target later. The expression must be understandable
   without phrases such as "now", "still", "also", "same as before", or image IDs.
3. Update when one or more current-state elements have changed significantly relative to the
   maintained state and the boxed temporal history, and describing the new state would help future
   tracking. Relevant elements may include action, pose, appearance, viewpoint, scale, visibility,
   interaction, or scene. You decide their importance from the images; do not apply a fixed
   category preference.
4. A change must be likely to remain useful for recognizing the target in later frames. Do not
   update solely because of left/right travel direction, an ordinary background change, transient
   rotor or limb motion, sharper or blurrier rendering, or a small pose, viewpoint, or scale change.
   A substantial viewpoint, scale, visibility, interaction, or appearance change remains useful
   when it materially changes how the target should be recognized. Scene is relevant only when it
   changes target visibility or a distinctive target interaction.
5. A state may legitimately return to an earlier state. Judge the current evidence rather than
   enforcing monotonic change.
6. "keep" is a verified negative label: choose it only when the evidence is sufficient and the
   maintained state remains adequate. Ambiguous evidence must be "uncertain", never "keep".
7. The replacement must describe only visually supported target state. Do not infer names,
   intentions, emotions, or off-screen events.
8. Before answering, first describe the boxed target in IMAGE 3 internally, then check temporal
   consistency across IMAGE 2, identity consistency with IMAGE 1, and whether the sentence is a
   useful complete replacement. Begin the evidence field with "Image 3 shows" and never call an
   IMAGE 1 or IMAGE 2 observation current.
9. If the maintained description says the target disappeared but Image 3 visibly contains the
   boxed target, this is a reappearance: choose update, include visibility in changed_elements,
   and produce a useful current referring expression that explicitly states reappearance.

Return exactly one JSON object with no Markdown and no extra keys."""

STATE_API_USER_TEMPLATE = """Initial maintained target description (may be coarse or inaccurate): {initial_identity}
Currently maintained target description: {current_state}
Current frame gap from initialization: {frame_gap} frames

Return exactly:
{{
  "decision": "update | keep | uncertain",
  "changed_elements": ["action | pose | appearance | viewpoint | scale | visibility | interaction |
scene | other"],
  "memory_update": "a complete standalone referring expression, or null",
  "confidence": 0.0,
  "evidence_sufficiency": "sufficient | insufficient",
  "significant_change": true,
  "identity_consistent": true,
  "standalone_complete": true,
  "evidence": "concise visible evidence from the boxed target and temporal history"
}}

For update: changed_elements is non-empty, memory_update is a non-empty standalone referring
expression of 5 to 30 English words for the currently boxed target, and significant_change is true.
Focus the sentence on the elements that changed significantly. A visually supported correction of
the maintained text may be a large rewrite and is a valid update. The permanent visual identity
remains in Image 1 and need not be copied from the possibly inaccurate initial text.
For keep: changed_elements is empty, memory_update is null, evidence is sufficient, and
significant_change is false. The maintained description need not match incidental direction,
background, motion blur, or other transient details word for word in order to remain adequate.
For uncertain: changed_elements is empty, memory_update is null, and evidence is insufficient.
confidence is your confidence in the selected decision, not target visibility."""


def state_api_prompt_contract() -> dict[str, object]:
    """返回可直接写入跨服务器 bundle 的版本化 Prompt 合同。"""

    return {
        "name": STATE_API_TEACHER_PROMPT_NAME,
        "version": STATE_API_TEACHER_PROMPT_VERSION,
        "expected_image_count": 3,
        "system_prompt": STATE_API_SYSTEM_PROMPT,
        "user_prompt_template": STATE_API_USER_TEMPLATE,
        "change_elements": list(STATE_API_CHANGE_ELEMENTS),
    }


__all__ = [
    "STATE_API_CHANGE_ELEMENTS",
    "STATE_API_SYSTEM_PROMPT",
    "STATE_API_TEACHER_PROMPT_NAME",
    "STATE_API_TEACHER_PROMPT_VERSION",
    "STATE_API_USER_TEMPLATE",
    "state_api_prompt_contract",
]
