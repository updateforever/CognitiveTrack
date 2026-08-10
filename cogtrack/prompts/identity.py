"""独立实例身份核验与候选级跟踪 Prompt。"""

from .common import TRACKING_SYSTEM_PROMPT, PromptSpec, target_text_section

IDENTITY_PROMPT_NAME = "identity_verification"
IDENTITY_PROMPT_VERSION = "2.0.0"
CANDIDATE_IDENTITY_PROMPT_NAME = "candidate_identity_hard_negative"
CANDIDATE_IDENTITY_PROMPT_VERSION = "2.0.0"

_CANDIDATE_IDENTITY_OUTPUT_SCHEMA = """Return exactly these six keys:
{
  "target_presence": "present",
  "identity_match": "same | different | uncertain",
  "localizability": "localizable",
  "bbox_norm1000_xyxy": [x1, y1, x2, y2],
  "target_text": "short identity description",
  "reasoning": "brief comparison evidence"
}"""


def build_identity_prompt(target_text: str = "") -> PromptSpec:
    """构造参考目标与候选目标的二图身份判别任务。"""

    system_prompt = """You are an instance identity verifier for long-term tracking.
Judge whether two observations show the exact same physical instance, not merely the same class.
Return exactly one JSON object without Markdown or additional text."""
    user_prompt = f"""Image 1: initialized target identity reference.
Image 2: current candidate crop.

Compare instance-specific color patterns, texture, shape, accessories, damage/marks, and other
stable cues. Allow reasonable pose, scale, illumination, and viewpoint changes. If discriminative
evidence is hidden or insufficient, return uncertain.

{target_text_section(target_text)}

Return exactly:
{{
  "identity_match": "same | different | uncertain",
  "reasoning": "brief comparison evidence"
}}
Do not return a confidence score, bounding box, Markdown, or extra keys."""
    return PromptSpec(
        name=IDENTITY_PROMPT_NAME,
        version=IDENTITY_PROMPT_VERSION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_image_count=2,
    )


def build_candidate_identity_prompt(target_text: str = "") -> PromptSpec:
    """构造带显式候选框的六字段身份困难负样本 Prompt。"""

    user_prompt = f"""Task: candidate-level identity verification for long-term tracking.

Image 1: initialized identity reference; its target is marked by a green box.
Image 2: a visible, localizable candidate of the same semantic class, marked by a red box.
The boxes identify which observations to compare; a box never implies that identities match.

For this candidate-level auxiliary task, target_presence describes whether the marked candidate is
visible, identity_match describes whether it is the exact initialized physical instance, and the
reported bbox is the marked candidate location even when identity_match is different. Compare
instance-specific texture, markings, shape, accessories, damage, and other stable visual evidence.

{target_text_section(target_text)}

{_CANDIDATE_IDENTITY_OUTPUT_SCHEMA}"""
    return PromptSpec(
        name=CANDIDATE_IDENTITY_PROMPT_NAME,
        version=CANDIDATE_IDENTITY_PROMPT_VERSION,
        system_prompt=TRACKING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_image_count=2,
    )
