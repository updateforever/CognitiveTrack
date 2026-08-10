"""版本化的 pair、mosaic 和隔离 identity Prompt。

主跟踪的记忆更新已经并入 pair/mosaic 第三字段；旧的独立 memory Prompt 不从
公共入口导出，避免线上误跑两次 VLM。
"""

from .common import PromptSpec
from .identity import (
    CANDIDATE_IDENTITY_PROMPT_NAME,
    CANDIDATE_IDENTITY_PROMPT_VERSION,
    IDENTITY_PROMPT_NAME,
    IDENTITY_PROMPT_VERSION,
    build_candidate_identity_prompt,
    build_identity_prompt,
)
from .mosaic import MOSAIC_PROMPT_NAME, MOSAIC_PROMPT_VERSION, build_mosaic_prompt
from .pair import PAIR_PROMPT_NAME, PAIR_PROMPT_VERSION, build_pair_prompt

__all__ = [
    "CANDIDATE_IDENTITY_PROMPT_NAME",
    "CANDIDATE_IDENTITY_PROMPT_VERSION",
    "IDENTITY_PROMPT_NAME",
    "IDENTITY_PROMPT_VERSION",
    "MOSAIC_PROMPT_NAME",
    "MOSAIC_PROMPT_VERSION",
    "PAIR_PROMPT_NAME",
    "PAIR_PROMPT_VERSION",
    "PromptSpec",
    "build_candidate_identity_prompt",
    "build_identity_prompt",
    "build_mosaic_prompt",
    "build_pair_prompt",
]
