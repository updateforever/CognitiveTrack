"""独立目标身份核验的编排。"""

from dataclasses import dataclass
from typing import Any, Optional

from ..prompts.identity import build_identity_prompt
from ..protocol.enums import IdentityMatch
from ..vlm.base import GenerationConfig, VLMBackend, VLMResponse
from ..vlm.parser import ParsedIdentityOutput, parse_identity_output


@dataclass(frozen=True)
class IdentityVerificationResult:
    """身份核验结构化结果和原始生成响应。"""

    identity_match: IdentityMatch
    reasoning: str
    raw: ParsedIdentityOutput
    response: VLMResponse
    prompt_name: str
    prompt_version: str


class IdentityVerifier:
    """使用统一 VLMBackend 执行参考目标与候选目标的二图核验。"""

    def __init__(
        self,
        backend: VLMBackend,
    ) -> None:
        if not isinstance(backend, VLMBackend):
            # ABC 的明确检查能尽早发现把模型对象误当 backend 的配置错误。
            raise TypeError("backend 必须实现 VLMBackend")
        self.backend = backend

    def verify(
        self,
        reference_image: Any,
        candidate_image: Any,
        target_text: str = "",
        generation_config: Optional[GenerationConfig] = None,
    ) -> IdentityVerificationResult:
        """执行核验；证据不足时由模型显式输出 uncertain。"""

        prompt = build_identity_prompt(target_text)
        response = self.backend.generate(
            [reference_image, candidate_image],
            prompt.user_prompt,
            system_prompt=prompt.system_prompt,
            generation_config=generation_config,
        )
        parsed = parse_identity_output(response.text)
        return IdentityVerificationResult(
            identity_match=parsed.identity_match,
            reasoning=parsed.reasoning,
            raw=parsed,
            response=response,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
        )
