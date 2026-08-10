"""CLIP tokenizer 的 optional import 包装，用于 USE_NLP 数据集。

当前环境没有安装 OpenAI 的 `clip` 包。这个模块提供清晰的错误提示，
并在未来需要时方便添加 vendored SimpleTokenizer。
"""

from __future__ import annotations

from typing import Any


def tokenize_text(text: str | list[str]) -> Any:
    """尝试调用 CLIP tokenizer；失败时抛出有用的错误信息。

    返回形状为 ``[batch, 77]`` 的 token tensor，与 OpenAI CLIP 兼容。
    """

    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "USE_NLP 数据集需要 CLIP tokenizer，但当前环境未安装 `clip` 包。\n"
            "解决方案：\n"
            "  1. pip install git+https://github.com/openai/CLIP.git\n"
            "  2. 或将该数据集从 tracker 配置的 use_nlp_datasets 列表中移除\n"
            "  3. 或在 runner 外部完成 tokenization 并传入 init_nlp_tokens"
        ) from exc
    return clip.tokenize(text)


__all__ = ["tokenize_text"]
