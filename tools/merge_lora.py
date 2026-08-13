#!/usr/bin/env python3
"""把 PEFT LoRA 合并成可由 vLLM 直接加载的独立模型目录。

Qwen3-VL 在部分 vLLM 版本中不支持在线多模态 LoRA：启用 ``--enable-lora``
会错误包装视觉塔线性层。离线合并不会改变推理范式，同时能让服务端走普通
Qwen3-VL + FlashAttention 路径。输出目录同时保存 processor/tokenizer，因而
可以直接作为 ``vllm serve`` 的模型参数。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合并 Qwen3-VL PEFT LoRA")
    parser.add_argument("--base-model", required=True, help="基础模型目录")
    parser.add_argument("--adapter", required=True, help="PEFT adapter 目录")
    parser.add_argument("--output", required=True, help="合并模型输出目录（必须不存在或为空）")
    parser.add_argument("--max-shard-size", default="5GB", help="safetensors 分片上限")
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_path = Path(args.base_model).expanduser().resolve()
    adapter_path = Path(args.adapter).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"基础模型目录不存在: {base_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json 不存在: {adapter_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"输出目录必须不存在或为空: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[1/4] 以 BF16/CPU 加载基础模型: {base_path}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_path),
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    print(f"[2/4] 加载并安全合并 LoRA: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)

    print(f"[3/4] 保存合并权重: {output_path}", flush=True)
    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    print("[4/4] 保存 processor/tokenizer", flush=True)
    processor = AutoProcessor.from_pretrained(
        str(base_path),
        local_files_only=True,
        use_fast=False,
    )
    processor.save_pretrained(str(output_path))
    print(f"[完成] {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
