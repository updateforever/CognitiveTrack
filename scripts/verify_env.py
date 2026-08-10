#!/usr/bin/env python
"""环境自检：确认 torch / flash-attn / vLLM / ms-swift 真的可用。

这里刻意不止于 ``import``。flash-attn 是按 (Python ABI, torch ABI, CUDA 大版本)
三元组预编译的，装错组合时 ``import flash_attn`` 常常仍然成功，直到第一次真正
调 kernel 才炸 ——  而那通常发生在训练跑了半小时之后。所以下面对 flash-attn
会实际执行一次前向。

退出码 0 表示全部通过，1 表示有失败项（末尾会汇总）。
"""

import platform
import sys
import traceback
from typing import Callable, List, Tuple

Check = Tuple[str, Callable[[], str]]


def _torch_info() -> str:
    import torch

    return (
        f"{torch.__version__}  cuda={torch.version.cuda}  "
        f"available={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}"
    )


def _gpu_info() -> str:
    import torch

    if not torch.cuda.is_available():
        return "无可用 GPU"
    names = []
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        names.append(f"{torch.cuda.get_device_name(index)} sm{capability[0]}{capability[1]}")
    unique = sorted(set(names))
    return f"{torch.cuda.device_count()} 卡: " + "; ".join(unique)


def _flash_attn_import() -> str:
    import flash_attn
    from flash_attn import flash_attn_func  # noqa: F401 - 确认符号真能解析

    return f"{flash_attn.__version__}"


def _flash_attn_kernel() -> str:
    """真正跑一次 flash-attn 前向，并与 SDPA 参考实现对比数值。

    只 import 不算验证：ABI 不匹配的典型症状就是 import 成功、调用即崩。
    这里顺带比对 SDPA，确保拿到的不只是「没崩」，而是「结果对」。
    """

    import torch
    from flash_attn import flash_attn_func

    if not torch.cuda.is_available():
        return "跳过（无 GPU）"

    torch.manual_seed(0)
    batch, seq_len, heads, dim = 2, 128, 8, 64
    # flash-attn 要 (batch, seqlen, heads, dim)，SDPA 要 (batch, heads, seqlen, dim)。
    query, key, value = (
        torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    )
    flash_output = flash_attn_func(query, key, value, causal=True)

    reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)

    max_delta = (flash_output.float() - reference.float()).abs().max().item()
    if max_delta > 2e-2:
        raise RuntimeError(f"flash-attn 与 SDPA 结果不一致，max|Δ|={max_delta:.3e}")
    return f"前向 OK, 与 SDPA 对齐 max|Δ|={max_delta:.2e}"


def _vllm_info() -> str:
    import vllm

    return f"{vllm.__version__}"


def _vllm_cuda_major() -> str:
    """确认 vLLM 扩展链接的 CUDA 大版本与 torch 一致。

    flash-attn 和 vLLM 各自都是按 CUDA 大版本编译的。这两者错配是本环境
    最隐蔽的坑，所以显式查一次 vLLM 扩展实际链接了哪个 libcudart。
    """

    import subprocess
    from pathlib import Path

    import vllm

    candidates = sorted(Path(vllm.__file__).parent.glob("_C*.so"))
    if not candidates:
        return "找不到 vllm/_C*.so，跳过"
    result = subprocess.run(
        ["ldd", str(candidates[0])],
        capture_output=True,
        text=True,
        check=False,
    )
    libs = [line.split()[0] for line in result.stdout.splitlines() if "libcudart" in line]
    if not libs:
        return f"{candidates[0].name}: 未显式链接 libcudart（静态链接）"

    import torch

    torch_major = (torch.version.cuda or "").split(".")[0]
    detail = f"{candidates[0].name} -> {libs[0]}, torch cuda={torch.version.cuda}"
    if torch_major and f"libcudart.so.{torch_major}" not in libs[0]:
        raise RuntimeError(f"vLLM 与 torch 的 CUDA 大版本不一致: {detail}")
    return detail


def _import_version(module_name: str) -> Callable[[], str]:
    def _check() -> str:
        import importlib

        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "(无 __version__)"))

    return _check


def _has_module(module_name: str) -> bool:
    """只看能不能找到，不 import——避免为一次存在性检查付出导入副作用。"""
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def _cogtrack_import() -> str:
    """确认 editable 安装生效，且核心接口可导入。"""

    from pathlib import Path

    import cogtrack
    from cogtrack.vlm import VLMBackend  # noqa: F401 - 确认核心接口可导入

    return f"from {Path(cogtrack.__file__).parent}"


def _parity_deps() -> str:
    """原版 SUTrack 那一侧的依赖。

    tools/verify_sutrack_parity.py 在同一个 torch 上同时跑原版 SUTrack 和本项目
    的实现来证明移植没引入误差，所以原版仓库的依赖也得齐。缺了不影响本项目的
    推理和训练，只会让那条证据链跑不起来，所以单独列一项而不是混进上面。
    """
    missing = [name for name in ("easydict", "lmdb", "pycocotools", "decord") if not _has_module(name)]
    if missing:
        raise RuntimeError(f"缺 {missing}，tools/verify_sutrack_parity.py 会 ModuleNotFoundError")
    return "easydict / lmdb / pycocotools / decord 齐备"


def _clip_tokenizer() -> str:
    """OpenAI CLIP：原版 SUTrack 的 use_nlp=True 分支用它做 caption tokenize。

    只 import 不够——BPE 词表是随包分发的数据文件，缺了要到真正 tokenize 时才
    炸。所以这里实际编一句话，并核对已知输出，顺带锁住词表没被换过。
    """
    import clip

    tokens = clip.tokenize(["a man riding a bicycle"])
    shape = tuple(tokens.shape)
    checksum = int(tokens.sum())
    if shape != (1, 77):
        raise RuntimeError(f"clip.tokenize 形状异常: {shape}")
    if checksum != 118656:
        raise RuntimeError(
            f"clip BPE 词表与核对 parity 时不一致 (sum={checksum}, 期望 118656)，"
            "parity 结论不可复现"
        )
    return f"tokenize OK, shape={shape} sum={checksum}"


CHECKS: List[Check] = [
    ("python", lambda: f"{platform.python_version()}  ({sys.executable})"),
    ("torch", _torch_info),
    ("gpu", _gpu_info),
    ("flash_attn import", _flash_attn_import),
    ("flash_attn kernel", _flash_attn_kernel),
    ("vllm", _vllm_info),
    ("vllm cuda 一致性", _vllm_cuda_major),
    ("transformers", _import_version("transformers")),
    ("ms-swift", _import_version("swift")),
    ("peft", _import_version("peft")),
    ("trl", _import_version("trl")),
    ("xformers", _import_version("xformers")),
    ("numpy", _import_version("numpy")),
    ("qwen_vl_utils", _import_version("qwen_vl_utils")),
    ("parity 依赖", _parity_deps),
    ("clip tokenizer", _clip_tokenizer),
    ("cogtrack", _cogtrack_import),
]


def main() -> int:
    failures: List[str] = []
    verbose = "--verbose" in sys.argv

    for label, check in CHECKS:
        try:
            print(f"  {label:22s} {check()}")
        except Exception as error:  # noqa: BLE001 - 汇总所有失败项再退出
            print(f"  {label:22s} 失败: {type(error).__name__}: {error}")
            if verbose:
                traceback.print_exc()
            failures.append(label)

    print()
    if failures:
        print(f"失败 {len(failures)}/{len(CHECKS)}: {failures}")
        print("加 --verbose 看完整 traceback。")
        return 1
    print(f"全部通过 ({len(CHECKS)} 项)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
