#!/usr/bin/env bash
# CognitiveTrack 环境配方：从零建一个 torch2.8 + vLLM + flash-attn + ms-swift 的 conda 环境。
#
# 这个脚本刻意不去改动已有环境，而是新建。原因是原地升降级出来的环境依赖
# 它自己的历史，换台机器复现不出来；从零装一遍跑通，配方本身才算被验证过。
#
# ---------------------------------------------------------------------------
# 为什么钉这一套版本
#
# vLLM 把 torch 钉成 ``==`` 而不是 ``>=``，所以 `pip install vllm` 会直接替换
# 已装的 torch，然后 xformers / torchvision / torchaudio 全部 ABI 失配。这是
# 本环境唯一真正的难点。对策是把所有硬钉包放进同一条 pip 命令，让 resolver
# 一次看到全部约束，不留中间态。
#
# 版本从 flash-attn 反推（v2.8.3.post1 的 50 个官方 wheel 实测覆盖情况）：
#
#     cp310 + cu12 -> torch 2.4 / 2.5 / 2.6 / 2.7 / 2.8
#     torch 2.9    -> 只有 cu13 + cp312 这一个 wheel
#
# 所以只要还在 Python 3.10 + CUDA 12 上，torch 上限就是 2.8：
#
#     torch 2.8.0  -> vLLM 0.11.0（0.11.2 钉 torch==2.9.0）
#                  -> xformers 0.0.32.post1 / torchvision 0.23.0 / torchaudio 2.8.0
#
# 不要试图用 torch2.9 + cu13 的 flash-attn wheel 去配 vLLM：vLLM 的
# _C.abi3.so 链接 libcudart.so.12，是 CUDA 12 编译的，撞的是 CUDA 大版本，
# 换 Python 版本解决不了。
#
# 另外，PyPI 上的 torch 2.8.0 本身就是 cu128 构建（依赖里写着
# nvidia-cuda-runtime-cu12==12.8.90），所以不需要 --extra-index-url，也不需要
# +cu128 本地版本号。少一个来源就少一类解析歧义。
# ---------------------------------------------------------------------------
#
# 用法：
#   bash scripts/setup_env.sh                    # 建 cogtrack28
#   ENV_NAME=myenv bash scripts/setup_env.sh     # 换名字
#   DRY_RUN=1 bash scripts/setup_env.sh          # 只打印命令，不执行

set -euo pipefail

ENV_NAME="${ENV_NAME:-cogtrack28}"
DRY_RUN="${DRY_RUN:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 版本区（改这里，不要改下面的逻辑）----------------------------------
PYTHON_VERSION="3.10"
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
XFORMERS_VERSION="0.0.32.post1"
VLLM_VERSION="0.11.0"
TRANSFORMERS_VERSION="4.57.1"
MS_SWIFT_VERSION="4.3.1"
FLASH_ATTN_VERSION="2.8.3.post1"

# flash-attn wheel 的 tag 必须同时匹配 Python、torch 和 CUDA 大版本。
FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL//+/%2B}"
# ------------------------------------------------------------------------

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

# CLAUDE.md 记录过 PYTHONPATH / PIP_TARGET / PYTHONUSERBASE 污染导致的排查噩梦。
# 这几个变量会让 pip 把包装到环境外面去，症状是「装了但 import 不到」。
for var in PYTHONPATH PIP_TARGET PYTHONUSERBASE; do
    if [[ -n "${!var:-}" ]]; then
        echo "检测到 ${var}=${!var}，已在本脚本内 unset。" >&2
    fi
done
unset PYTHONPATH PIP_TARGET PYTHONUSERBASE || true

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

echo "=== 目标 ==="
echo "  环境名      ${ENV_NAME}"
echo "  路径        ${ENV_PREFIX}"
echo "  python      ${PYTHON_VERSION}"
echo "  torch       ${TORCH_VERSION}  (PyPI 版本即 cu128 构建)"
echo "  vllm        ${VLLM_VERSION}"
echo "  flash-attn  ${FLASH_ATTN_VERSION}"
echo "  项目根       ${PROJECT_ROOT}"
echo

if [[ -d "${ENV_PREFIX}" && "${DRY_RUN}" != "1" ]]; then
    echo "环境 ${ENV_NAME} 已存在：${ENV_PREFIX}" >&2
    echo "换个 ENV_NAME，或先 conda env remove -n ${ENV_NAME}。" >&2
    exit 1
fi

echo "=== 1. 创建空环境 ==="
# 用 conda-forge 而不是 defaults：Anaconda 的 defaults 频道现在要求显式接受
# ToS（含商业使用授权条款），conda-forge 没有这个要求。这里只需要一个干净的
# Python 解释器，其余全部走 pip，所以频道选择不影响后面任何版本决策。
run conda create -y -n "${ENV_NAME}" \
    --override-channels -c conda-forge \
    "python=${PYTHON_VERSION}" pip

PY="${ENV_PREFIX}/bin/python"
if [[ "${DRY_RUN}" != "1" && ! -x "${PY}" ]]; then
    echo "conda create 之后找不到 ${PY}" >&2
    exit 1
fi

echo
echo "=== 2. 一次性安装全部硬钉包 ==="
# 关键：torch / torchvision / torchaudio / xformers / vllm / transformers 必须
# 同一条命令。分开装会让 vLLM 覆盖 torch，这正是这个环境最容易踩的坑。
# ms-swift 也放进来：它自己不钉 torch，但会拖 peft/trl/accelerate，一起解析
# 才能保证它们挑的版本和这套 torch 相容。
run "${PY}" -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "xformers==${XFORMERS_VERSION}" \
    "vllm==${VLLM_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "ms-swift==${MS_SWIFT_VERSION}" \
    "qwen-vl-utils>=0.0.14" \
    "accelerate>=0.30" \
    "timm>=0.9" \
    "opencv-python>=4.8" \
    "Pillow>=10.0" \
    "PyYAML>=6.0" \
    "tqdm>=4.66" \
    "pytest>=8.0" \
    "pytest-cov>=5.0" \
    "ruff>=0.8" \
    "easydict==1.13" \
    "lmdb==2.2.1" \
    "pycocotools==2.0.11" \
    "decord==0.6.0" \
    "tabulate==0.10.0" \
    "loguru==0.7.3" \
    "ftfy==6.3.1"
# 最后那七个不是 CognitiveTrack 本身要的，是 tools/verify_sutrack_parity.py 跑
# 「原版 SUTrack」那一侧要的：该脚本在同一个 torch 上同时跑两套实现来证明移植
# 没引入误差，所以原版仓库的依赖也得在这个环境里齐。缺了只会在 parity 时报
# ModuleNotFoundError，正常推理/训练不受影响，但那条证据链就断了。

echo
echo "=== 3. 安装 flash-attn 预编译 wheel ==="
# --no-deps：wheel 的 metadata 会声明 torch 依赖，装依赖有可能再次动 torch。
# 此时 torch 已经是正确版本，不需要它插手。
run "${PY}" -m pip install --no-deps "${FLASH_ATTN_URL}"

echo
echo "=== 3.5 安装 OpenAI CLIP（parity 用，非 PyPI 包）==="
# 原版 SUTrack 的 use_nlp=True 分支 import clip 做 caption tokenize。这个包不在
# PyPI 上（PyPI 的 clip 是另一个无关项目），只能从 GitHub 装，因此单独一步。
# --no-deps：它的 setup.py 会拖 torch，此时 torch 已经是正确版本。
# 钉 commit 而不用 main：tokenizer 的 BPE 词表变了会让 parity 结论不可复现。
if "${PY}" -c "import clip" >/dev/null 2>&1; then
    echo "clip 已存在，跳过"
else
    run "${PY}" -m pip install --no-deps \
        "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
fi

echo
echo "=== 4. 以 editable 装入本项目 ==="
# --no-deps：依赖已在第 2 步按钉死版本装好，不让 setuptools 用 >= 约束再解一遍。
run "${PY}" -m pip install -e "${PROJECT_ROOT}" --no-deps --no-build-isolation

echo
echo "=== 5. 验证 ==="
if [[ "${DRY_RUN}" != "1" ]]; then
    "${PY}" "${PROJECT_ROOT}/scripts/verify_env.py"
fi

echo
echo "完成。用法："
echo "  conda activate ${ENV_NAME}"
echo "  # 或不激活直接用: ${PY}"
echo
echo "SUTrack 逐帧一致性此前是在 cogtrack (torch 2.9.0) 上核对的，"
echo "换到本环境后建议重跑确认结论不变、并记录新的绝对数字："
echo "  ${PY} tools/verify_sutrack_parity.py all --dataset videocube --frames 400"
