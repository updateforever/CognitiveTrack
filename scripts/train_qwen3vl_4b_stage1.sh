#!/usr/bin/env bash
# Qwen3-VL-4B Stage-1 SFT：默认采用 LoRA，冻结视觉塔；如需复现实验性全参方案，
# 显式设置 TUNER_TYPE=full 并提供 FSDP2/ZeRO-3 分片。
#
# 必填：MODEL_PATH、TRAIN_DATA、VAL_DATA、DATASET_ROOT。

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export TUNER_TYPE=${TUNER_TYPE:-lora}
export FSDP_MODE=${FSDP_MODE:-fsdp2}
export FREEZE_LLM=${FREEZE_LLM:-false}
export FREEZE_VIT=${FREEZE_VIT:-true}
export FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export ATTN_IMPL=${ATTN_IMPL:-flash_attn}
export MAX_LENGTH=${MAX_LENGTH:-2048}
export MAX_PIXELS=${MAX_PIXELS:-200704}
export EPOCHS=${EPOCHS:-1}
# 4B 先用保守 batch=2；真实冒烟后再决定是否提高。
export BATCH_SIZE=${BATCH_SIZE:-2}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
# LoRA 比全参更新可使用更高学习率；5e-5 在一轮 16 万样本上保持保守。
export LEARNING_RATE=${LEARNING_RATE:-5e-5}
export WARMUP_RATIO=${WARMUP_RATIO:-0.05}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
export LOGGING_STEPS=${LOGGING_STEPS:-5}
export SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
export EVAL_STRATEGY=${EVAL_STRATEGY:-epoch}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
export DATALOADER_WORKERS=${DATALOADER_WORKERS:-2}
export REPORT_TO=${REPORT_TO:-none}

exec bash "$SCRIPT_DIR/train_sft.sh" "$@"
