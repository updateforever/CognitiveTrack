#!/usr/bin/env bash
# Qwen3-VL-4B tracking/mixed SFT：默认冻结 ViT，全参训练 LLM 与视觉对齐层。
# 必填：MODEL_PATH、TRAIN_DATA、VAL_DATA、DATASET_ROOT。

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export TUNER_TYPE=${TUNER_TYPE:-full}
export SFT_SUPERVISION_PROFILE=${SFT_SUPERVISION_PROFILE:-tracking_sft}
export FREEZE_VIT=${FREEZE_VIT:-true}
export FREEZE_LLM=${FREEZE_LLM:-false}
export FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
export TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
export ATTN_IMPL=${ATTN_IMPL:-flash_attn}
export MAX_LENGTH=${MAX_LENGTH:-4096}
export MAX_PIXELS=${MAX_PIXELS:-200704}
export EPOCHS=${EPOCHS:-1}
export BATCH_SIZE=${BATCH_SIZE:-1}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
if [[ "$TUNER_TYPE" == "full" ]]; then
    export LEARNING_RATE=${LEARNING_RATE:-1e-5}
    # 4×L40S 或 2×H100 默认 ZeRO-2；2×L40S 若 OOM 再显式改为 zero3。
    if [[ -z "${DEEPSPEED:-}" && -z "${FSDP_MODE:-}" ]]; then
        export DEEPSPEED=zero2
    fi
else
    export LEARNING_RATE=${LEARNING_RATE:-5e-5}
fi
export WARMUP_RATIO=${WARMUP_RATIO:-0.05}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
export LOGGING_STEPS=${LOGGING_STEPS:-5}
export SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
export EVAL_STRATEGY=${EVAL_STRATEGY:-epoch}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
export DATALOADER_WORKERS=${DATALOADER_WORKERS:-2}
export REPORT_TO=${REPORT_TO:-none}
export QWEN_MODEL_FAMILY=${QWEN_MODEL_FAMILY:-qwen3_vl}

exec bash "$SCRIPT_DIR/train_sft.sh" "$@"
