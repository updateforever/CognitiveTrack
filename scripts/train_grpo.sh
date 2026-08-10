#!/usr/bin/env bash
# 使用 ms-swift GRPO 优化认知跟踪结构化输出。
#
# GRPO_DATA 必须由 tracking/export_swift_dataset.py --mode grpo 生成：输入
# messages 不包含 assistant 答案，参考答案位于 solution。默认启用四个独立
# reward：格式、presence、bbox IoU、内部一致性。

set -euo pipefail

: "${MODEL_PATH:?请设置本地模型目录 MODEL_PATH}"
: "${GRPO_DATA:?请设置 GRPO train.jsonl 路径 GRPO_DATA}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
REWARD_PLUGIN="$PROJECT_ROOT/cogtrack/training/grpo_rewards.py"
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/grpo_qwen_vl"}
VAL_DATA=${VAL_DATA:-}
DATASET_ROOT=${DATASET_ROOT:-}

# 与 SFT 保持同一官方 Grounding 编码，避免 adapter 前后改变 bbox token 格式。
export QWENVL_BBOX_FORMAT=${QWENVL_BBOX_FORMAT:-new}

FAMILY_CHECK=(
    python "$PROJECT_ROOT/tracking/validate_qwen_training_view.py"
    --model "$MODEL_PATH"
    --dataset "$GRPO_DATA"
)
if [[ -n "$VAL_DATA" ]]; then
    FAMILY_CHECK+=(--dataset "$VAL_DATA")
fi
if [[ -n "${QWEN_MODEL_FAMILY:-}" ]]; then
    FAMILY_CHECK+=(--expected-family "$QWEN_MODEL_FAMILY")
fi
if [[ -n "$DATASET_ROOT" ]]; then
    (cd "$DATASET_ROOT" && "${FAMILY_CHECK[@]}")
else
    "${FAMILY_CHECK[@]}"
fi

if [[ -n "${SWIFT_BIN:-}" ]]; then
    SWIFT_COMMAND=("$SWIFT_BIN")
elif command -v swift >/dev/null 2>&1; then
    SWIFT_COMMAND=(swift)
elif command -v conda >/dev/null 2>&1; then
    SWIFT_COMMAND=(conda run --no-capture-output -n "${MS_SWIFT_ENV:-cogtrack}" swift)
else
    echo "错误：找不到 swift 或 conda，请先安装/激活 ms-swift。" >&2
    exit 1
fi

read -r -a REWARD_WEIGHT_ARGS <<< "${REWARD_WEIGHTS:-0.5 1.5 2.0 0.5}"
if [[ "${#REWARD_WEIGHT_ARGS[@]}" -ne 4 ]]; then
    echo "错误：REWARD_WEIGHTS 必须包含 4 个空格分隔的权重。" >&2
    exit 1
fi

ARGS=(
    rlhf
    --rlhf_type grpo
    --model "$MODEL_PATH"
    --dataset "$GRPO_DATA"
    --output_dir "$OUTPUT_DIR"
    --external_plugins "$REWARD_PLUGIN"
    --reward_funcs cogtrack_format cogtrack_presence cogtrack_bbox cogtrack_consistency
    --reward_weights "${REWARD_WEIGHT_ARGS[@]}"
    --tuner_type lora
    --target_modules all-linear
    --lora_rank "${LORA_RANK:-16}"
    --lora_alpha "${LORA_ALPHA:-32}"
    --freeze_vit "${FREEZE_VIT:-true}"
    --torch_dtype "${TORCH_DTYPE:-bfloat16}"
    --max_length "${MAX_LENGTH:-4096}"
    --max_completion_length "${MAX_COMPLETION_LENGTH:-256}"
    --num_generations "${NUM_GENERATIONS:-4}"
    --temperature "${TEMPERATURE:-0.7}"
    --top_p "${TOP_P:-0.9}"
    --beta "${KL_BETA:-0.04}"
    --num_train_epochs "${EPOCHS:-1}"
    --per_device_train_batch_size "${BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRAD_ACC_STEPS:-8}"
    --learning_rate "${LEARNING_RATE:-5e-6}"
    --gradient_checkpointing true
    --logging_steps "${LOGGING_STEPS:-1}"
    --save_steps "${SAVE_STEPS:-100}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
    --dataset_num_proc "${DATASET_NUM_PROC:-4}"
    --dataloader_num_workers "${DATALOADER_WORKERS:-4}"
    --report_to "${REPORT_TO:-none}"
)

if [[ -n "$VAL_DATA" ]]; then
    ARGS+=(--val_dataset "$VAL_DATA" --eval_strategy steps --eval_steps "${EVAL_STEPS:-100}")
fi
if [[ -n "${ADAPTERS:-}" ]]; then
    ARGS+=(--adapters "$ADAPTERS")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR"
if [[ -n "$DATASET_ROOT" ]]; then
    cd "$DATASET_ROOT"
fi
echo "[CognitiveTrack] 启动 GRPO：model=$MODEL_PATH data=$GRPO_DATA output=$OUTPUT_DIR"
exec "${SWIFT_COMMAND[@]}" "${ARGS[@]}" "$@"
