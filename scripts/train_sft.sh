#!/usr/bin/env bash
# 使用 ms-swift 对本地 Qwen-VL 做 LoRA 或全参数 SFT。
#
# 必填环境变量：
#   MODEL_PATH=/path/to/Qwen-VL
#   TRAIN_DATA=/path/to/train.jsonl
# 可选环境变量：VAL_DATA、OUTPUT_DIR、DATASET_ROOT、MS_SWIFT_ENV，以及下方
# 所有训练超参数。命令行追加参数会原样传给 ms-swift，便于覆盖实验配置。

set -euo pipefail

: "${MODEL_PATH:?请设置本地模型目录 MODEL_PATH}"
: "${TRAIN_DATA:?请设置 SFT train.jsonl 路径 TRAIN_DATA}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
OUTPUT_DIR=${OUTPUT_DIR:-"$PROJECT_ROOT/outputs/sft_qwen_vl"}
VAL_DATA=${VAL_DATA:-}
DATASET_ROOT=${DATASET_ROOT:-}
TUNER_TYPE=${TUNER_TYPE:-lora}
if [[ "$TUNER_TYPE" == "full" ]]; then
    DEFAULT_EPOCHS=1
    DEFAULT_LEARNING_RATE=1e-5
else
    DEFAULT_EPOCHS=3
    DEFAULT_LEARNING_RATE=1e-4
fi

# CognitiveTrack 的 Qwen 数据使用官方 cookbook JSON bbox 表示；禁止回退到
# legacy 特殊 token。坐标本身由 ms-swift 按模型族处理：2.5=绝对像素，3=norm1000。
export QWENVL_BBOX_FORMAT=${QWENVL_BBOX_FORMAT:-new}

# 在加载模型、占用 GPU 之前检查模型代际与数据坐标视图。Qwen2.5-VL 和
# Qwen3-VL 的坐标协议不同，二者 JSONL 不能交叉使用。
FAMILY_CHECK=(
    python "$PROJECT_ROOT/tracking/validate_qwen_training_view.py"
    --model "$MODEL_PATH"
    --dataset "$TRAIN_DATA"
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

# 优先使用当前环境中的 swift；否则回退到项目开发时验证过的 Conda 环境。
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

ARGS=(
    sft
    --model "$MODEL_PATH"
    --dataset "$TRAIN_DATA"
    --output_dir "$OUTPUT_DIR"
    --tuner_type "$TUNER_TYPE"
    --torch_dtype "${TORCH_DTYPE:-bfloat16}"
    --attn_impl "${ATTN_IMPL:-flash_attn}"
    --max_length "${MAX_LENGTH:-4096}"
    # 两图训练必须显式限制每张图的视觉 token；否则原始高清视频会造成显存
    # 波动，且与在线 max_image_side=648 的量级差异过大。
    --max_pixels "${MAX_PIXELS:-200704}"
    --num_train_epochs "${EPOCHS:-$DEFAULT_EPOCHS}"
    --per_device_train_batch_size "${BATCH_SIZE:-1}"
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRAD_ACC_STEPS:-16}"
    --learning_rate "${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
    --warmup_ratio "${WARMUP_RATIO:-0.05}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
    --logging_steps "${LOGGING_STEPS:-5}"
    --save_strategy "${SAVE_STRATEGY:-steps}"
    --save_steps "${SAVE_STEPS:-200}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
    --dataset_num_proc "${DATASET_NUM_PROC:-4}"
    --dataloader_num_workers "${DATALOADER_WORKERS:-4}"
    --report_to "${REPORT_TO:-none}"
)

if [[ "$TUNER_TYPE" == "lora" ]]; then
    ARGS+=(
        --target_modules all-linear
        --lora_rank "${LORA_RANK:-16}"
        --lora_alpha "${LORA_ALPHA:-32}"
        --lora_dropout "${LORA_DROPOUT:-0.05}"
        --freeze_vit "${FREEZE_VIT:-true}"
    )
elif [[ "$TUNER_TYPE" == "full" ]]; then
    # VLM 常规全参 SFT：语言模型与对齐层全参训练，默认冻结视觉编码器，保护其
    # 既有视觉/grounding 能力。若要连视觉塔一起训练，再显式设 FREEZE_VIT=false。
    # 8×24GB 仍需设置 DEEPSPEED=zero3 或 FSDP_MODE=fsdp2 之一。
    ARGS+=(
        --freeze_llm "${FREEZE_LLM:-false}"
        --freeze_vit "${FREEZE_VIT:-true}"
        --freeze_aligner "${FREEZE_ALIGNER:-false}"
    )
    if [[ -n "${DEEPSPEED:-}" && -n "${FSDP_MODE:-}" ]]; then
        echo "错误：DEEPSPEED 与 FSDP_MODE 只能选择一种分片方案。" >&2
        exit 1
    fi
    if [[ -n "${DEEPSPEED:-}" ]]; then
        ARGS+=(--deepspeed "$DEEPSPEED")
    elif [[ -n "${FSDP_MODE:-}" ]]; then
        ARGS+=(--fsdp "$FSDP_MODE")
        if [[ -n "${FSDP_CONFIG:-}" ]]; then
            ARGS+=(--fsdp_config "$FSDP_CONFIG")
        fi
    else
        echo "错误：全量微调必须设置 DEEPSPEED=zero3 或 FSDP_MODE=fsdp2。" >&2
        exit 1
    fi
else
    echo "错误：TUNER_TYPE 目前只支持 lora 或 full，实际为 $TUNER_TYPE。" >&2
    exit 1
fi

if [[ -n "$VAL_DATA" ]]; then
    EVAL_STRATEGY=${EVAL_STRATEGY:-steps}
    ARGS+=(--val_dataset "$VAL_DATA" --eval_strategy "$EVAL_STRATEGY")
    if [[ "$EVAL_STRATEGY" == "steps" ]]; then
        ARGS+=(--eval_steps "${EVAL_STEPS:-200}")
    fi
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$OUTPUT_DIR"
if [[ -n "$DATASET_ROOT" ]]; then
    # 相对 images 路径以数据集根目录解析；TRAIN_DATA/VAL_DATA 建议传绝对路径。
    cd "$DATASET_ROOT"
fi
echo "[CognitiveTrack] 启动 SFT：model=$MODEL_PATH data=$TRAIN_DATA output=$OUTPUT_DIR"
exec "${SWIFT_COMMAND[@]}" "${ARGS[@]}" "$@"
