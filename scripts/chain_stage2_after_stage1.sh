#!/usr/bin/env bash
# Wait for a successful Stage-1 run and a validated Stage-2 release, then continue
# training the same LoRA adapter on Stage-2.

set -euo pipefail

: "${STAGE1_PID:?set STAGE1_PID to the active top-level swift process}"
: "${STAGE2_BUILD_PID:?set STAGE2_BUILD_PID to the active data builder process}"
: "${STAGE1_OUTPUT:?set STAGE1_OUTPUT}"
: "${STAGE2_DATASET_ROOT:?set STAGE2_DATASET_ROOT}"
: "${STAGE2_OUTPUT:?set STAGE2_OUTPUT}"
: "${MODEL_PATH:?set MODEL_PATH}"

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_BIN=${ENV_BIN:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin"}
EXPECTED_STAGE1_STEPS=${EXPECTED_STAGE1_STEPS:-19005}
POLL_SECONDS=${POLL_SECONDS:-60}

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

log "waiting for Stage-1 pid=$STAGE1_PID"
while kill -0 "$STAGE1_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done

STAGE1_RUN=$(find "$STAGE1_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'v*-*' | sort | tail -n 1)
if [[ -z "$STAGE1_RUN" || ! -f "$STAGE1_RUN/logging.jsonl" ]]; then
    log "Stage-1 logging.jsonl not found; refusing to start Stage-2"
    exit 1
fi
"$ENV_BIN/python" - "$STAGE1_RUN/logging.jsonl" "$EXPECTED_STAGE1_STEPS" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
completed = False
with open(path, encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        if record.get("global_step") == expected and "model_parameter_info" in record:
            completed = True
if not completed:
    raise SystemExit(f"Stage-1 did not finish at global_step={expected}")
PY

log "Stage-1 completed; waiting for Stage-2 dataset"
while [[ ! -f "$STAGE2_DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl" || \
         ! -f "$STAGE2_DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl" ]]; do
    if rg -q '^\[错误\]' "$STAGE2_DATASET_ROOT/build.log" 2>/dev/null; then
        log "Stage-2 data build failed; refusing to train"
        exit 1
    fi
    if ! kill -0 "$STAGE2_BUILD_PID" 2>/dev/null; then
        log "Stage-2 builder exited without complete JSONL; refusing to train"
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

"$ENV_BIN/python" "$PROJECT_ROOT/tracking/validate_qwen_training_view.py" \
    --model "$MODEL_PATH" \
    --dataset "$STAGE2_DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl" \
    --dataset "$STAGE2_DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl" \
    --expected-family qwen3_vl

STAGE1_ADAPTER=$(find "$STAGE1_RUN" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    | sort -V | tail -n 1)
if [[ -z "$STAGE1_ADAPTER" || ! -f "$STAGE1_ADAPTER/adapter_model.safetensors" ]]; then
    log "no complete Stage-1 adapter checkpoint found"
    exit 1
fi

mkdir -p "$STAGE2_OUTPUT"
log "starting Stage-2 from adapter=$STAGE1_ADAPTER"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export MASTER_PORT=${MASTER_PORT:-29520}
export PATH="$ENV_BIN:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export SWIFT_BIN="$ENV_BIN/swift"
export DATASET_ROOT="$STAGE2_DATASET_ROOT"
export TRAIN_DATA="$STAGE2_DATASET_ROOT/ms_swift/qwen3_vl/train.jsonl"
export VAL_DATA="$STAGE2_DATASET_ROOT/ms_swift/qwen3_vl/val.jsonl"
export OUTPUT_DIR="$STAGE2_OUTPUT"
export ADAPTERS="$STAGE1_ADAPTER"
export TUNER_TYPE=lora
export BATCH_SIZE=${BATCH_SIZE:-3}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export EPOCHS=${EPOCHS:-1}
export MAX_PIXELS=${MAX_PIXELS:-200704}
export LOGGING_STEPS=${LOGGING_STEPS:-20}
export SAVE_STRATEGY=steps
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
export EVAL_STRATEGY=no
export REPORT_TO=none

exec bash "$PROJECT_ROOT/scripts/train_qwen3vl_4b_stage1.sh" \
    --save_strategy steps --save_steps "$SAVE_STEPS" --eval_strategy no
