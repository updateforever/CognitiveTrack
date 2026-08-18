#!/usr/bin/env bash
# 生成额外约 1,500 条经双次教师 + 独立 verifier 裁决的状态更新标签。
# 这里只生成标签候选与审计报告，不直接改写 tracking_sft 或 MGIT release。

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
cd "$PROJECT_ROOT"

RELEASE_NAME=${1:-}
if [[ -z "$RELEASE_NAME" ]]; then
  echo "历史用法: bash tools/archive/state_teacher_v1/generate_state_update_teacher_labels.sh <label-release-name>" >&2
  exit 2
fi
: "${VERIFIER_MODEL:?请设置独立 verifier 权重路径 VERIFIER_MODEL}"

PLAN_DIR="data/plans/${RELEASE_NAME}"
REPORT_DIR="data/reports/${RELEASE_NAME}"
PLAN_PATH="$PLAN_DIR/teacher_fixed_anchor_sampling_plan.json"
LABEL_PATH="$PLAN_DIR/state_update_teacher_labels.jsonl"
TEACHER_REPORT="$REPORT_DIR/state_update_teacher_labels_report.json"
PYTHON=${PYTHON:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin/python"}
TEACHER_MODEL=${TEACHER_MODEL:-/root/public/models/Qwen/Qwen3-VL-32B-Instruct}
MAX_OUTPUT_LABELS=${MAX_OUTPUT_LABELS:-1500}
MIN_OUTPUT_LABELS=${MIN_OUTPUT_LABELS:-1200}
TEACHER_PLAN_CASES_PER_SEQUENCE=${TEACHER_PLAN_CASES_PER_SEQUENCE:-40}
SEED=${SEED:-20260817}

mkdir -p "$PLAN_DIR" "$REPORT_DIR"

echo ">>> [1/2] 固化 LaSOT/TNL2K 固定锚点教师候选计划"
"$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k \
  --output-dir "data/releases/${RELEASE_NAME}_plan_only" \
  --context-mode mosaic \
  --reference-mode visual_box \
  --reference-policy fixed_identity_anchor \
  --memory-supervision masked_null \
  --max-samples-per-sequence "$TEACHER_PLAN_CASES_PER_SEQUENCE" \
  --absent-ratio 0.20 \
  --val-ratio 0.05 \
  --seed "$SEED" \
  --qwen-model-families qwen3_vl \
  --plan-only \
  --force
cp "data/releases/${RELEASE_NAME}_plan_only/sampling_plan.json" "$PLAN_PATH"

echo ">>> [2/2] Qwen3-VL-32B 双次生成 + 独立 verifier，裁剪为约 $MAX_OUTPUT_LABELS 条"
"$PYTHON" tools/archive/state_teacher_v1/build_teacher_state_update_labels.py \
  --sampling-plan "$PLAN_PATH" \
  --output "$LABEL_PATH" \
  --report "$TEACHER_REPORT" \
  --teacher-model "$TEACHER_MODEL" \
  --verifier-model "$VERIFIER_MODEL" \
  --require-independent-verifier \
  --datasets lasot,tnl2k \
  --sequences-per-dataset "${SEQUENCES_PER_DATASET:-120}" \
  --stride "${TEACHER_STRIDE:-2}" \
  --max-steps-per-sequence "${MAX_STEPS_PER_SEQUENCE:-30}" \
  --min-planned-frames "${MIN_PLANNED_FRAMES:-8}" \
  --min-anchor-gap "${MIN_ANCHOR_GAP:-120}" \
  --max-output-labels "$MAX_OUTPUT_LABELS" \
  --min-output-labels "$MIN_OUTPUT_LABELS" \
  --seed "$SEED" \
  --max-image-side "${TEACHER_MAX_IMAGE_SIDE:-896}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --teacher-tensor-parallel-size "${TEACHER_TP_SIZE:-1}" \
  --verifier-tensor-parallel-size "${VERIFIER_TP_SIZE:-1}"

echo "完成额外 teacher/verifier 标签：$LABEL_PATH"
echo "这些标签仍需与 MGIT state_update 计划合并后，才能生成最终 state_update_sft release。"
