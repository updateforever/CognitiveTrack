#!/usr/bin/env bash
# 生成大规模 tracking SFT：presence、bbox、固定三图轨迹和错误历史鲁棒性。
# 本脚本不读取任何状态文本标签；MGIT action 分段留给 state_update SFT。
#
# 用法：
#   bash scripts/generate_tracking_sft_data.sh <release-name>
#   DRY_RUN=1 bash scripts/generate_tracking_sft_data.sh smoke_tracking_sft

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

RELEASE_NAME=${1:-}
if [[ -z "$RELEASE_NAME" ]]; then
  echo "用法: bash scripts/generate_tracking_sft_data.sh <release-name>" >&2
  exit 2
fi

OUT_DIR="data/releases/${RELEASE_NAME}"
PLAN_DIR="data/plans/${RELEASE_NAME}"
REPORT_DIR="data/reports/${RELEASE_NAME}"

DATASETS=${DATASETS:-"lasot tnl2k mgit"}
ABSENT_RATIO=${ABSENT_RATIO:-0.20}
MAX_CASES_PER_SEQ=${MAX_CASES_PER_SEQ:-20}
MGIT_CAP=${MGIT_CAP:-200}
HISTORY_CORRUPTION_RATIO=${HISTORY_CORRUPTION_RATIO:-0.15}
FRAME_STRIDE=${FRAME_STRIDE:-1}
VAL_RATIO=${VAL_RATIO:-0.05}
SEED=${SEED:-20260817}
MGIT_VERSION=${MGIT_VERSION:-tiny}
PYTHON=${PYTHON:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin/python"}
REUSE_EXISTING_ASSETS=${REUSE_EXISTING_ASSETS:-0}

BUILD_WRITE_ARGS=(--force)
if [[ "$REUSE_EXISTING_ASSETS" == "1" ]]; then
  if [[ -f "$OUT_DIR/source_samples.jsonl" || -f "$OUT_DIR/build_report.json" ]]; then
    echo "错误：恢复模式只适用于尚未完成 source_samples/build_report 的中断构建：$OUT_DIR" >&2
    exit 1
  fi
  BUILD_WRITE_ARGS=(--reuse-existing-assets)
fi

EXTRA_ARGS=()
if [[ -n "${DRY_RUN:-}" ]]; then
  DRY_RUN_SEQS=${DRY_RUN_SEQS:-12}
  EXTRA_ARGS+=(--limit-sequences-per-dataset "$DRY_RUN_SEQS")
  echo ">>> DRY_RUN：每个数据源最多 $DRY_RUN_SEQS 个序列"
fi

echo "=============================================================="
echo " VLT-v6.4 tracking SFT 数据生成"
echo "   输出目录       : $OUT_DIR"
echo "   数据源         : $DATASETS"
echo "   absent 比例    : $ABSENT_RATIO"
echo "   默认 case 上限 : $MAX_CASES_PER_SEQ"
echo "   MGIT case 上限 : $MGIT_CAP"
echo "   错误历史比例   : $HISTORY_CORRUPTION_RATIO"
echo "   复用已有图片   : $REUSE_EXISTING_ASSETS"
echo "   seed           : $SEED"
echo "=============================================================="

mkdir -p "$PLAN_DIR" "$REPORT_DIR"

PLAN_PATH="$OUT_DIR/sampling_plan.json"
echo
if [[ "$REUSE_EXISTING_ASSETS" == "1" && -f "$PLAN_PATH" ]]; then
  echo ">>> [1/4] 复用已固化的 reference/current 采样计划"
else
  echo ">>> [1/4] 固化 reference/current 采样计划"
  "$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
    --datasets $DATASETS \
    --output-dir "$OUT_DIR" \
    --context-mode mosaic \
    --reference-mode visual_box \
    --reference-policy sampled_prior_present \
    --memory-supervision masked_null \
    --frame-stride "$FRAME_STRIDE" \
    --max-samples-per-sequence "$MAX_CASES_PER_SEQ" \
    --max-samples-per-dataset "mgit=$MGIT_CAP" \
    --absent-ratio "$ABSENT_RATIO" \
    --history-corruption-ratio "$HISTORY_CORRUPTION_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --seed "$SEED" \
    --mgit-version "$MGIT_VERSION" \
    --allow-missing-mgit-sequences \
    --qwen-model-families qwen3_vl \
    --plan-only \
    --force \
    "${EXTRA_ARGS[@]}"
fi

if [[ ! -f "$PLAN_PATH" ]]; then
  echo "错误：未找到采样计划 $PLAN_PATH" >&2
  exit 1
fi
cp "$PLAN_PATH" "$PLAN_DIR/sampling_plan.json"

echo
echo ">>> [2/4] 重放计划并构建三图资产与 Qwen3-VL 训练视图"
"$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
  --datasets $DATASETS \
  --output-dir "$OUT_DIR" \
  --context-mode mosaic \
  --reference-mode visual_box \
  --reference-policy sampled_prior_present \
  --memory-supervision masked_null \
  --sampling-plan "$PLAN_PATH" \
  --frame-stride "$FRAME_STRIDE" \
  --max-samples-per-sequence "$MAX_CASES_PER_SEQ" \
  --max-samples-per-dataset "mgit=$MGIT_CAP" \
  --absent-ratio "$ABSENT_RATIO" \
  --history-corruption-ratio "$HISTORY_CORRUPTION_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --mgit-version "$MGIT_VERSION" \
  --allow-missing-mgit-sequences \
  --qwen-model-families qwen3_vl \
  "${BUILD_WRITE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

SOURCE_JSONL="$OUT_DIR/source_samples.jsonl"
BUILD_REPORT="$OUT_DIR/build_report.json"
TRAIN_JSONL="$OUT_DIR/ms_swift/qwen3_vl/train.jsonl"
VAL_JSONL="$OUT_DIR/ms_swift/qwen3_vl/val.jsonl"

echo
echo ">>> [3/4] 审计 tracking_sft loss 与三态边界"
"$PYTHON" tracking/validate_sft_supervision.py \
  --dataset "$TRAIN_JSONL" \
  --dataset "$VAL_JSONL" \
  --profile tracking_sft \
  | tee "$REPORT_DIR/supervision_audit.txt"

echo
echo ">>> [4/4] 审计 9 个主场景与 27 个有效视觉组合"
TAXONOMY_ARGS=()
if [[ -z "${DRY_RUN:-}" ]]; then
  TAXONOMY_ARGS+=(--require-complete-coverage)
fi
"$PYTHON" tracking/validate_tracking_sft_taxonomy.py \
  --dataset "$SOURCE_JSONL" \
  --build-report "$BUILD_REPORT" \
  "${TAXONOMY_ARGS[@]}" \
  | tee "$REPORT_DIR/taxonomy_audit.json"

{
  echo "release: $RELEASE_NAME"
  echo "generated_at: $(date -Iseconds)"
  echo "prompt_version: 6.4.0"
  echo "supervision_profile: tracking_sft"
  echo "seed: $SEED"
  for path in "$PLAN_PATH" "$SOURCE_JSONL" "$TRAIN_JSONL" "$VAL_JSONL"; do
    printf '%s  rows=%s  sha256=%s\n' \
      "$path" "$(wc -l < "$path")" "$(sha256sum "$path" | cut -d' ' -f1)"
  done
} | tee "$REPORT_DIR/artifacts.txt"

echo
echo "完成：$OUT_DIR"
echo "该 release 只负责 tracking_sft；MGIT 分段文本未混入本数据。"
