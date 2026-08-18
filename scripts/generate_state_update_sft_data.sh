#!/usr/bin/env bash
# 生成官方 MGIT 分段来源的 state_update_sft。
#
# 该入口只负责“全量可靠 MGIT 分段 + present hard-null”这一部分。额外约 1,500 条
# teacher/verifier 标签必须在独立 verifier 通过后，使用同一固定锚点计划合并；未经
# verifier 的 teacher 输出不会被本脚本自动混入训练集。

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

RELEASE_NAME=${1:-}
if [[ -z "$RELEASE_NAME" ]]; then
  echo "用法: bash scripts/generate_state_update_sft_data.sh <release-name>" >&2
  exit 2
fi

OUT_DIR="data/releases/${RELEASE_NAME}"
PLAN_DIR="data/plans/${RELEASE_NAME}"
REPORT_DIR="data/reports/${RELEASE_NAME}"
MGIT_VERSION=${MGIT_VERSION:-tiny}
BOUNDARY_MARGIN=${BOUNDARY_MARGIN:-30}
HARD_NULL_PER_SEGMENT=${HARD_NULL_PER_SEGMENT:-1}
ABSENT_PER_RUN=${ABSENT_PER_RUN:-0}
MAX_CASES_PER_SEQUENCE=${MAX_CASES_PER_SEQUENCE:-10000}
SEED=${SEED:-20260817}
VAL_RATIO=${VAL_RATIO:-0.05}
PYTHON=${PYTHON:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin/python"}

EXTRA_ARGS=()
if [[ -n "${LIMIT_SEQUENCES:-}" ]]; then
  EXTRA_ARGS+=(--limit-sequences "$LIMIT_SEQUENCES")
fi

mkdir -p "$PLAN_DIR" "$REPORT_DIR"
PLAN_PATH="$PLAN_DIR/mgit_state_update_sampling_plan.json"
LABELS_PATH="$PLAN_DIR/mgit_state_update_labels.jsonl"

echo ">>> [1/4] 挖掘 MGIT 全部可靠分段更新与稳定 hard-null"
"$PYTHON" tracking/plan_mgit_state_update_data.py \
  --output-plan "$PLAN_PATH" \
  --output-labels "$LABELS_PATH" \
  --report "$REPORT_DIR/mgit_state_update_mining.json" \
  --mgit-version "$MGIT_VERSION" \
  --boundary-margin "$BOUNDARY_MARGIN" \
  --hard-null-per-segment "$HARD_NULL_PER_SEGMENT" \
  --absent-per-run "$ABSENT_PER_RUN" \
  --max-cases-per-sequence "$MAX_CASES_PER_SEQUENCE" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}"

echo ">>> [2/4] 构建固定三图 state_update_sft 训练视图"
"$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
  --datasets mgit \
  --output-dir "$OUT_DIR" \
  --context-mode mosaic \
  --reference-mode visual_box \
  --reference-policy fixed_identity_anchor \
  --memory-supervision explicit \
  --memory-labels "$LABELS_PATH" \
  --sft-supervision-profile state_update_sft \
  --sampling-plan "$PLAN_PATH" \
  --max-samples-per-sequence "$MAX_CASES_PER_SEQUENCE" \
  --absent-ratio "$ABSENT_PER_RUN" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --mgit-version "$MGIT_VERSION" \
  --allow-missing-mgit-sequences \
  --qwen-model-families qwen3_vl \
  --force

echo ">>> [3/4] 审计 state_update_sft 的逐行全监督"
"$PYTHON" tracking/validate_sft_supervision.py \
  --dataset "$OUT_DIR/ms_swift/qwen3_vl/train.jsonl" \
  --dataset "$OUT_DIR/ms_swift/qwen3_vl/val.jsonl" \
  --profile state_update_sft \
  | tee "$REPORT_DIR/supervision_audit.txt"

echo ">>> [4/4] 固化 release checksum 与行数"
{
  echo "release: $RELEASE_NAME"
  echo "generated_at: $(date -Iseconds)"
  echo "prompt_version: 6.4.0"
  echo "supervision_profile: state_update_sft"
  for path in \
    "$PLAN_PATH" \
    "$LABELS_PATH" \
    "$OUT_DIR/source_samples.jsonl" \
    "$OUT_DIR/ms_swift/qwen3_vl/train.jsonl" \
    "$OUT_DIR/ms_swift/qwen3_vl/val.jsonl"; do
    printf '%s  rows=%s  sha256=%s\n' \
      "$path" "$(wc -l < "$path")" "$(sha256sum "$path" | cut -d' ' -f1)"
  done
} | tee "$REPORT_DIR/artifacts.txt"

echo
echo "完成 MGIT state_update_sft：$OUT_DIR"
echo "下一步：将独立 verifier 通过的额外约 1,500 条 teacher 标签按同一固定锚点计划合并；"
echo "候选输出（reviewed=false）不得直接进入该 release。"
