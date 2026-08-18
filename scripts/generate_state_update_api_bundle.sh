#!/usr/bin/env bash
# 生成可通过 ModelScope 搬到远端的 state_update API 标注输入包。

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

RELEASE_NAME=${1:-}
if [[ -z "$RELEASE_NAME" ]]; then
  echo "用法: bash scripts/generate_state_update_api_bundle.sh <bundle-release-name>" >&2
  exit 2
fi

PYTHON=${PYTHON:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin/python"}
SEED=${SEED:-20260817}
PLAN_CASES_PER_SEQUENCE=${PLAN_CASES_PER_SEQUENCE:-40}
SEQUENCES_PER_DATASET=${SEQUENCES_PER_DATASET:-120}
STRIDE=${STRIDE:-2}
MAX_STEPS_PER_SEQUENCE=${MAX_STEPS_PER_SEQUENCE:-30}
MIN_ANCHOR_GAP=${MIN_ANCHOR_GAP:-120}
MAX_IMAGE_SIDE=${MAX_IMAGE_SIDE:-648}

PLAN_SOURCE="data/releases/${RELEASE_NAME}_candidate_plan"
PLAN_DIR="data/plans/${RELEASE_NAME}"
SELECTED_PLAN="$PLAN_DIR/api_annotation_sampling_plan.json"
RENDERED="data/annotation_work/${RELEASE_NAME}_rendered"
BUNDLE="data/annotation_bundles/${RELEASE_NAME}"

if [[ -e "$BUNDLE" ]]; then
  echo "错误：bundle 已存在，请使用新的 release 名：$BUNDLE" >&2
  exit 1
fi
mkdir -p "$PLAN_DIR"

echo ">>> [1/4] 固化 LaSOT/TNL2K 大候选计划"
"$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k \
  --output-dir "$PLAN_SOURCE" \
  --context-mode mosaic \
  --reference-mode visual_box \
  --reference-policy fixed_identity_anchor \
  --memory-supervision masked_null \
  --max-samples-per-sequence "$PLAN_CASES_PER_SEQUENCE" \
  --absent-ratio 0.20 \
  --val-ratio 0.05 \
  --seed "$SEED" \
  --qwen-model-families qwen3_vl \
  --plan-only \
  --force

echo ">>> [2/4] 选择状态、消失和重现决策链"
"$PYTHON" tracking/select_state_update_api_plan.py \
  --input-plan "$PLAN_SOURCE/sampling_plan.json" \
  --output-plan "$SELECTED_PLAN" \
  --datasets lasot,tnl2k \
  --sequences-per-dataset "$SEQUENCES_PER_DATASET" \
  --stride "$STRIDE" \
  --max-steps-per-sequence "$MAX_STEPS_PER_SEQUENCE" \
  --min-anchor-gap "$MIN_ANCHOR_GAP" \
  --seed "$SEED"

ABSENT_RATIO=$("$PYTHON" - "$SELECTED_PLAN" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(format(float(payload["requested_absent_ratio"]), ".17g"))
PY
)

echo ">>> [3/4] 渲染与正式三图范式一致的便携资产"
"$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k \
  --output-dir "$RENDERED" \
  --context-mode mosaic \
  --reference-mode visual_box \
  --reference-policy fixed_identity_anchor \
  --memory-supervision masked_null \
  --sampling-plan "$SELECTED_PLAN" \
  --max-samples-per-sequence "$MAX_STEPS_PER_SEQUENCE" \
  --absent-ratio "$ABSENT_RATIO" \
  --history-corruption-ratio 0 \
  --max-image-side "$MAX_IMAGE_SIDE" \
  --val-ratio 0.05 \
  --seed "$SEED" \
  --qwen-model-families qwen3_vl \
  --force

echo ">>> [4/4] 构造 ModelScope 可传输 annotation_input"
"$PYTHON" tracking/prepare_state_update_api_bundle.py \
  --source-release "$RENDERED" \
  --sampling-plan "$SELECTED_PLAN" \
  --output-dir "$BUNDLE"

echo "完成：$BUNDLE"
echo "该目录可直接上传到 ModelScope dataset repo；不包含 API 密钥或原始完整数据集。"
