#!/usr/bin/env bash
# 合并 MGIT 官方分段标签和额外 teacher/verifier 标签，生成最终 state_update_sft。
#
# 用法：
#   bash scripts/build_state_update_sft_release.sh \
#     <release-name> <mgit-plan> <mgit-labels> <teacher-plan> <teacher-labels> <teacher-report>
#
# 两个 source plan 都必须是 fixed_identity_anchor。该脚本不会把候选 teacher 输出直接
# 送入训练：merge 工具会拒绝 independently_verified=false 或 dry-run 报告。

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

if [[ $# -ne 6 ]]; then
  echo "用法: bash scripts/build_state_update_sft_release.sh <release-name> <mgit-plan> <mgit-labels> <teacher-plan> <teacher-labels> <teacher-report>" >&2
  exit 2
fi

RELEASE_NAME=$1
MGIT_PLAN=$2
MGIT_LABELS=$3
TEACHER_PLAN=$4
TEACHER_LABELS=$5
TEACHER_REPORT=$6

OUT_DIR="data/releases/${RELEASE_NAME}"
PLAN_DIR="data/plans/${RELEASE_NAME}"
REPORT_DIR="data/reports/${RELEASE_NAME}"
MERGED_PLAN="$PLAN_DIR/state_update_sampling_plan.json"
MERGED_LABELS="$PLAN_DIR/state_update_labels.jsonl"
MERGE_REPORT="$REPORT_DIR/state_update_merge.json"
PYTHON=${PYTHON:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin/python"}
SEED=${SEED:-20260817}
MAX_CASES_PER_SEQUENCE=${MAX_CASES_PER_SEQUENCE:-10000}
VAL_RATIO=${VAL_RATIO:-0.05}
MGIT_VERSION=${MGIT_VERSION:-tiny}

mkdir -p "$PLAN_DIR" "$REPORT_DIR"

echo ">>> [1/4] 严格合并两类已验证状态标签"
"$PYTHON" tracking/merge_state_update_sft_data.py \
  --mgit-plan "$MGIT_PLAN" \
  --mgit-labels "$MGIT_LABELS" \
  --teacher-plan "$TEACHER_PLAN" \
  --teacher-labels "$TEACHER_LABELS" \
  --teacher-report "$TEACHER_REPORT" \
  --output-plan "$MERGED_PLAN" \
  --output-labels "$MERGED_LABELS" \
  --report "$MERGE_REPORT" \
  --seed "$SEED" \
  --max-cases-per-sequence "$MAX_CASES_PER_SEQUENCE"

ABSENT_RATIO=$("$PYTHON" - "$MERGED_PLAN" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(format(float(payload["requested_absent_ratio"]), ".17g"))
PY
)

echo ">>> [2/4] 构建统一固定三图 state_update_sft 训练视图"
"$PYTHON" tracking/synthesize_vlt_v6_dataset.py \
  --datasets lasot tnl2k mgit \
  --output-dir "$OUT_DIR" \
  --context-mode mosaic \
  --reference-mode visual_box \
  --reference-policy fixed_identity_anchor \
  --memory-supervision explicit \
  --memory-labels "$MERGED_LABELS" \
  --sft-supervision-profile state_update_sft \
  --sampling-plan "$MERGED_PLAN" \
  --max-samples-per-sequence "$MAX_CASES_PER_SEQUENCE" \
  --absent-ratio "$ABSENT_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --mgit-version "$MGIT_VERSION" \
  --allow-missing-mgit-sequences \
  --qwen-model-families qwen3_vl \
  --force

echo ">>> [3/4] 审计统一 release 的逐行全监督"
"$PYTHON" tracking/validate_sft_supervision.py \
  --dataset "$OUT_DIR/ms_swift/qwen3_vl/train.jsonl" \
  --dataset "$OUT_DIR/ms_swift/qwen3_vl/val.jsonl" \
  --profile state_update_sft \
  | tee "$REPORT_DIR/supervision_audit.txt"

echo ">>> [4/4] 写入合并 release 摘要"
"$PYTHON" - "$OUT_DIR/dataset_info.json" "$MERGE_REPORT" "$REPORT_DIR/release_summary.json" <<'PY'
import json
import sys
from pathlib import Path

dataset_info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
merge_report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
summary = {
    "schema_version": "cogtrack.state_update_sft_release.v1",
    "synthesis_profile": "vlt_v6",
    "sft_supervision_profile": "state_update_sft",
    "dataset_info": dataset_info,
    "merge": merge_report,
}
Path(sys.argv[3]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "完成统一 state_update_sft release：$OUT_DIR"
echo "MGIT 分段标签和约 1,500 条 teacher/verifier 标签已使用同一固定 identity anchor 合并。"
