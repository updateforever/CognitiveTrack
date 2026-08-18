#!/usr/bin/env bash
# Build the self-contained v6.4 tracking/state-update mixed SFT release.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname -- "$SCRIPT_DIR")
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.conda/envs/cogtrack-l40/bin/python"}

TRACKING_ROOT=${TRACKING_ROOT:-"$PROJECT_ROOT/data/releases/cogtrack_vlt_v640_tracking_sft_r80_20_case20_mgit200_robust15_v1"}
STATE_ROOT=${STATE_ROOT:-"$PROJECT_ROOT/data/releases/cogtrack_vlt_v640_state_update_sft_combined_3063_v1"}
PREVIEW_ROOT=${PREVIEW_ROOT:-"$PROJECT_ROOT/data/previews/cogtrack_sft_v640_cases"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PROJECT_ROOT/data/releases/cogtrack_v640_mixed_sft_full_v1"}

exec "$PYTHON_BIN" "$PROJECT_ROOT/tracking/build_mixed_sft_release.py" \
    --tracking-root "$TRACKING_ROOT" \
    --state-root "$STATE_ROOT" \
    --preview-root "$PREVIEW_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    "$@"
