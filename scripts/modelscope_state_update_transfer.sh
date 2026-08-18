#!/usr/bin/env bash
# 通过 ModelScope dataset repo 上传/下载 annotation input 或 result。

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
用法：
  bash scripts/modelscope_state_update_transfer.sh upload <owner/repo> <local-dir> <path-in-repo>
  bash scripts/modelscope_state_update_transfer.sh download <owner/repo> <local-dir> <path-in-repo>

认证使用 MODELSCOPE_API_TOKEN 或预先执行 modelscope login；脚本不会打印 token。
EOF
  exit 2
}

ACTION=${1:-}
REPO_ID=${2:-}
LOCAL_DIR=${3:-}
REMOTE_PATH=${4:-}
if [[ -z "$ACTION" || -z "$REPO_ID" || -z "$LOCAL_DIR" || -z "$REMOTE_PATH" ]]; then
  usage
fi

case "$ACTION" in
  upload)
    if [[ ! -d "$LOCAL_DIR" || ! -f "$LOCAL_DIR/SHA256SUMS" ]]; then
      echo "上传目录缺少 SHA256SUMS：$LOCAL_DIR" >&2
      exit 1
    fi
    (cd "$LOCAL_DIR" && sha256sum -c SHA256SUMS)
    modelscope upload "$REPO_ID" "$LOCAL_DIR" "$REMOTE_PATH" \
      --repo-type dataset \
      --max-workers "${MODELSCOPE_MAX_WORKERS:-8}" \
      --commit-message "CognitiveTrack state-update annotation transfer"
    ;;
  download)
    if [[ -e "$LOCAL_DIR" ]]; then
      echo "下载目标已存在，请指定新目录：$LOCAL_DIR" >&2
      exit 1
    fi
    modelscope download "$REPO_ID" \
      --repo-type dataset \
      --local-dir "$LOCAL_DIR" \
      --include "$REMOTE_PATH/**" \
      --max-workers "${MODELSCOPE_MAX_WORKERS:-8}"
    CHECKSUM=$(find "$LOCAL_DIR" -path "*/$REMOTE_PATH/SHA256SUMS" -type f -print -quit)
    if [[ -z "$CHECKSUM" ]]; then
      echo "下载结果未找到 $REMOTE_PATH/SHA256SUMS" >&2
      exit 1
    fi
    CHECKSUM_DIR=$(dirname "$CHECKSUM")
    (cd "$CHECKSUM_DIR" && sha256sum -c SHA256SUMS)
    echo "校验通过：$CHECKSUM_DIR"
    ;;
  *) usage ;;
esac
