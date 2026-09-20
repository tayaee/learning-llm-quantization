#!/usr/bin/env bash
# Upload 템플릿 — 새 실행 리프를 만들 때 이 파일을 복사한다.
# 사용: HF_USER=<user> HF_TOKEN=<token> ./upload.sh
# 계정·토큰 중 하나라도 없으면 에러 종료한다 (잘못된 업로드 방지).
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_NAME="${OUTPUT_NAME:-Llama-3.1-Tulu-3-8B-GPTQ-Int4-128g}"
OUTPUT_DIR="$RUN_DIR/output"

[ -n "${HF_USER:-}" ] || { echo "[upload][ERROR] HF_USER 미지정." >&2; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "[upload][ERROR] HF_TOKEN 미지정." >&2; exit 1; }
if [ ! -d "$OUTPUT_DIR" ] || [ -z "$(ls -A "$OUTPUT_DIR")" ]; then
  echo "[upload][ERROR] output/이 비어 있음. ./quantize.sh를 먼저 실행하세요." >&2; exit 1
fi

REPO="$OUTPUT_NAME"
case "$REPO" in */*) ;; *) REPO="$HF_USER/$REPO";; esac

uv run --with huggingface_hub python3 - "$OUTPUT_DIR" "$REPO" <<'PYEOF'
import sys
from huggingface_hub import HfApi
out_dir, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id, exist_ok=True, private=False)
api.upload_folder(folder_path=out_dir, repo_id=repo_id, ignore_patterns=["calibration.txt"])
print(f"[upload] 완료: https://huggingface.co/{repo_id}")
PYEOF
