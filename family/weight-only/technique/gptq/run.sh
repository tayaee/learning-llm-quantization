#!/usr/bin/env bash
# GPTQ 양자화 wrapper 템플릿 (family/weight-only/technique/gptq/run.sh)
# 새 실행 리프를 만들 때 이 파일을 복사해 값을 고정한다.
# 사용법: HF_MODEL=... OUTPUT_NAME=... ./run.sh [--upload ...]
# 어느 디렉토리에서 실행해도 동일 결과 (이 파일 기준 절대경로).
set -euo pipefail

TECH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$TECH_DIR" # infra/가 있는 루트를 상향 탐색 (디렉터리 깊이 무관)
while [ ! -d "$ROOT_DIR/infra" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$TECH_DIR" # uv `.python-version` 탐색 기준 고정 (루트 3.12, technique 단위 override 가능)

INFRA="${INFRA:-dgx-spark-1x}" # 기본 하드웨어 (override 예: INFRA=runpod-h100-1x ./run.sh)
ENV_FILE="$ROOT_DIR/infra/$INFRA/env.sh"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "[run] unknown INFRA='$INFRA' (infra/ 아래 slug 확인)" >&2; exit 1
fi

HF_USER="${HF_USER:-tayaee}" # 기본 업로드 계정 (override: HF_USER=... ./run.sh)
export HF_USER

HF_MODEL="${HF_MODEL:-allenai/Llama-3.1-Tulu-3-8B}"
OUTPUT_NAME="${OUTPUT_NAME:-Llama-3.1-Tulu-3-8B-GPTQ-Int4-128g}"
OUTPUT_DIR="${OUTPUT_DIR:-$TECH_DIR/tool/python/allenai-llama-3.1-tulu-3-8b__to__$OUTPUT_NAME/output}"
EXTRA_ARGS="${EXTRA_ARGS:-}" # 예: "--bits 4 --group-size 128 --upload --hf-user myorg"

mkdir -p "$OUTPUT_DIR"

# shellcheck disable=SC2086
uv run "$TECH_DIR/src/gptq.py" \
  --hf-model "$HF_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "$OUTPUT_NAME" \
  --show-inference-instruction \
  $EXTRA_ARGS "$@"
