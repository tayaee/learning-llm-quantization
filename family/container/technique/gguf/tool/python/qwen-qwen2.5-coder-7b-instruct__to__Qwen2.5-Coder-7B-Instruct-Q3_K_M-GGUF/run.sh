#!/usr/bin/env bash
# GGUF Q3_K_M — Qwen/Qwen2.5-Coder-7B-Instruct → Qwen2.5-Coder-7B-Instruct-Q3_K_M-GGUF
# 관찰 포인트: 7B급 3비트 한계 테스트 (Perplexity 급증 지점·HumanEval 점수 비교)
# 어느 디렉토리에서 실행해도 동일 결과 (이 파일 기준 절대경로).
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TECH_DIR="$(cd "$RUN_DIR/.." && pwd)"
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

HF_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_NAME="Qwen2.5-Coder-7B-Instruct-Q3_K_M-GGUF"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$OUTPUT_DIR"

# shellcheck disable=SC2086
uv run "$TECH_DIR/src/gguf.py" \
  --hf-model "$HF_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "$OUTPUT_NAME" \
  --show-inference-instruction \
  --quant-type Q3_K_M \
  ${EXTRA_ARGS:-} "$@"
# Hub 업로드 시: ./run.sh --upload  (HF_TOKEN 필요, 업로드 계정=$HF_USER)
