#!/usr/bin/env bash
# EXL2 4.25bpw — mistralai/Codestral-22B-v0.1 → Codestral-22B-v0.1-EXL2-4.25bpw
# 관찰 포인트: 단계별 비트레이트 빌드 (4.25 → 3.5), FIM 코드 완성 정확도
# NOTE: 게이트 모델이므로 HF_TOKEN 필요 ( https://huggingface.co/mistralai/Codestral-22B-v0.1 승인 후)
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

HF_MODEL="mistralai/Codestral-22B-v0.1"
OUTPUT_NAME="Codestral-22B-v0.1-EXL2-4.25bpw"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$OUTPUT_DIR"

# shellcheck disable=SC2086
uv run "$TECH_DIR/src/exl2.py" \
  --hf-model "$HF_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "$OUTPUT_NAME" \
  --show-inference-instruction \
  --bpw 4.25 \
  ${EXTRA_ARGS:-} "$@"
# Hub 업로드 시: ./run.sh --upload  (HF_TOKEN 필요, 업로드 계정=$HF_USER)
