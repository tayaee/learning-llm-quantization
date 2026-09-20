#!/usr/bin/env bash
# EXL2 3.0bpw — mistralai/Devstral-Small-2505 → Devstral-Small-2505-EXL2-3.0bpw
# 관찰 포인트: 3비트에서 특수 토큰·JSON Schema·함수 호출 파싱 붕괴 여부 검증
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

HF_MODEL="mistralai/Devstral-Small-2505"
OUTPUT_NAME="Devstral-Small-2505-EXL2-3.0bpw"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$OUTPUT_DIR"

# 역할 분리: 이 스크립트는 양자화만 수행. 업로드는 ./upload.sh 사용
case " ${EXTRA_ARGS:-} $* " in
  *" --upload "*) echo "[quantize][ERROR] --upload 감지: HF_USER=<user> HF_TOKEN=<token> ./upload.sh 로 수행하세요." >&2; exit 1;;
esac

# shellcheck disable=SC2086
uv run "$TECH_DIR/src/exl2.py" \
  --hf-model "$HF_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "$OUTPUT_NAME" \
  --show-inference-instruction \
  --bpw 3.0 \
  ${EXTRA_ARGS:-} "$@"
# Hub 업로드 시: HF_USER=<user> HF_TOKEN=<token> ./run.sh --upload  (계정·토큰 명시 필수)
