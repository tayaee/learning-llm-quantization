#!/usr/bin/env bash
# AWQ INT4 — mistralai/Devstral-Small-2505 → Devstral-Small-2505-AWQ-Int4-128g
# 관찰 포인트: Activation Outlier 보호(AWQ)가 에이전트 tool-calling 성공률에 미치는 영향
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
OUTPUT_NAME="Devstral-Small-2505-AWQ-Int4-128g"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$OUTPUT_DIR"

# 역할 분리: 이 스크립트는 양자화만 수행. 업로드는 ./upload.sh 사용
case " ${EXTRA_ARGS:-} $* " in
  *" --upload "*) echo "[quantize][ERROR] --upload 감지: HF_USER=<user> HF_TOKEN=<token> ./upload.sh 로 수행하세요." >&2; exit 1;;
esac

# shellcheck disable=SC2086
uv run "$TECH_DIR/src/awq-autoawq.py" \
  --hf-model "$HF_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "$OUTPUT_NAME" \
  --show-inference-instruction \
  --w-bit 4 --q-group-size 128 --zero-point --version GEMM \
  --nsamples 128 --seqlen 512 \
  ${EXTRA_ARGS:-} "$@"
# Hub 업로드 시: HF_USER=<user> HF_TOKEN=<token> ./run.sh --upload  (계정·토큰 명시 필수)
