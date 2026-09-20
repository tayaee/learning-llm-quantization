#!/usr/bin/env bash
# GPTQ INT4 (MoE-Quant, 8 GPU) — deepseek-ai/DeepSeek-V4-Flash-Base → DeepSeek-V4-Flash-GPTQ-4b-128g-experts
# 어느 디렉토리에서 실행해도 동일 결과 (이 파일 기준 절대경로).
# 의존성은 src/*.py 상단 PEP 723 블록이 선언 (venv/requirements 불필요).
# NOTE: 입력은 FP8-Base 체크아웃(~284GB). MoE-Quant는 V3/R1 검증, V4 호환 미확인.
# 인프라: INFRA=<infra-slug> ./run.sh  (기본 runpod-h100-8x)
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TECH_DIR="$(cd "$RUN_DIR/.." && pwd)"
ROOT_DIR="$TECH_DIR" # infra/가 있는 루트를 상향 탐색 (디렉터리 깊이 무관)
while [ ! -d "$ROOT_DIR/infra" ] && [ "$ROOT_DIR" != "/" ]; do ROOT_DIR="$(dirname "$ROOT_DIR")"; done
cd "$TECH_DIR" # uv `.python-version` 탐색 기준 고정 (루트 3.12, technique 단위 override 가능)

INFRA="${INFRA:-runpod-h100-8x}" # 기본 하드웨어: 8xH100 80GB 단일 노드
ENV_FILE="$ROOT_DIR/infra/$INFRA/env.sh"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "[run] unknown INFRA='$INFRA' (infra/ 아래 slug 확인)" >&2; exit 1
fi

HF_MODEL="deepseek-ai/DeepSeek-V4-Flash-Base"
OUTPUT_NAME="DeepSeek-V4-Flash-GPTQ-4b-128g-experts"
OUTPUT_DIR="$RUN_DIR/output"
mkdir -p "$OUTPUT_DIR"

# 역할 분리: 이 스크립트는 양자화만 수행. 업로드는 ./upload.sh 사용
case " ${EXTRA_ARGS:-} $* " in
  *" --upload "*) echo "[quantize][ERROR] --upload 감지: HF_USER=<user> HF_TOKEN=<token> ./upload.sh 로 수행하세요." >&2; exit 1;;
esac

# shellcheck disable=SC2086
uv run "$TECH_DIR/src/moe-quant.py" \
  --hf-model "$HF_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --output-model-name "$OUTPUT_NAME" \
  --show-inference-instruction \
  --dataset open-thoughts --nsamples 512 --seqlen 4096 \
  --bits 4 --group-size 128 \
  --quantization-order activation --quantization-scale mse \
  --quantize-only-experts --num-gpus 8 \
  ${EXTRA_ARGS:-} "$@"
# Hub 업로드 시: HF_USER=<user> HF_TOKEN=<token> ./run.sh --upload  (계정·토큰 명시 필수)
