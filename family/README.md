# family/ — 양자화 기법 분류 트리

1단계 `family` = **무엇을 양자화하는가** (서빙 요건을 가름),
2단계 `technique/` 아래 `<technique>` = **어떤 기법인가**,
3단계 `tool/` 아래 `<tool>` = **무엇으로 수행하는가**
(`python` = 자체 PEP 723 파이썬 구현, 그 외는 외부 도구 slug).

```
family/<family>/technique/<technique>/tool/<tool>/
├── src/<impl>.py           # PEP 723 인라인 의존성 (`uv run src/...` 즉시 실행, venv 불필요)
├── quantize.sh             # (일부) 양자화 wrapper 템플릿
├── upload.sh               # (일부) 업로드 템플릿
├── README.md               # 기법 문서
├── .python-version         # 필요 시 technique 단위 override (기본은 루트 3.12)
└── <input-model-slug>__to__<output-model-slug>/
    ├── quantize.sh         # 양자화만 수행 (--upload 전달 시 에러, 호출 위치 무관)
    ├── upload.sh           # output/ 업로드 전용 (HF_USER/HF_TOKEN 명시 필수)
    └── output/             # 산출물 (gitignore, Hub 업로드용)
```

실행 예 (어느 디렉토리에서든 동일 결과, INFRA 기본값 dgx-spark-1x):

```bash
./family/weight-only/technique/gptq/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-GPTQ-Int4-128g/quantize.sh
INFRA=runpod-h100-1x ./family/container/technique/gguf/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-Q4_K_M-GGUF/quantize.sh
HF_USER=<user> HF_TOKEN=<token> ./family/weight-only/technique/awq/tool/autoawq/.../upload.sh
```

Python 버전은 루트 `.python-version`(3.12)이 기본이며, `quantize.sh`가 technique
디렉터리로 `cd`한 뒤 `uv run`하므로 technique 아래 `.python-version`이
자동 override된다. 상세는 `infra/README.md` 참조.

## 현재 배치

- `weight-only/` — 활성화 FP16 유지: `gptq/`(`tool/python` legacy, `tool/gptqmodel`, `tool/moe-quant` DeepSeek MoE 전용), `awq/`(`tool/autoawq` 기본, `tool/llmcompressor`, `tool/lmdeploy`), `spin-quant/`(`tool/python`, `tool/meta`, `tool/llmcompressor`), `exl2/`(`tool/python`), `hqq/`, `auto-round/`, `aqlm/` (그 외는 `tool/python/` 단일)
- `weight-activation/` — 활성화까지 양자화: `smoothquant/` (W8A8), `fp8/` (E4M3/E5M2), `quarot/` (W4A4/W4A16)
- `container/` — 패키징 포맷 (양자화 방식과 직교): `gguf/` (Q4_K_M, Q8_0…)

다중 모드 기법(SpinQuant W4A16/W4A8, QuaRot W4A16/W4A4)은 대표 모드의
family에 1회만 배치하고, 모드별 실행은 `<input>__to__<output>` 디렉터리 +
`--mode` 값으로 구분한다.

## slug 규칙

- `<input-model-slug>`: HF ID 소문자화 + `/` → `-` (예: `allenai-llama-3.1-tulu-3-8b`)
- `<output-model-slug>`: `--output-model-name` 그대로 (예: `Llama-3.1-Tulu-3-8B-GPTQ-Int4-128g`)
- 같은 입력 모델의 다중 설정(Q4_K_M vs Q8_0 등)은 output slug로 구분되므로 충돌 없음

## 양자화·업로드 분리 (명시 필수, 기본값 없음)

- `./quantize.sh`: 양자화만 수행. `--upload` 전달 시 에러 종료.
- `HF_USER=<user> HF_TOKEN=<token> ./upload.sh`: `output/` 업로드 전용.
  계정·토큰 중 하나라도 없거나 `output/`이 비어 있으면 에러 종료한다.
- 모든 `src/*.py`는 `--hf-user` > `HF_USER` env 순으로 계정을 결정하고,
  둘 다 없으면 에러 종료한다 (`whoami()` 조회 없음).

## 코딩 모델 실습 세트 (dgx-spark-1x, 128GB 통합메모리)

| 모델 | 4비트 | 3비트 | 관찰 포인트 |
|---|---|---|---|
| Qwen2.5-Coder-32B | AWQ / EXL2 4.0bpw | GGUF Q3_K_M / EXL2 3.2bpw | 벤치 점수 유지율 Baseline |
| DeepSeek-Coder-V2-Lite (MoE 16B/2.4B) | AutoGPTQ / GGUF Q4_K_M | GGUF Q3_K_S | Expert 양자화 붕괴 여부 |
| Codestral-22B (게이트) | EXL2 4.25bpw | EXL2 3.5bpw | FIM 완성 정확도, 단계별 bpw |
| Devstral-Small-2505 | AWQ W4A16 | EXL2 3.0bpw | tool-calling·JSON 파싱 보존 |
| Qwen2.5-Coder-7B | GGUF Q4_K_S | GGUF Q3_K_M | 파이프라인 디버깅, 3비트 하한선 |
