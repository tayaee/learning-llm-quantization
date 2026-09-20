# learning-llm-quantization

DGX Spark(128GB 통합메모리) 기준 오픈웨이트 코딩 모델의 4비트·3비트 양자화
파이프라인 모음. 각 모델별로 **시도 우선순위(preference) 순**으로 정렬되어 있다
(앞번호 = 먼저 시도할 것). ✅ = `run.sh` 제공, — = 스크립트 직접 실행.

## 세대 구분

- **Gen1 (2023)**: GPTQ, AWQ, GGUF K-quants — 스칼라 W-only 정석
- **Gen2 (2024)**: HQQ, AutoRound, QuaRot/SpinQuant, FP8, EXL2 — 최적화·회전·가변비트
- **Gen3 (2025–26)**: GPTQModel(+FOEM), llmcompressor 레시피, MoE-Quant, NVFP4, AQLM — 생태계·MoE·극저비트

## 0. allenai/Llama-3.1-Tulu-3-8B (베이스라인, Llama-3.1 8B instruction-tuned)

전 기법의 기준점. 열화가 적은 순서대로 나열 (근소 차이는 워크로드별 역전 가능).

**4비트** (W4A16 위주, W4A4/W4A8는 별도 표기):
1. ✅ AWQ-Int4 (Gen1) — instruction 유지 + Marlin. tier 내 대안:
   `llmcompressor`(vLLM 직행), `lmdeploy`(lmdeploy 서빙)
   `family/weight-only/technique/awq/tool/autoawq/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-AWQ-Int4-128g/run.sh`
   `family/weight-only/technique/awq/tool/llmcompressor/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-AWQ-W4A16/run.sh`
   `family/weight-only/technique/awq/tool/lmdeploy/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-AWQ-4bit/run.sh`
2. ✅ GPTQ-Int4 g128 (Gen1) — Hessian 보상, ppl parity
   `family/weight-only/technique/gptq/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-GPTQ-Int4-128g/run.sh`
3. ✅ AutoRound-Int4 (Gen2) — 반올림 학습, 4비트 SOTA급
   `family/weight-only/technique/auto-round/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-AutoRound-Int4-128g/run.sh`
4. ✅ SpinQuant-W4A16, llmcompressor datafree (Gen2) — Llama-3.1-8B 실측에서 회전+GPTQ가 무회전 대비 정확도·ppl 모두 개선
   `family/weight-only/technique/spin-quant/tool/llmcompressor/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-SpinQuant-W4A16/run.sh`
5. ✅ QuaRot-W4A16 (Gen2) — SpinQuant와 동급, 고정 Hadamard
   `family/weight-activation/technique/quarot/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-QuaRot-W4A16/run.sh`
6. ✅ SpinQuant-Int4, 자체 구현 (Gen2)
   `family/weight-only/technique/spin-quant/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-SpinQuant-Int4-128g/run.sh`
7. ✅ HQQ-Int4 (Gen2) — 무캘리브 고속 검증용
   `family/weight-only/technique/hqq/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-HQQ-Int4-64g/run.sh`
8. ✅ GGUF Q4_K_M (Gen1) — CPU/llama.cpp용
   `family/container/technique/gguf/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-Q4_K_M-GGUF/run.sh`
9. ✅ QuaRot-W4A4 (Gen2) — 활성화까지 4비트, 전용 커널 필요
   `family/weight-activation/technique/quarot/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-QuaRot-W4A4/run.sh`
10. ✅ SpinQuant-W4A8-meta (Gen2) — 학습 회전 최고 품질 후보, 수십 분 소요
    `family/weight-only/technique/spin-quant/tool/meta/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-SpinQuant-W4A8-meta/run.sh`

**3비트** (소형 모델이라 하락이 가파름 — ppl·벤치 병행 측정):
1. ✅ GPTQ-Int3 (Gen1) — 캘리브레이션 보상이 저비트에서 버팀
   `family/weight-only/technique/gptq/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-GPTQ-Int3-128g/run.sh`
2. — GGUF Q3_K_M (Gen1) — 스크립트 지원(`--quant-type Q3_K_M`), run.sh 미제공
3. — EXL2 3.2bpw (Gen2) — 혼합 정밀도, 스크립트 지원
4. ✅ AQLM-2bit (Gen3) — 2비트 극한용
   `family/weight-only/technique/aqlm/tool/python/allenai-llama-3.1-tulu-3-8b__to__Llama-3.1-Tulu-3-8B-AQLM-2bit-1x16/run.sh`

## 1. Qwen2.5-Coder-32B-Instruct (Dense 32B, FP16 ~64GB)

de-facto 코딩 벤치 표준. Dense라 캘리브레이션 수렴이 가장 안정적.

**4비트** (원본 성능 98%+ 목표):
1. ✅ AWQ-Int4 (Gen1, `awq/tool/autoawq`) — instruction 유지 + Marlin
   `family/weight-only/technique/awq/tool/autoawq/qwen-qwen2.5-coder-32b-instruct__to__Qwen2.5-Coder-32B-Instruct-AWQ-Int4-128g/run.sh`
2. — GPTQ-Int4 g128 (Gen1) — Qwen ppl 실측상 근소 우위, 대안 1순위
3. ✅ EXL2 4.0bpw (Gen2) — 동일 비트레이트 Pareto 최적
   `family/weight-only/technique/exl2/tool/python/qwen-qwen2.5-coder-32b-instruct__to__Qwen2.5-Coder-32B-Instruct-EXL2-4.0bpw/run.sh`
4. — GGUF Q4_K_M (Gen1) — llama.cpp 서빙용
5. — HQQ-Int4 (Gen2) — 무캘리브 고속 검증용

**3비트** (Perplexity 열화폭 비교 연습):
1. ✅ EXL2 3.2bpw (Gen2) — 혼합 정밀도 3비트대 최상
   `family/weight-only/technique/exl2/tool/python/qwen-qwen2.5-coder-32b-instruct__to__Qwen2.5-Coder-32B-Instruct-EXL2-3.2bpw/run.sh`
2. ✅ GGUF Q3_K_M (Gen1, ~14~15GB) — 3비트 문법 유지 확인용
   `family/container/technique/gguf/tool/python/qwen-qwen2.5-coder-32b-instruct__to__Qwen2.5-Coder-32B-Instruct-Q3_K_M-GGUF/run.sh`
3. — GPTQ-Int3 (Gen1)
4. — AQLM-2bit (Gen3) — 극한 압축용

## 2. DeepSeek-Coder-V2-Lite-Instruct (MoE 16B / Active 2.4B, FP16 ~32GB)

MLA+MoE 특수 아키텍처. MoE는 GPTQ 계열이 AWQ보다 실측 우위.

**4비트**:
1. ✅ GPTQ-Int4 (Gen1) — MoE Hessian 보상이 라우팅 안정에 유리
   `family/weight-only/technique/gptq/tool/python/deepseek-ai-deepseek-coder-v2-lite-instruct__to__DeepSeek-Coder-V2-Lite-Instruct-GPTQ-Int4-128g/run.sh`
2. ✅ GGUF Q4_K_M (Gen1)
   `family/container/technique/gguf/tool/python/deepseek-ai-deepseek-coder-v2-lite-instruct__to__DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF/run.sh`
3. — AWQ-Int4 (Gen1) — 대안
4. — HQQ-Int4 (Gen2) — 검증용

**3비트** (Expert 라우팅 불균형·인덴트 깨짐 관찰):
1. ✅ GGUF Q3_K_S (Gen1)
   `family/container/technique/gguf/tool/python/deepseek-ai-deepseek-coder-v2-lite-instruct__to__DeepSeek-Coder-V2-Lite-Instruct-Q3_K_S-GGUF/run.sh`
2. — GPTQ-Int3 (Gen1)
3. — Mixed-precision (Attention 4-bit + Expert 3-bit) — 실험 과제

## 3. Codestral-22B-v0.1 (Dense 22B, FP16 ~44GB, 게이트 모델)

80+ 언어·32k 컨텍스트. Mistral 공식 권장(EXL2) 우선. 게이트 모델이라 `HF_TOKEN` 필요.

**4비트**:
1. ✅ EXL2 4.25bpw (Gen2)
   `family/weight-only/technique/exl2/tool/python/mistralai-codestral-22b-v0.1__to__Codestral-22B-v0.1-EXL2-4.25bpw/run.sh`
2. — EXL2 4.0bpw (Gen2) — bpw만 낮춘 비교용
3. — GGUF Q4_K_M (Gen1)
4. — AWQ-Int4 (Gen1)

**3비트** (FIM 완성 정확도 추적):
1. ✅ EXL2 3.5bpw (Gen2, ~10~11GB)
   `family/weight-only/technique/exl2/tool/python/mistralai-codestral-22b-v0.1__to__Codestral-22B-v0.1-EXL2-3.5bpw/run.sh`
2. — EXL2 3.0bpw (Gen2)
3. — GGUF Q3_K_M (Gen1)

## 4. Devstral-Small-2505 (Dense 24B, FP16 ~48GB)

에이전틱 코딩·tool-calling 특화. Outlier 보호가 함수 호출 성공률에 직결.

**4비트**:
1. ✅ AWQ-Int4 W4A16 (Gen1)
   `family/weight-only/technique/awq/tool/autoawq/mistralai-devstral-small-2505__to__Devstral-Small-2505-AWQ-Int4-128g/run.sh`
2. — GPTQ-Int4 (Gen1) — AWQ 대비 벤치마크용
3. — EXL2 4.0bpw (Gen2)
4. — GGUF Q4_K_M (Gen1)

**3비트** (특수 토큰·JSON Schema 파싱 붕괴 검증):
1. ✅ EXL2 3.0bpw (Gen2)
   `family/weight-only/technique/exl2/tool/python/mistralai-devstral-small-2505__to__Devstral-Small-2505-EXL2-3.0bpw/run.sh`
2. — GGUF Q3_K_M (Gen1)

## 5. Qwen2.5-Coder-7B-Instruct (Dense 7B, FP16 ~14GB)

파이프라인 디버깅·고속 이터레이션용 (수 분 내 변환). 소형 모델은 3비트 하락이 급격.

**4비트**:
1. — AWQ-Int4 (Gen1) — 품질 우선
2. — GPTQ-Int4 (Gen1)
3. ✅ GGUF Q4_K_S (Gen1) — 디버깅·속도 우선
   `family/container/technique/gguf/tool/python/qwen-qwen2.5-coder-7b-instruct__to__Qwen2.5-Coder-7B-Instruct-Q4_K_S-GGUF/run.sh`
4. — EXL2 4.0bpw (Gen2)

**3비트** (하한선 체감: Perplexity·HumanEval 비교):
1. ✅ GGUF Q3_K_M (Gen1)
   `family/container/technique/gguf/tool/python/qwen-qwen2.5-coder-7b-instruct__to__Qwen2.5-Coder-7B-Instruct-Q3_K_M-GGUF/run.sh`
2. — EXL2 3.0bpw (Gen2)

## 실행 관행

```bash
./<run.sh>                    # 로컬 실행 (INFRA 기본 dgx-spark-1x, --upload OFF)
./<run.sh> --upload           # 동작 확인 후: HF_USER(기본 tayaee)로 Hub 업로드, HF_TOKEN 필요
INFRA=runpod-h100-1x ./<run.sh>  # 클라우드 실행
```

- Python 버전: 루트 `.python-version`(3.12), `gptq/`만 3.11 override (auto-gptq 미지원)
- 구조·slug 규칙·업로드 관행: `family/README.md`, infra 정의: `infra/README.md`
- 그 외: `Qwen/Qwen3.8-27B`(보류), `deepseek-ai/DeepSeek-V4-Flash-Base`(MoE-Quant, 8×H100)
