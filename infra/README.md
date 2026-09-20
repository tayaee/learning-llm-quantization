# infra/ — 실행 인프라 정의

`run.sh`는 `INFRA=<infra-slug>` 환경변수로 이 디렉토리 아래 `env.sh`를 소싱한다.
비밀(`HF_TOKEN` 등)은 env.sh에 절대 넣지 않고, 호출자가 미리 export한다.

```bash
./run.sh                                            # 기본 하드웨어 dgx-spark-1x
INFRA=dgx-spark-2x SPARK_NODE=1 ./run.sh
INFRA=runpod-h100-1x HF_TOKEN=... ./run.sh
```

## slug 규칙

- 베어메탈/단일머신: `<device>-<scale>` → `dgx-spark-1x`, `dgx-spark-2x`
- 클라우드: `<provider>-<gpu>-<count>` → `runpod-h100-1x`, `runpod-b200-2x`
- 개수는 물리 단위(GPU 카드 또는 Spark 본체) 기준

## env 계약 (run.sh가 기대하는 변수)

| 변수 | 용도 |
|---|---|
| `HF_HOME`, `UV_CACHE_DIR` | 캐시 위치 (RunPod는 `/workspace` 영속 볼륨으로) |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU 인덱스 |
| `OMP_NUM_THREADS` | CPU 스레드 (DGX Spark ARM 효율) |
| `PYTORCH_CUDA_ALLOC_CONF` | 단편화 완화 |
| `VLLM_TP_SIZE` | 서빙 가이드용 tensor-parallel 기본값 |

## GPU 사이징 (양자화 작업 기준)

| 모델 규모 | 예시 | 권장 infra | 근거 |
|---|---|---|---|
| ~8B (bf16 ~16GB) | Llama-3.1-Tulu-3-8B | `dgx-spark-1x` 또는 `runpod-h100-1x` | 단일 GPU 여유 |
| ~27B (bf16 ~54GB) | Qwen3.8-27B | `runpod-h100-1x` (80GB, 볼륨 250GB+) | A800 80GB 단일 실측 선행 |
| ~284B MoE (FP8 ~284GB) | DeepSeek-V4-Flash-Base | `runpod-h100-8x` (80GB×8, 볼륨 1TB+, RAM 1TB+) | expert 병렬 필수, 671B 기준 2시간 |

RunPod 팟 체크리스트: GPU 타입·수량, 컨테이너 볼륨 크기, CUDA 12.4+,
`HF_TOKEN`(게이트 모델용; Apache/MIT 모델은 불필요).

## dgx-spark-2x 운용

양자화 작업 자체는 단일 GPU 작업이므로, 2x는 **두 run.sh를 각 본체에서**
나눠 실행하는 용도다 (`SPARK_NODE=0/1`로 매트릭스 분할).
듀얼노드 TP 서빙은 별도 Ray 구성이 필요하며 이 레포 범위가 아니다.

## Python 버전

루트 `.python-version`(3.12)이 기본. `run.sh`는 실행 전 technique 디렉터리로
`cd`하므로, technique 아래 `.python-version`이 있으면 자동 override된다.
현재 `gptq/`만 3.11 override (auto-gptq 3.12 미지원, 저장소 archived).
`aqlm/`도 3.12 빌드 실패 시 동일 방식으로 추가한다.
