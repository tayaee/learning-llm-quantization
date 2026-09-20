# RunPod 8xH100 SXM 80GB 단일 노드 — 284B MoE GPTQ 양자화용
# 용도: DeepSeek-V4-Flash-Base(FP8, ~284GB) → INT4 (MoE-Quant expert 병렬)
# 팟 요구사항: GPU 8xH100 80GB, 볼륨 1TB+ (가중치 284GB + 출력 ~150GB + 캐시),
#   RAM 1TB+ 권장 (offload_activations), CUDA 12.4+, NCCL 사용
# 사용: INFRA=runpod-h100-8x ./run.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
