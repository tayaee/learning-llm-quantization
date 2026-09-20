# RunPod H100 1x (ephemeral — 캐시는 /workspace 영속 볼륨으로)
# 사용: INFRA=runpod-h100-1x HF_TOKEN=... ./run.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
