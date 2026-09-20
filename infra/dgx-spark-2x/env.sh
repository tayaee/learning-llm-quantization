# NVIDIA DGX Spark 2대 운용 (각 본체에서 run.sh를 나눠 실행)
# 사용: SPARK_NODE=0 INFRA=dgx-spark-2x ./run.sh
export SPARK_COUNT=2
export SPARK_NODE="${SPARK_NODE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
