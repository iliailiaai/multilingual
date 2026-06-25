#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

MODEL=Qwen/Qwen3.5-35B-A3B
SERVED_MODEL_NAME=local-model
HOST=0.0.0.0
PORT=8000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=16384

echo "Starting vLLM on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}"
