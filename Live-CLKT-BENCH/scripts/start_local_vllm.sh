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
CUDAGRAPH_NUM_WARMUPS=4
COMPILATION_CONFIG="{\"cudagraph_num_of_warmups\":${CUDAGRAPH_NUM_WARMUPS},\"cudagraph_capture_sizes\":[1,2],\"compile_sizes\":[\"cudagraph_capture_sizes\"]}"

echo "Starting vLLM on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using compilation config: ${COMPILATION_CONFIG}"

vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --compilation-config "${COMPILATION_CONFIG}"
