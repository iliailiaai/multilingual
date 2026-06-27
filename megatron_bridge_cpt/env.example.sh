#!/usr/bin/env bash

# Copy to env.local.sh and edit on the training machine.
# Do not source this file blindly if paths differ.

MEGATRON_BRIDGE_CPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${MEGATRON_BRIDGE_CPT_DIR}/.." && pwd)}"
DEFAULT_MB_REPO="${MEGATRON_BRIDGE_CPT_DIR}/Megatron-Bridge"

export MB_REPO="${MB_REPO:-${DEFAULT_MB_REPO}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_MODEL="${HF_MODEL:-Qwen/Qwen3-1.7B}"
export WORKDIR="${WORKDIR:-/data/qwen3_1p7b_cpt_10b}"

export RAW_CORPUS_DIR="${RAW_CORPUS_DIR:-${PROJECT_ROOT}/corpus_download/data}"
export CPT_JSONL="${CPT_JSONL:-${WORKDIR}/raw/qwen3_cpt_10b.jsonl}"
export LANGUAGE_JSONL_DIR="${LANGUAGE_JSONL_DIR:-${WORKDIR}/raw_by_language}"
export LANGUAGE_MANIFEST="${LANGUAGE_MANIFEST:-${LANGUAGE_JSONL_DIR}/language_manifest.json}"
export TARGET_TOKENS="${TARGET_TOKENS:-10000000000}"

export DATA_OUTPUT_PREFIX="${DATA_OUTPUT_PREFIX:-${WORKDIR}/megatron_data/qwen3_cpt_10b}"
export DATA_PREFIX="${DATA_PREFIX:-${DATA_OUTPUT_PREFIX}_text_document}"
export LANGUAGE_DATA_PREFIX_DIR="${LANGUAGE_DATA_PREFIX_DIR:-${WORKDIR}/megatron_data_by_language}"
export PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-32}"

export LANGUAGE_VECTOR_DIR="${LANGUAGE_VECTOR_DIR:-${PROJECT_ROOT}/recover/collect_language_vectors/language_vectors_bucket/flores_plus/Qwen3-1.7B/full}"
export LANGUAGE_STEERING_ENABLE="${LANGUAGE_STEERING_ENABLE:-true}"
export LANGUAGE_STEERING_ALPHA="${LANGUAGE_STEERING_ALPHA:-1.0}"
export LANGUAGE_STEERING_SCALING="${LANGUAGE_STEERING_SCALING:-none}"
export LANGUAGE_STEERING_LAYERS="${LANGUAGE_STEERING_LAYERS:-7}"
export LANGUAGE_VECTOR_LAYER_OFFSET="${LANGUAGE_VECTOR_LAYER_OFFSET:-1}"
export LANGUAGE_BLEND_WEIGHT_KEY="${LANGUAGE_BLEND_WEIGHT_KEY:-written_tokens}"
export LANGUAGE_STEERING_FREEZE="${LANGUAGE_STEERING_FREEZE:-true}"

export IMPORT_CKPT="${IMPORT_CKPT:-${WORKDIR}/checkpoints/qwen3_1p7b_hf_import}"
export TRAIN_CKPT="${TRAIN_CKPT:-${WORKDIR}/checkpoints/qwen3_1p7b_cpt}"
export EXPORT_HF="${EXPORT_HF:-${WORKDIR}/hf_export/qwen3_1p7b_cpt}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export TP="${TP:-1}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"

export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export TRAIN_ITERS="${TRAIN_ITERS:-9540}"

export LR="${LR:-1.0e-5}"
export MIN_LR="${MIN_LR:-1.0e-6}"
export LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-200}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"

export RESUME="${RESUME:-0}"
export MEGATRON_EXPORT_CKPT="${MEGATRON_EXPORT_CKPT:-${TRAIN_CKPT}}"
