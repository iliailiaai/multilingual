#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/megatron_bridge_cpt/env.local.sh}"

if [ -f "${ENV_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
else
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/megatron_bridge_cpt/env.example.sh"
fi

if [ ! -f "${LANGUAGE_MANIFEST}" ]; then
    echo "[ERROR] Missing language manifest: ${LANGUAGE_MANIFEST}" >&2
    exit 1
fi
if [ "${LANGUAGE_STEERING_ENABLE:-true}" = "true" ] && [ ! -d "${LANGUAGE_VECTOR_DIR}" ]; then
    echo "[ERROR] Missing language vector directory: ${LANGUAGE_VECTOR_DIR}" >&2
    exit 1
fi

cd "${MB_REPO}"
mkdir -p "${TRAIN_CKPT}"

TORCHRUN_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

CHECKPOINT_OVERRIDES=()
if [ "${RESUME}" = "1" ]; then
    CHECKPOINT_OVERRIDES=(
        "checkpoint.load=${TRAIN_CKPT}"
        "checkpoint.pretrained_checkpoint=null"
    )
else
    CHECKPOINT_OVERRIDES=(
        "checkpoint.load=null"
        "checkpoint.pretrained_checkpoint=${IMPORT_CKPT}"
    )
fi

FREEZE_ARG=()
if [ "${LANGUAGE_STEERING_FREEZE}" != "true" ]; then
    FREEZE_ARG=(--no_language_steering_freeze)
fi

LANGUAGE_STEERING_ARGS=()
if [ "${LANGUAGE_STEERING_ENABLE:-true}" = "true" ]; then
    LANGUAGE_STEERING_ARGS=(
        --language_vector_dir "${LANGUAGE_VECTOR_DIR}"
        --language_steering_alpha "${LANGUAGE_STEERING_ALPHA}"
        --language_steering_scaling "${LANGUAGE_STEERING_SCALING}"
        --language_steering_layers "${LANGUAGE_STEERING_LAYERS}"
        --language_vector_layer_offset "${LANGUAGE_VECTOR_LAYER_OFFSET}"
        "${FREEZE_ARG[@]}"
    )
fi

torchrun "${TORCHRUN_ARGS[@]}" scripts/training/run_recipe.py \
    --recipe qwen3_1p7b_pretrain_config \
    --dataset llm-pretrain \
    --seq_length "${SEQ_LENGTH}" \
    --hf_path "${HF_MODEL}" \
    --language_manifest "${LANGUAGE_MANIFEST}" \
    --language_blend_weight_key "${LANGUAGE_BLEND_WEIGHT_KEY}" \
    "${LANGUAGE_STEERING_ARGS[@]}" \
    "dataset.num_workers=${DATASET_NUM_WORKERS}" \
    "tokenizer.tokenizer_model=${HF_MODEL}" \
    "checkpoint.save=${TRAIN_CKPT}" \
    "checkpoint.save_interval=${SAVE_INTERVAL}" \
    "train.train_iters=${TRAIN_ITERS}" \
    "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "train.micro_batch_size=${MICRO_BATCH_SIZE}" \
    "validation.eval_interval=${EVAL_INTERVAL}" \
    "validation.eval_iters=${EVAL_ITERS}" \
    "logger.log_interval=${LOG_INTERVAL}" \
    "optimizer.lr=${LR}" \
    "optimizer.min_lr=${MIN_LR}" \
    "optimizer.weight_decay=${WEIGHT_DECAY}" \
    "scheduler.lr_warmup_iters=${LR_WARMUP_ITERS}" \
    "model.tensor_model_parallel_size=${TP}" \
    "model.pipeline_model_parallel_size=${PP}" \
    "model.context_parallel_size=${CP}" \
    "model.sequence_parallel=${SEQUENCE_PARALLEL}" \
    "${CHECKPOINT_OVERRIDES[@]}"
