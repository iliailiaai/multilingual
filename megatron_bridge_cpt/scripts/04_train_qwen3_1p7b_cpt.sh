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

if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "[ERROR] Missing Megatron indexed dataset: ${DATA_PREFIX}.bin/.idx" >&2
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

torchrun "${TORCHRUN_ARGS[@]}" scripts/training/run_recipe.py \
    --recipe qwen3_1p7b_pretrain_config \
    --dataset llm-pretrain \
    --seq_length "${SEQ_LENGTH}" \
    --hf_path "${HF_MODEL}" \
    "dataset.blend=[[${DATA_PREFIX}],null]" \
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
