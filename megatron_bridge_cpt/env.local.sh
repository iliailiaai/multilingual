MEGATRON_BRIDGE_CPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${MEGATRON_BRIDGE_CPT_DIR}/.." && pwd)"
DEFAULT_MB_REPO="${MEGATRON_BRIDGE_CPT_DIR}/Megatron-Bridge"

export MB_REPO="${DEFAULT_MB_REPO}"
export PYTHON_BIN="python"

export HF_MODEL="Qwen/Qwen3-1.7B"
export WORKDIR="/workdir/main/qwen3_1p7b_cpt_10b-st"

export RAW_CORPUS_DIR="${PROJECT_ROOT}/corpus_download/data"
export CPT_JSONL="${WORKDIR}/raw/qwen3_cpt_10b.jsonl"
export LANGUAGE_JSONL_DIR="${WORKDIR}/raw_by_language"
export LANGUAGE_MANIFEST="${LANGUAGE_JSONL_DIR}/language_manifest.json"
export TARGET_TOKENS="10000000000"

export DATA_OUTPUT_PREFIX="${WORKDIR}/megatron_data/qwen3_cpt_10b"
export DATA_PREFIX="${DATA_OUTPUT_PREFIX}_text_document"
export LANGUAGE_DATA_PREFIX_DIR="${WORKDIR}/megatron_data_by_language"
export PREPROCESS_WORKERS="32"

export LANGUAGE_VECTOR_DIR="${PROJECT_ROOT}/recover/collect_language_vectors/language_vectors_bucket/flores_plus/Qwen3-1.7B/full"
export LANGUAGE_STEERING_ALPHA="1.0"
export LANGUAGE_STEERING_SCALING="none"
export LANGUAGE_STEERING_LAYERS="7"
export LANGUAGE_VECTOR_LAYER_OFFSET="1"
export LANGUAGE_BLEND_WEIGHT_KEY="written_tokens"
export LANGUAGE_STEERING_FREEZE="true"

export IMPORT_CKPT="${WORKDIR}/checkpoints/qwen3_1p7b_hf_import"
export TRAIN_CKPT="${WORKDIR}/checkpoints/qwen3_1p7b_cpt"
export EXPORT_HF="${WORKDIR}/hf_export/qwen3_1p7b_cpt"

# Inside container, the host GPUs 2,3,4 are usually remapped to visible devices 0,1,2.
export CUDA_VISIBLE_DEVICES=0,1,2
export GPUS_PER_NODE=3
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export TP=1
export PP=1
export CP=1
export SEQUENCE_PARALLEL=false

export SEQ_LENGTH=4096
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=192
export TRAIN_ITERS=12716

export LR=1.0e-5
export MIN_LR=1.0e-6
export LR_WARMUP_ITERS=1000
export WEIGHT_DECAY=0.1

export SAVE_INTERVAL=4240
export EVAL_INTERVAL=250
export EVAL_ITERS=10
export LOG_INTERVAL=10
export DATASET_NUM_WORKERS=3

export RESUME=0
export MEGATRON_EXPORT_CKPT="${TRAIN_CKPT}"

# Fill this after find command if needed.
export PREPROCESS_SCRIPT="megatron_bridge_cpt/Megatron-LM/tools/preprocess_data.py"
