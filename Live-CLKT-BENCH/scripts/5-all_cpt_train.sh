#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=3

LANGUAGES="en ja fr es zh German Dutch Russian Ukrainian Polish Czech Portuguese Italian Urdu Persian Irish Welsh Arabic Hebrew Finnish Estonian Hungarian Turkish Azerbaijani Kazakh Uzbek Indonesian Thai"

MODEL_NAME=Qwen/Qwen3-1.7B
MODEL_DIR_NAME=Qwen/Qwen3-1.7B

BATCH_SIZE=32
LEARNING_RATE=5e-4
NUM_TRAIN_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=2
RANK=16
ALPHA=32
DROPOUT=0.1

is_trained() {
    local output_dir="$1"
    local checkpoint_dir="${output_dir}/checkpoints"
    local count=0

    if [ -d "${checkpoint_dir}" ]; then
        count=$(find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-epoch-*" | wc -l)
    elif [ -d "${output_dir}" ]; then
        count=$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-epoch-*" | wc -l)
    fi

    [ "${count}" -ge "${NUM_TRAIN_EPOCHS}" ]
}

for TRAIN_LANG in ${LANGUAGES}; do
    echo "============================================================"
    echo "[CPT] Language: ${TRAIN_LANG}"

    OUTPUT_DIR="test_models/combined/${TRAIN_LANG}/${MODEL_DIR_NAME}"
    TRAIN_FILES=(
        "test_data/benchmark/sports/${TRAIN_LANG}/train_doc.jsonl"
        "test_data/benchmark/movie/${TRAIN_LANG}/train_doc.jsonl"
        "test_data/benchmark/music/${TRAIN_LANG}/train_doc.jsonl"
    )

    missing=0
    for train_file in "${TRAIN_FILES[@]}"; do
        if [ ! -f "${train_file}" ]; then
            echo "[WARN] Missing train file: ${train_file}"
            missing=1
        fi
    done

    if [ "${missing}" -eq 1 ]; then
        echo "[SKIP] ${TRAIN_LANG}: benchmark train docs are incomplete."
        continue
    fi

    if is_trained "${OUTPUT_DIR}"; then
        echo "[SKIP] ${TRAIN_LANG}: ${MODEL_NAME} already has ${NUM_TRAIN_EPOCHS} checkpoints."
        continue
    fi

    python3 demo_experiment/cpt.py \
        --model_name "${MODEL_NAME}" \
        --train_file "${TRAIN_FILES[@]}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --rank "${RANK}" \
        --alpha "${ALPHA}" \
        --dropout "${DROPOUT}"
done
