#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_LANG=en
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MODEL_DIR_NAME=Qwen2.5-1.5B-Instruct

python3 demo_experiment/cpt.py \
    --model_name "${MODEL_NAME}" \
    --train_file \
        "test_data/benchmark/sports/${TRAIN_LANG}/train_doc.jsonl" \
        "test_data/benchmark/movie/${TRAIN_LANG}/train_doc.jsonl" \
        "test_data/benchmark/music/${TRAIN_LANG}/train_doc.jsonl" \
    --output_dir "test_models/combined/${TRAIN_LANG}/${MODEL_DIR_NAME}" \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --rank 16 \
    --alpha 32 \
    --dropout 0.1
