#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 demo_experiment/cpt.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --train_file test_data/benchmark/sports/en/train_doc.jsonl \
    --output_dir test_models/sports/en/Qwen2.5-1.5B-Instruct \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --rank 16 \
    --alpha 32 \
    --dropout 0.1
