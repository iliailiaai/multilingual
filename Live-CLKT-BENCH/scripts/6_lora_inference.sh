#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_LANG=en
MODEL_DIR_NAME=Qwen/Qwen3-1.7B
MODEL_DIR="test_models/combined/${TRAIN_LANG}/${MODEL_DIR_NAME}"
CHECKPOINTS="checkpoint-epoch-3"

PYTHONPATH=lib python3 demo_experiment/lora_inference.py \
    --model_dir "${MODEL_DIR}" \
    --test_file_path "test_data/benchmark/sports/${TRAIN_LANG}/test_mc.jsonl" \
    --output_dir "test_data/inference_output/sports/${TRAIN_LANG}/finetune_combined" \
    --temperature 0.6 \
    --checkpoints ${CHECKPOINTS}

PYTHONPATH=lib python3 demo_experiment/lora_inference.py \
    --model_dir "${MODEL_DIR}" \
    --test_file_path "test_data/benchmark/movie/${TRAIN_LANG}/test_mc.jsonl" \
    --output_dir "test_data/inference_output/movie/${TRAIN_LANG}/finetune_combined" \
    --temperature 0.6 \
    --checkpoints ${CHECKPOINTS}

PYTHONPATH=lib python3 demo_experiment/lora_inference.py \
    --model_dir "${MODEL_DIR}" \
    --test_file_path "test_data/benchmark/music/${TRAIN_LANG}/test_mc.jsonl" \
    --output_dir "test_data/inference_output/music/${TRAIN_LANG}/finetune_combined" \
    --temperature 0.6 \
    --checkpoints ${CHECKPOINTS}
