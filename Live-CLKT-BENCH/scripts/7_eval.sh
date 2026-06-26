#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_LANG=en
MODEL_DIR_NAME=Qwen3-1.7B
CKPT=checkpoint-epoch-1

python3 demo_experiment/eval.py \
    --pred_file "test_data/inference_output/sports/${TRAIN_LANG}/finetune_combined/${MODEL_DIR_NAME}/${CKPT}_pred.jsonl" \
    --output_file "test_data/eval_result/sports/${TRAIN_LANG}/finetune_combined/${MODEL_DIR_NAME}_${CKPT}_pred.json"

python3 demo_experiment/eval.py \
    --pred_file "test_data/inference_output/movie/${TRAIN_LANG}/finetune_combined/${MODEL_DIR_NAME}/${CKPT}_pred.jsonl" \
    --output_file "test_data/eval_result/movie/${TRAIN_LANG}/finetune_combined/${MODEL_DIR_NAME}_${CKPT}_pred.json"

python3 demo_experiment/eval.py \
    --pred_file "test_data/inference_output/music/${TRAIN_LANG}/finetune_combined/${MODEL_DIR_NAME}/${CKPT}_pred.jsonl" \
    --output_file "test_data/eval_result/music/${TRAIN_LANG}/finetune_combined/${MODEL_DIR_NAME}_${CKPT}_pred.json"
