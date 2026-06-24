#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 demo_experiment/eval.py \
    --pred_file test_data/inference_output/sports/en/finetune/Qwen2.5-1.5B-Instruct/checkpoint-epoch-3_pred.jsonl \
    --output_file test_data/eval_result/sports/en/finetune/Qwen2.5-1.5B-Instruct_checkpoint-epoch-3_pred.json
