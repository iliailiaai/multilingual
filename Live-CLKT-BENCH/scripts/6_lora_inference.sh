#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=lib python3 demo_experiment/lora_inference.py \
    --model_dir test_models/sports/en/Qwen2.5-1.5B-Instruct \
    --test_file_path test_data/benchmark/sports/en/test_mc.jsonl \
    --output_dir test_data/inference_output/sports/en/finetune \
    --temperature 0.6
