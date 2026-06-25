#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_PROVIDER=openrouter

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir test_data/factQA/sports/2026-03-01_2026-04-10 \
    --training_docs_dir test_data/train_docs/sports/2026-03-01_2026-04-10 \
    --output_dir test_data/benchmark_add/sports \
    --test_languages en ja fr es zh \
    --val_ratio 0.2 \
    --eval_model Qwen/Qwen2.5-3B-Instruct \
    --domain sports \
    --tp 1 \
    --gpu_mem 0.9 \
    --llm_provider "${LLM_PROVIDER}"
