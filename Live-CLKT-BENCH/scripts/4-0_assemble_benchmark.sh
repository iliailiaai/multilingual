#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir test_data/factQA/sports/2026-03-01_2026-04-10 \
    --training_docs_dir test_data/train_docs/sports/2026-03-01_2026-04-10 \
    --output_dir test_data/benchmark/sports \
    --test_languages en ja fr es zh \
    --val_ratio 0.2
