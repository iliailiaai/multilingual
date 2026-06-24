#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain sports \
    --training_docs_dir test_data/train_docs/sports/2026-03-01_2026-04-10 \
    --output_dir test_data/factQA/sports \
    --test_languages en ja fr es zh
