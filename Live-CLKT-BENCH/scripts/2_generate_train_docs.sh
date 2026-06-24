#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain sports \
    --entity_file test_data/entities/sports/2026-03-01_2026-04-10.json \
    --output_dir test_data/train_docs/sports \
    --test_languages en ja fr es zh
