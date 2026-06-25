#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

START=2025-06-01
END=2026-06-10
STAMP="${START}_${END}"
LANGUAGES="en ja fr es zh"

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir "test_data/factQA/sports/${STAMP}" \
    --training_docs_dir "test_data/train_docs/sports/${STAMP}" \
    --output_dir test_data/benchmark/sports \
    --test_languages ${LANGUAGES} \
    --val_ratio 0.2

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir "test_data/factQA/movie/${STAMP}" \
    --training_docs_dir "test_data/train_docs/movie/${STAMP}" \
    --output_dir test_data/benchmark/movie \
    --test_languages ${LANGUAGES} \
    --val_ratio 0.2

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir "test_data/factQA/music/${STAMP}" \
    --training_docs_dir "test_data/train_docs/music/${STAMP}" \
    --output_dir test_data/benchmark/music \
    --test_languages ${LANGUAGES} \
    --val_ratio 0.2
