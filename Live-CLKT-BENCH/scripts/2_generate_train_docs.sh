#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

START=2025-06-01
END=2026-06-10
STAMP="${START}_${END}"
LANGUAGES="en ja fr es zh"
LLM_PROVIDER=openrouter

PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain sports \
    --entity_file "test_data/entities/sports/${STAMP}.json" \
    --output_dir test_data/train_docs/sports \
    --test_languages ${LANGUAGES} \
    --llm_provider "${LLM_PROVIDER}"

PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain movie \
    --entity_file "test_data/entities/movie/${STAMP}.json" \
    --output_dir test_data/train_docs/movie \
    --test_languages ${LANGUAGES} \
    --llm_provider "${LLM_PROVIDER}"

PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain music \
    --entity_file "test_data/entities/music/${STAMP}.json" \
    --output_dir test_data/train_docs/music \
    --test_languages ${LANGUAGES} \
    --llm_provider "${LLM_PROVIDER}"
