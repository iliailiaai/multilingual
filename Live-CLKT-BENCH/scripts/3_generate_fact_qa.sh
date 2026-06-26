#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

START=2025-06-01
END=2026-06-10
STAMP="${START}_${END}"
LANGUAGES="en ja fr es zh German Dutch Russian Ukrainian Polish Czech Portuguese Italian Urdu Persian Irish Welsh Arabic Hebrew Finnish Estonian Hungarian Turkish Azerbaijani Kazakh Uzbek Indonesian Thai"
SOURCE_LANG=en
LLM_PROVIDER=local
WORKERS=1

PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain sports \
    --training_docs_dir "test_data/train_docs/sports/${STAMP}" \
    --output_dir test_data/factQA/sports \
    --source_lang "${SOURCE_LANG}" \
    --test_languages ${LANGUAGES} \
    --llm_provider "${LLM_PROVIDER}" \
    --workers "${WORKERS}"

PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain movie \
    --training_docs_dir "test_data/train_docs/movie/${STAMP}" \
    --output_dir test_data/factQA/movie \
    --source_lang "${SOURCE_LANG}" \
    --test_languages ${LANGUAGES} \
    --llm_provider "${LLM_PROVIDER}" \
    --workers "${WORKERS}"

PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain music \
    --training_docs_dir "test_data/train_docs/music/${STAMP}" \
    --output_dir test_data/factQA/music \
    --source_lang "${SOURCE_LANG}" \
    --test_languages ${LANGUAGES} \
    --llm_provider "${LLM_PROVIDER}" \
    --workers "${WORKERS}"
