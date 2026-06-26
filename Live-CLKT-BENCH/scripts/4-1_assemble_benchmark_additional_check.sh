#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

START=2025-06-01
END=2026-06-10
STAMP="${START}_${END}"
LANGUAGES="en ja fr es zh German Dutch Russian Ukrainian Polish Czech Portuguese Italian Urdu Persian Irish Welsh Arabic Hebrew Finnish Estonian Hungarian Turkish Azerbaijani Kazakh Uzbek Indonesian Thai"
LLM_PROVIDER=openrouter
EVAL_MODEL=Qwen/Qwen2.5-3B-Instruct
TP=1
GPU_MEM=0.9

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir "test_data/factQA/sports/${STAMP}" \
    --training_docs_dir "test_data/train_docs/sports/${STAMP}" \
    --output_dir test_data/benchmark_add/sports \
    --test_languages ${LANGUAGES} \
    --eval_model "${EVAL_MODEL}" \
    --domain sports \
    --tp "${TP}" \
    --gpu_mem "${GPU_MEM}" \
    --llm_provider "${LLM_PROVIDER}"

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir "test_data/factQA/movie/${STAMP}" \
    --training_docs_dir "test_data/train_docs/movie/${STAMP}" \
    --output_dir test_data/benchmark_add/movie \
    --test_languages ${LANGUAGES} \
    --eval_model "${EVAL_MODEL}" \
    --domain movie \
    --tp "${TP}" \
    --gpu_mem "${GPU_MEM}" \
    --llm_provider "${LLM_PROVIDER}"

PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir "test_data/factQA/music/${STAMP}" \
    --training_docs_dir "test_data/train_docs/music/${STAMP}" \
    --output_dir test_data/benchmark_add/music \
    --test_languages ${LANGUAGES} \
    --eval_model "${EVAL_MODEL}" \
    --domain music \
    --tp "${TP}" \
    --gpu_mem "${GPU_MEM}" \
    --llm_provider "${LLM_PROVIDER}"
