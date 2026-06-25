#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

START=2025-06-01
END=2026-06-10
MAX_E=10

PYTHONPATH=lib python3 data_generation/0_collect_entity.py \
    --domain sports \
    --start_str "${START}" \
    --end_str "${END}" \
    --output_dir test_data/entities \
    --max_entity "${MAX_E}"

PYTHONPATH=lib python3 data_generation/0_collect_entity.py \
    --domain movie \
    --start_str "${START}" \
    --end_str "${END}" \
    --output_dir test_data/entities \
    --max_entity "${MAX_E}"

PYTHONPATH=lib python3 data_generation/0_collect_entity.py \
    --domain music \
    --start_str "${START}" \
    --end_str "${END}" \
    --output_dir test_data/entities \
    --max_entity "${MAX_E}"
