#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/megatron_bridge_cpt/env.local.sh}"

if [ -f "${ENV_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
else
    # shellcheck source=/dev/null
    source "${ROOT_DIR}/megatron_bridge_cpt/env.example.sh"
fi

if [ ! -f "${CPT_JSONL}" ]; then
    echo "[ERROR] Missing CPT_JSONL: ${CPT_JSONL}" >&2
    exit 1
fi

PREPROCESS_SCRIPT="${PREPROCESS_SCRIPT:-}"
if [ -z "${PREPROCESS_SCRIPT}" ]; then
    for candidate in \
        "${MB_REPO}/Megatron-LM/tools/preprocess_data.py" \
        "${MB_REPO}/megatron-lm/tools/preprocess_data.py" \
        "${MB_REPO}/tools/preprocess_data.py"
    do
        if [ -f "${candidate}" ]; then
            PREPROCESS_SCRIPT="${candidate}"
            break
        fi
    done
fi

if [ -z "${PREPROCESS_SCRIPT}" ] || [ ! -f "${PREPROCESS_SCRIPT}" ]; then
    echo "[ERROR] Could not find Megatron preprocess_data.py. Set PREPROCESS_SCRIPT explicitly." >&2
    exit 1
fi

mkdir -p "$(dirname "${DATA_OUTPUT_PREFIX}")"

"${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" \
    --input "${CPT_JSONL}" \
    --json-keys text \
    --output-prefix "${DATA_OUTPUT_PREFIX}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${HF_MODEL}" \
    --dataset-impl mmap \
    --append-eod \
    --workers "${PREPROCESS_WORKERS}" \
    --log-interval 10000

echo "[OK] Megatron data prefix: ${DATA_PREFIX}"
