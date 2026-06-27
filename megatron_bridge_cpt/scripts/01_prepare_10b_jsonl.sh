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

mkdir -p "${LANGUAGE_JSONL_DIR}" "${LANGUAGE_DATA_PREFIX_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/megatron_bridge_cpt/prepare_10b_jsonl.py" \
    --inputs "${RAW_CORPUS_DIR}"/*.jsonl \
    --by-language \
    --output-dir "${LANGUAGE_JSONL_DIR}" \
    --manifest "${LANGUAGE_MANIFEST}" \
    --megatron-prefix-dir "${LANGUAGE_DATA_PREFIX_DIR}" \
    --target-tokens "${TARGET_TOKENS}" \
    --tokenizer "${HF_MODEL}" \
    --shuffle-files
