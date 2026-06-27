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

mkdir -p "$(dirname "${CPT_JSONL}")"

"${PYTHON_BIN}" "${ROOT_DIR}/megatron_bridge_cpt/prepare_10b_jsonl.py" \
    --inputs "${RAW_CORPUS_DIR}"/*.jsonl \
    --output "${CPT_JSONL}" \
    --target-tokens "${TARGET_TOKENS}" \
    --shuffle-files
