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

if [ ! -f "${LANGUAGE_MANIFEST}" ]; then
    echo "[ERROR] Missing LANGUAGE_MANIFEST: ${LANGUAGE_MANIFEST}" >&2
    echo "Run scripts/01_prepare_10b_jsonl.sh first." >&2
    exit 1
fi

PREPROCESS_SCRIPT="${PREPROCESS_SCRIPT:-}"
if [ -z "${PREPROCESS_SCRIPT}" ]; then
    for candidate in \
        "${MB_REPO}/3rdparty/Megatron-LM/tools/preprocess_data.py" \
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

while IFS=$'\t' read -r language jsonl_path output_prefix
do
    if [ -z "${language}" ]; then
        continue
    fi
    if [ -z "${output_prefix}" ] || [ "${output_prefix}" = "None" ]; then
        echo "[ERROR] Missing megatron_output_prefix for ${language} in ${LANGUAGE_MANIFEST}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${output_prefix}")"
    if [ -f "${output_prefix}_text_document.bin" ] && [ -f "${output_prefix}_text_document.idx" ]; then
        echo "[SKIP] ${language}: ${output_prefix}_text_document.bin/.idx already exist"
        continue
    fi

    echo "[PREPROCESS] ${language}: ${jsonl_path} -> ${output_prefix}_text_document"
    "${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" \
        --input "${jsonl_path}" \
        --json-keys text \
        --output-prefix "${output_prefix}" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "${HF_MODEL}" \
        --append-eod \
        --workers "${PREPROCESS_WORKERS}" \
        --log-interval 10000
done < <("${PYTHON_BIN}" - "${LANGUAGE_MANIFEST}" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

for entry in manifest["languages"]:
    print(f"{entry['language']}\t{entry['jsonl_path']}\t{entry['megatron_output_prefix']}")
PY
)

echo "[OK] Per-language Megatron prefixes are listed in ${LANGUAGE_MANIFEST}"
