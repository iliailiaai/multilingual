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

cd "${MB_REPO}"
mkdir -p "$(dirname "${EXPORT_HF}")"

"${PYTHON_BIN}" examples/conversion/convert_checkpoints.py export \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${MEGATRON_EXPORT_CKPT}" \
    --hf-path "${EXPORT_HF}"

echo "[OK] Exported Megatron checkpoint ${MEGATRON_EXPORT_CKPT} to ${EXPORT_HF}"
