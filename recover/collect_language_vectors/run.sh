#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME=Qwen/Qwen3-1.7B
# #Qwen/Qwen2-7B-Instruct or google/gemma-2-2b-it or meta-llama/Llama-3.1-8B-Instruct are used in the paper
MODEL_ID="${MODEL_NAME##*/}"
OUTPUT_DIR="language_vectors_bucket/flores_plus/${MODEL_ID}/full"
FAILED_LOG="${OUTPUT_DIR}/failed_languages.log"

LANGUAGES=(
    eng  # en / English
    jpn  # ja / Japanese
    fra  # fr / French
    spa  # es / Spanish
    cmn  # zh / Chinese, filtered to Hans in collect.py
    deu  # German
    nld  # Dutch
    rus  # Russian
    ukr  # Ukrainian
    pol  # Polish
    ces  # Czech
    por  # Portuguese
    ita  # Italian
    urd  # Urdu
    pes  # Persian
    gle  # Irish
    cym  # Welsh
    arb  # Arabic
    heb  # Hebrew
    fin  # Finnish
    est  # Estonian
    hun  # Hungarian
    tur  # Turkish
    aze  # Azerbaijani
    kaz  # Kazakh
    uzb  # Uzbek
    ind  # Indonesian
    tha  # Thai
)

mkdir -p "${OUTPUT_DIR}"
: > "${FAILED_LOG}"

for LANG in "${LANGUAGES[@]}"
do 
    if [ -f "${OUTPUT_DIR}/${LANG}.npy" ]; then
        echo "[SKIP] ${LANG}: vector already exists at ${OUTPUT_DIR}/${LANG}.npy"
        continue
    fi

    if ! python collect.py \
        --language $LANG \
        --model_name_or_path $MODEL_NAME \
        --dataset_name openlanguagedata/flores_plus \
        --split dev \
        --num_layers 7 \
        --skip_existing
    then
        echo "[WARN] ${LANG}: failed, continuing. See traceback above."
        echo "${LANG}" >> "${FAILED_LOG}"
    fi
done

if [ -s "${FAILED_LOG}" ]; then
    echo "[WARN] Some languages failed:"
    cat "${FAILED_LOG}"
    exit 1
fi
