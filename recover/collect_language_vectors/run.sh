#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=Qwen/Qwen3-1.7B
# #Qwen/Qwen2-7B-Instruct or google/gemma-2-2b-it or meta-llama/Llama-3.1-8B-Instruct are used in the paper

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

for LANG in "${LANGUAGES[@]}"
do 
    python collect.py \
        --language $LANG \
        --model_name_or_path $MODEL_NAME \
        --dataset_name openlanguagedata/flores_plus \
        --split dev \
        --num_layers 7
done
