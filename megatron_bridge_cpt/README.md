# Qwen3-1.7B CPT with Megatron-Bridge

Runbook for continued pretraining `Qwen/Qwen3-1.7B` on the local 10B-token corpus.

This directory does not install dependencies. Run these scripts on the training machine where Megatron-Bridge, Megatron-Core, Transformer Engine, PyTorch, CUDA/NCCL, and the Qwen tokenizer environment are already prepared.

## What This Uses

- Megatron-Bridge Qwen3 recipe: `qwen3_1p7b_pretrain_config`
- HF import/export: `examples/conversion/convert_checkpoints.py`
- Training entrypoint: `scripts/training/run_recipe.py`
- Pretraining data: Megatron indexed dataset prefix ending in `_text_document`

Megatron-Bridge docs say Qwen3 1.7B is supported, Qwen3 0.6B-4B pretrain uses TP=1/PP=1, and LLM pretraining uses `GPTDatasetConfig` with `data_path`, `blend`, or `blend_per_split`.

## Files

- `env.example.sh` - copy/edit this on the training machine.
- `prepare_10b_jsonl.py` - combines downloaded corpus JSONL files into one `{"text": ...}` JSONL capped by token count.
- `scripts/01_prepare_10b_jsonl.sh` - wrapper around `prepare_10b_jsonl.py`.
- `scripts/02_preprocess_megatron_dataset.sh` - builds Megatron `.bin/.idx`.
- `scripts/03_import_hf_to_megatron.sh` - converts HF Qwen3-1.7B to Megatron checkpoint.
- `scripts/04_train_qwen3_1p7b_cpt.sh` - launches CPT with Megatron-Bridge.
- `scripts/05_export_megatron_to_hf.sh` - converts trained Megatron checkpoint back to HF.

## 0. Configure

```bash
cd /path/to/multilingual
cp megatron_bridge_cpt/env.example.sh megatron_bridge_cpt/env.local.sh
vim megatron_bridge_cpt/env.local.sh
source megatron_bridge_cpt/env.local.sh
```

Important paths to set:

- `MB_REPO`: Megatron-Bridge repo path, for example `/opt/Megatron-Bridge`.
- `RAW_CORPUS_DIR`: directory with downloaded `*.jsonl` corpora, for example `corpus_download/data`.
- `WORKDIR`: fast shared filesystem for prepared data and checkpoints.

## 1. Build 10B Raw JSONL

```bash
bash megatron_bridge_cpt/scripts/01_prepare_10b_jsonl.sh
```

Output:

- `$CPT_JSONL`
- `$CPT_JSONL.manifest.json`

The script uses existing per-row `token_count` when present, so it does not need to re-tokenize.

## 2. Preprocess to Megatron Indexed Dataset

```bash
bash megatron_bridge_cpt/scripts/02_preprocess_megatron_dataset.sh
```

Output:

- `${DATA_OUTPUT_PREFIX}_text_document.bin`
- `${DATA_OUTPUT_PREFIX}_text_document.idx`

The training data prefix is:

```bash
echo "$DATA_PREFIX"
```

## 3. Import HF Checkpoint to Megatron

```bash
bash megatron_bridge_cpt/scripts/03_import_hf_to_megatron.sh
```

Output:

- `$IMPORT_CKPT`

This is a model-weight initialization checkpoint for CPT, not a resume checkpoint with optimizer/RNG state.

## 4. Train CPT

Fresh CPT from imported Qwen3 weights:

```bash
bash megatron_bridge_cpt/scripts/04_train_qwen3_1p7b_cpt.sh
```

Resume interrupted CPT:

```bash
RESUME=1 bash megatron_bridge_cpt/scripts/04_train_qwen3_1p7b_cpt.sh
```

For the default `SEQ_LENGTH=4096` and `GLOBAL_BATCH_SIZE=256`, `TRAIN_ITERS=9540` is about:

```text
4096 * 256 * 9540 = 10,003,415,040 tokens
```

Tune these in `env.local.sh` for the actual GPU count/memory.

## 5. Export Back to HF

By default export reads `$TRAIN_CKPT`. To export a specific iteration, set `MEGATRON_EXPORT_CKPT` to that `iter_*` path.

```bash
MEGATRON_EXPORT_CKPT="$TRAIN_CKPT/iter_0009540" \
bash megatron_bridge_cpt/scripts/05_export_megatron_to_hf.sh
```

Output:

- `$EXPORT_HF`

## Practical Defaults

For 8 GPUs and Qwen3-1.7B:

- `TP=1`
- `PP=1`
- `CP=1`
- `SEQ_LENGTH=4096`
- `MICRO_BATCH_SIZE=1`
- `GLOBAL_BATCH_SIZE=256`
- `LR=1e-5`
- `MIN_LR=1e-6`
- `TRAIN_ITERS=9540`

If memory is tight, lower `MICRO_BATCH_SIZE` stays at `1`, lower `GLOBAL_BATCH_SIZE`, or enable recompute overrides manually. If throughput is too low and memory is fine, raise `MICRO_BATCH_SIZE`.

## References

- Megatron-Bridge HF conversion guide: https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-guide.html
- Megatron-Bridge Qwen docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/qwen.html
- Megatron-Bridge recipe usage: https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html
- Megatron-Bridge checkpointing: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html
