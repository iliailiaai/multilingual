# Qwen3-1.7B CPT with Megatron-Bridge + Language Steering

Runbook for continued pretraining `Qwen/Qwen3-1.7B` on the local multilingual 10B-token corpus.

This directory does not install dependencies. Run these scripts on the training machine where Megatron-Bridge, Megatron-Core, Transformer Engine, PyTorch, CUDA/NCCL, and the Qwen tokenizer environment are already prepared.

## What This Uses

- Megatron-Bridge Qwen3 recipe: `qwen3_1p7b_pretrain_config`
- HF import/export: `examples/conversion/convert_checkpoints.py`
- Training entrypoint: `scripts/training/run_recipe.py`
- One Megatron indexed dataset prefix per source corpus language
- Runtime language steering for the first `LANGUAGE_STEERING_LAYERS=7` transformer layers
- `language_ids` from the dataloader batch to choose the vector per sample

Megatron-Bridge docs say Qwen3 1.7B is supported, Qwen3 0.6B-4B pretrain uses TP=1/PP=1, and LLM pretraining uses `GPTDatasetConfig` with `data_path`, `blend`, or `blend_per_split`.

## Files

- `env.example.sh` - copy/edit this on the training machine.
- `prepare_10b_jsonl.py` - writes per-language `{"text": ...}` JSONL files and a language manifest.
- `scripts/01_prepare_10b_jsonl.sh` - builds per-language raw JSONL files.
- `scripts/02_preprocess_megatron_dataset.sh` - builds one Megatron `.bin/.idx` prefix per language.
- `scripts/03_import_hf_to_megatron.sh` - converts HF Qwen3-1.7B to Megatron checkpoint.
- `scripts/04_train_qwen3_1p7b_cpt.sh` - launches CPT with language steering.
- `scripts/05_export_megatron_to_hf.sh` - converts trained Megatron checkpoint back to HF.

Bridge-local changes live in:

- `Megatron-Bridge/src/megatron/bridge/data/language_tagged_gpt.py`
- `Megatron-Bridge/src/megatron/bridge/training/language_steering.py`
- patched `Megatron-Bridge/src/megatron/bridge/training/gpt_step.py`
- patched `Megatron-Bridge/src/megatron/bridge/training/setup.py`
- patched `Megatron-Bridge/scripts/training/run_recipe.py`

## 0. Configure

```bash
cd /path/to/multilingual
cp megatron_bridge_cpt/env.example.sh megatron_bridge_cpt/env.local.sh
vim megatron_bridge_cpt/env.local.sh
source megatron_bridge_cpt/env.local.sh

export TAG=26.06
docker pull nvcr.io/nvidia/nemo:${TAG}

cd /home/st107742/projects/multilingual

docker run --rm -it \
  --gpus '"device=2,3,4"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -w /workdir \
  -v "$PWD":/workdir \
  -v /home/st107742/projects/multilingual/main:/workdir/main \
  --entrypoint bash \
  nvcr.io/nvidia/nemo:${TAG}
```

Important paths to set:

- `MB_REPO`: Megatron-Bridge repo path. By default this points to `megatron_bridge_cpt/Megatron-Bridge`.
- `RAW_CORPUS_DIR`: directory with downloaded `*.jsonl` corpora, for example `corpus_download/data`.
- `WORKDIR`: fast shared filesystem for prepared data and checkpoints.
- `LANGUAGE_VECTOR_DIR`: directory with language vectors named like `eng.npy`, `deu.npy`, etc.
- `PREPROCESS_SCRIPT`: set this if Megatron-LM `tools/preprocess_data.py` is not under `$MB_REPO/3rdparty/Megatron-LM`.

## 1. Build Per-Language 10B Raw JSONL

```bash
bash megatron_bridge_cpt/scripts/01_prepare_10b_jsonl.sh
```

Output:

- `$LANGUAGE_JSONL_DIR/*.jsonl`
- `$LANGUAGE_MANIFEST`

The manifest stores:

- source corpus language, for example `de` or `en_instruct`
- steering vector language, for example `deu` or `eng`
- numeric `language_id`
- raw JSONL path
- target Megatron prefix path
- token/doc counts

For raw instruct rows with `messages` or `conversations`, `prepare_10b_jsonl.py` renders text through the Qwen tokenizer chat template. Already-rendered rows with only `text` are passed through.

## 2. Preprocess Per-Language Megatron Datasets

```bash
bash megatron_bridge_cpt/scripts/02_preprocess_megatron_dataset.sh
```

Output:

```text
$LANGUAGE_DATA_PREFIX_DIR/<language>/qwen3_cpt_<language>_text_document.bin
$LANGUAGE_DATA_PREFIX_DIR/<language>/qwen3_cpt_<language>_text_document.idx
```

These prefixes are listed in `$LANGUAGE_MANIFEST` and loaded by the custom language-tagged dataset provider.

## 3. Import HF Checkpoint to Megatron

```bash
bash megatron_bridge_cpt/scripts/03_import_hf_to_megatron.sh
```

Output:

- `$IMPORT_CKPT`

This is a model-weight initialization checkpoint for CPT, not a resume checkpoint with optimizer/RNG state.

## 4. Train CPT with Language Steering

Fresh CPT from imported Qwen3 weights:

```bash
bash megatron_bridge_cpt/scripts/04_train_qwen3_1p7b_cpt.sh
```

Resume interrupted CPT:

```bash
RESUME=1 bash megatron_bridge_cpt/scripts/04_train_qwen3_1p7b_cpt.sh
```

The training script passes:

```bash
--language_manifest "$LANGUAGE_MANIFEST"
--language_vector_dir "$LANGUAGE_VECTOR_DIR"
--language_steering_layers "$LANGUAGE_STEERING_LAYERS"
--language_vector_layer_offset "$LANGUAGE_VECTOR_LAYER_OFFSET"
```

The dataloader tags each sample with `language_ids`. The model wrapper subtracts the selected language vector after each of the first 7 transformer layer forwards. It also freezes input embeddings, the first 7 transformer layers, and output embeddings only when they are tied.

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
- `LANGUAGE_STEERING_LAYERS=7`
- `LANGUAGE_STEERING_SCALING=none`

Keep `PP=1` for the first steering run. Pipeline parallelism can work later, but the first 7 layers and their language ids are easiest to reason about in a single pipeline stage.

## References

- Megatron-Bridge HF conversion guide: https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-guide.html
- Megatron-Bridge Qwen docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/qwen.html
- Megatron-Bridge recipe usage: https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html
- Megatron-Bridge checkpointing: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html
