# Multilingual CPT Patch Notes

This vendor copy is based on Megatron-Bridge commit `da42015c8033495cf6cc6523f8525fdb139a21d2`
as recorded in `.main.commit`.

Search for `Multilingual CPT patch` to find code-level markers.

## Added Files

- `src/megatron/bridge/data/language_tagged_gpt.py`
  - Wraps GPT/Blended datasets and adds `language_ids` / `source_language_ids` per sample.
  - Builds one blended GPT dataset from the per-language manifest produced by the outer CPT scripts.
- `src/megatron/bridge/training/language_steering.py`
  - Loads `.npy` / `.pt` language vectors.
  - Subtracts the selected language vector from hidden states for the first steering layers.
  - Freezes input embeddings, steered transformer layers, and tied output embeddings.

## Modified Files

- `scripts/training/run_recipe.py`
  - Adds CLI flags for `--language_manifest`, `--language_vector_dir`, steering alpha/scaling/layers, and freeze behavior.
  - Replaces the recipe dataset provider with `LanguageTaggedGPTDatasetProvider` when a manifest is supplied.
  - Attaches `LanguageSteeringConfig` to the runtime config when vectors are supplied.
- `src/megatron/bridge/training/setup.py`
  - Registers a pre-wrap hook that installs steering before DDP/optimizer setup.
- `src/megatron/bridge/training/gpt_step.py`
  - Moves `language_ids` and `source_language_ids` through batch preparation.
  - Passes language metadata into decoder block kwargs so the wrapped transformer layers can choose vectors.
