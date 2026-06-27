# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Multilingual CPT patch: language-tagged GPT datasets for runtime steering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider, GPTDatasetConfig
from megatron.bridge.utils.common_utils import print_rank_0


class LanguageTaggedDataset(Dataset):
    """Attach language ids to samples from a GPTDataset or BlendedDataset."""

    def __init__(self, dataset: Dataset, dataset_id_to_language_id: list[int]) -> None:
        self.dataset = dataset
        self.dataset_id_to_language_id = dataset_id_to_language_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.dataset[idx])
        dataset_id = sample.get("dataset_id", 0)
        if isinstance(dataset_id, torch.Tensor):
            dataset_id = dataset_id.item()
        dataset_id = int(dataset_id)
        language_id = self.dataset_id_to_language_id[dataset_id]
        language_id_tensor = torch.tensor(language_id, dtype=torch.long)
        sample["language_ids"] = language_id_tensor
        sample["source_language_ids"] = language_id_tensor
        return sample


def _getattr_or_default(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


@dataclass
class LanguageTaggedGPTDatasetProvider(DatasetProvider):
    """Build blended GPT datasets and tag each sample with a steering language id."""

    manifest_path: str
    seq_length: int
    split: str = "9999,8,2"
    random_seed: int = 1234
    reset_attention_mask: bool = False
    reset_position_ids: bool = False
    eod_mask_loss: bool = False
    skip_getting_attention_mask_from_dataset: bool = True
    num_dataset_builder_threads: int = 1
    path_to_cache: Optional[str] = None
    mmap_bin_files: bool = True
    mid_level_dataset_surplus: float = 0.005
    add_extra_token_to_sequence: bool = True
    blend_weight_key: str = "written_tokens"

    @classmethod
    def from_gpt_config(
        cls,
        base_config: GPTDatasetConfig,
        manifest_path: str,
        *,
        blend_weight_key: str = "written_tokens",
    ) -> "LanguageTaggedGPTDatasetProvider":
        return cls(
            manifest_path=manifest_path,
            seq_length=base_config.seq_length,
            split=_getattr_or_default(base_config, "split", "9999,8,2"),
            random_seed=_getattr_or_default(base_config, "random_seed", 1234),
            reset_attention_mask=_getattr_or_default(base_config, "reset_attention_mask", False),
            reset_position_ids=_getattr_or_default(base_config, "reset_position_ids", False),
            eod_mask_loss=_getattr_or_default(base_config, "eod_mask_loss", False),
            skip_getting_attention_mask_from_dataset=_getattr_or_default(
                base_config,
                "skip_getting_attention_mask_from_dataset",
                True,
            ),
            num_dataset_builder_threads=_getattr_or_default(base_config, "num_dataset_builder_threads", 1),
            path_to_cache=_getattr_or_default(base_config, "path_to_cache", None),
            mmap_bin_files=_getattr_or_default(base_config, "mmap_bin_files", True),
            mid_level_dataset_surplus=_getattr_or_default(base_config, "mid_level_dataset_surplus", 0.005),
            add_extra_token_to_sequence=_getattr_or_default(base_config, "add_extra_token_to_sequence", True),
            blend_weight_key=blend_weight_key,
            dataloader_type=_getattr_or_default(base_config, "dataloader_type", "single"),
            num_workers=_getattr_or_default(base_config, "num_workers", 8),
            data_sharding=_getattr_or_default(base_config, "data_sharding", True),
            pin_memory=_getattr_or_default(base_config, "pin_memory", True),
            drop_last=_getattr_or_default(base_config, "drop_last", True),
            persistent_workers=_getattr_or_default(base_config, "persistent_workers", True),
            trust_remote_code=_getattr_or_default(base_config, "trust_remote_code", None),
        )

    def _load_language_entries(self) -> list[dict[str, Any]]:
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        entries = []
        for entry in manifest.get("languages", []):
            prefix = entry.get("megatron_prefix")
            if not prefix:
                continue
            if int(entry.get("written_tokens", 0)) <= 0:
                continue
            entries.append(entry)
        if not entries:
            raise ValueError(f"No usable language entries found in manifest: {self.manifest_path}")
        return entries

    def _build_gpt_dataset_config(
        self,
        context: DatasetBuildContext,
        entries: list[dict[str, Any]],
    ) -> GPTDatasetConfig:
        prefixes = [entry["megatron_prefix"] for entry in entries]
        weights = [float(entry.get(self.blend_weight_key, 1.0)) for entry in entries]
        if any(weight <= 0 for weight in weights):
            weights = [1.0 for _ in entries]

        dataset_cfg = GPTDatasetConfig(
            random_seed=self.random_seed,
            reset_attention_mask=self.reset_attention_mask,
            reset_position_ids=self.reset_position_ids,
            eod_mask_loss=self.eod_mask_loss,
            seq_length=self.seq_length,
            num_dataset_builder_threads=self.num_dataset_builder_threads,
            blend=(prefixes, weights),
            blend_per_split=None,
            split=self.split,
            data_sharding=self.data_sharding,
            dataloader_type=self.dataloader_type,
            skip_getting_attention_mask_from_dataset=self.skip_getting_attention_mask_from_dataset,
            path_to_cache=self.path_to_cache,
            mmap_bin_files=self.mmap_bin_files,
            mid_level_dataset_surplus=self.mid_level_dataset_surplus,
            add_extra_token_to_sequence=self.add_extra_token_to_sequence,
            tokenizer=context.tokenizer,
        )

        if getattr(dataset_cfg, "token_dtype_code", None) is None:
            vocab_size = getattr(context.tokenizer, "vocab_size", None)
            if vocab_size is not None:
                import numpy

                dataset_cfg.token_dtype_code = 4 if vocab_size > numpy.iinfo(numpy.uint16).max + 1 else 8
        dataset_cfg.finalize()
        return dataset_cfg

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        entries = self._load_language_entries()
        dataset_cfg = self._build_gpt_dataset_config(context, entries)
        dataset_id_to_language_id = [int(entry["language_id"]) for entry in entries]

        print_rank_0(
            " > language-tagged GPT blend: "
            + ", ".join(f"{entry['language']}->{entry['vector_language']}" for entry in entries)
        )

        train_ds, valid_ds, test_ds = pretrain_train_valid_test_datasets_provider(
            [context.train_samples, context.valid_samples, context.test_samples],
            dataset_cfg,
        )

        def wrap(dataset: Optional[Dataset]) -> Optional[LanguageTaggedDataset]:
            if dataset is None:
                return None
            return LanguageTaggedDataset(dataset, dataset_id_to_language_id)

        return wrap(train_ds), wrap(valid_ds), wrap(test_ds)
