# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Multilingual CPT patch: runtime language steering for GPT/Qwen-style Megatron-Core models."""

from __future__ import annotations

import json
import os
import types
import weakref
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from megatron.bridge.utils.common_utils import print_rank_0


@dataclass
class LanguageSteeringConfig:
    """Configuration for fixed language-vector steering."""

    vector_dir: str
    manifest_path: str
    alpha: float = 1.0
    scaling_mode: Optional[str] = None
    remove_content: bool = True
    max_steering_layers: int = 7
    vector_layer_offset: int = 1
    freeze_embeddings_and_steered_layers: bool = True


def _load_raw_vectors(vector_dir: str) -> dict[str, np.ndarray]:
    vectors = {}
    for file_name in os.listdir(vector_dir):
        path = os.path.join(vector_dir, file_name)
        if file_name.endswith(".npy"):
            vectors[file_name[:-4]] = np.load(path)
        elif file_name.endswith(".pt"):
            vectors[file_name.split(".")[0]] = torch.load(path, map_location="cpu", weights_only=True).numpy()
    if not vectors:
        raise ValueError(f"No .npy/.pt language vectors found in {vector_dir}")
    return vectors


def _remove_content(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    avg = np.stack([vector[:, :1] for vector in vectors.values()]).mean(axis=0)
    return {language: vector - avg for language, vector in vectors.items()}


def _load_language_id_to_vector_language(manifest_path: str) -> dict[int, str]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if "language_id_to_vector_language" in manifest:
        return {int(idx): language for idx, language in manifest["language_id_to_vector_language"].items()}

    mapping = {}
    for entry in manifest.get("languages", []):
        mapping[int(entry["language_id"])] = entry["vector_language"]
    if not mapping:
        raise ValueError(f"No language id mapping found in manifest: {manifest_path}")
    return mapping


class LanguageSteering(nn.Module):
    """Subtract language vectors from hidden states for selected transformer layers."""

    def __init__(self, config: LanguageSteeringConfig) -> None:
        super().__init__()
        self.config = config
        raw_vectors = _load_raw_vectors(config.vector_dir)
        if config.remove_content:
            raw_vectors = _remove_content(raw_vectors)

        self.language_id_to_vector_language = _load_language_id_to_vector_language(config.manifest_path)
        max_language_id = max(self.language_id_to_vector_language)
        vectors_by_id = []
        for language_id in range(max_language_id + 1):
            vector_language = self.language_id_to_vector_language[language_id]
            if vector_language not in raw_vectors:
                available = ", ".join(sorted(raw_vectors))
                raise ValueError(
                    f"Missing vector for language id {language_id}: {vector_language}. "
                    f"Available vectors: {available}"
                )
            vector = raw_vectors[vector_language]
            if vector.ndim == 3 and vector.shape[1] == 1:
                vector = vector[:, 0, :]
            elif vector.ndim != 2:
                raise ValueError(
                    f"Expected vector shape [layers, 1, hidden] or [layers, hidden] for {vector_language}, "
                    f"got {vector.shape}"
                )
            vectors_by_id.append(torch.tensor(vector, dtype=torch.float32))

        vectors = torch.stack(vectors_by_id, dim=1)
        self.register_buffer("vectors", vectors, persistent=False)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int, language_ids: torch.Tensor) -> torch.Tensor:
        vector_idx = layer_idx + self.config.vector_layer_offset
        if vector_idx < 0 or vector_idx >= self.vectors.shape[0]:
            return hidden_states

        language_ids = language_ids.to(device=hidden_states.device, dtype=torch.long).view(-1)
        batch_size = hidden_states.shape[1]
        if language_ids.numel() == 1 and batch_size != 1:
            language_ids = language_ids.expand(batch_size)
        if language_ids.numel() != batch_size:
            raise ValueError(
                f"language_ids batch mismatch: got {language_ids.numel()} ids for hidden batch {batch_size}"
            )

        delta = self.vectors[vector_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
        delta = delta.index_select(0, language_ids)

        scaling_mode = self.config.scaling_mode
        if scaling_mode in (None, "none"):
            pass
        elif scaling_mode == "factor":
            delta = delta * self.config.alpha
        elif scaling_mode == "norm":
            delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.config.alpha
        elif scaling_mode == "relative_norm":
            hidden_norm = hidden_states.norm(dim=-1, keepdim=True)
            delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta = delta.unsqueeze(0) * hidden_norm * self.config.alpha
            return hidden_states - delta.to(hidden_states.dtype)
        else:
            raise ValueError(f"Unknown language steering scaling mode: {scaling_mode}")

        delta = delta.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        return hidden_states - delta.to(hidden_states.dtype)


def _global_layer_idx(layer: nn.Module, fallback_idx: int) -> int:
    layer_number = getattr(layer, "layer_number", None)
    if isinstance(layer_number, int):
        return layer_number - 1
    return fallback_idx


def _get_decoder_layers(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise ValueError("Language steering expected a GPT-like model with .decoder")
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("Language steering expected model.decoder.layers")
    return decoder, list(layers)


def install_language_steering(model: nn.Module, config: LanguageSteeringConfig) -> None:
    """Install instance-scoped forward wrappers on one Megatron model chunk."""

    if getattr(model, "_language_steering_installed", False):
        return

    decoder, layers = _get_decoder_layers(model)
    steering = LanguageSteering(config)
    try:
        steering.to(device=next(model.parameters()).device)
    except StopIteration:
        pass
    model.language_steering = steering

    decoder_ref = weakref.ref(decoder)
    wrapped_layers = 0
    for local_idx, layer in enumerate(layers):
        layer_idx = _global_layer_idx(layer, local_idx)
        if layer_idx >= config.max_steering_layers:
            continue
        if getattr(layer, "_language_steering_layer_forward_patched", False):
            continue

        orig_layer_forward = layer.forward

        def _layer_forward(
            self,
            *args,
            _orig_forward=orig_layer_forward,
            _layer_idx=layer_idx,
            **kwargs,
        ):
            output = _orig_forward(*args, **kwargs)
            decoder_obj = decoder_ref()
            language_ids = None
            if decoder_obj is not None:
                language_ids = getattr(decoder_obj, "_language_steering_current_language_ids", None)
            if language_ids is None:
                return output

            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states = steering(hidden_states, _layer_idx, language_ids)
            if isinstance(output, tuple):
                return (hidden_states, *output[1:])
            return hidden_states

        layer.forward = types.MethodType(_layer_forward, layer)
        layer._language_steering_layer_forward_patched = True
        wrapped_layers += 1

    orig_decoder_forward = decoder.forward

    def _decoder_forward(self, *args, language_ids=None, source_language_ids=None, **kwargs):
        previous = getattr(self, "_language_steering_current_language_ids", None)
        had_previous = hasattr(self, "_language_steering_current_language_ids")
        self._language_steering_current_language_ids = language_ids if language_ids is not None else source_language_ids
        try:
            return orig_decoder_forward(*args, **kwargs)
        finally:
            if had_previous:
                self._language_steering_current_language_ids = previous
            else:
                delattr(self, "_language_steering_current_language_ids")

    decoder.forward = types.MethodType(_decoder_forward, decoder)
    model._language_steering_installed = True
    print_rank_0(f"Installed language steering on {wrapped_layers} transformer layer(s)")


def _freeze_module(module: Any) -> int:
    if module is None:
        return 0
    frozen = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            frozen += parameter.numel()
        parameter.requires_grad = False
    return frozen


def freeze_embeddings_and_steered_layers(model: nn.Module, config: LanguageSteeringConfig) -> None:
    """Freeze embeddings and transformer layers covered by steering."""

    if getattr(model, "_language_steering_freeze_applied", False):
        return

    frozen_params = 0
    frozen_params += _freeze_module(getattr(model, "embedding", None))

    if getattr(model, "share_embeddings_and_output_weights", False):
        frozen_params += _freeze_module(getattr(model, "output_layer", None))

    _, layers = _get_decoder_layers(model)
    for local_idx, layer in enumerate(layers):
        layer_idx = _global_layer_idx(layer, local_idx)
        if layer_idx < config.max_steering_layers:
            frozen_params += _freeze_module(layer)

    model._language_steering_freeze_applied = True
    print_rank_0(f"Frozen {frozen_params:,} parameter values for language steering")


def create_language_steering_pre_wrap_hook(config: LanguageSteeringConfig):
    """Create a Bridge pre-wrap hook that installs steering before DDP/optimizer setup."""

    def language_steering_pre_wrap_hook(model: list[nn.Module]) -> list[nn.Module]:
        for model_chunk in model:
            install_language_steering(model_chunk, config)
            if config.freeze_embeddings_and_steered_layers:
                freeze_embeddings_and_steered_layers(model_chunk, config)
        return model

    return language_steering_pre_wrap_hook
