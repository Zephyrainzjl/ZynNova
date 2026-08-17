from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..views.generative import GenerativeTensorView
from ..views.transformer import TransformerInputView


def collate_generative_views(batch: Sequence[GenerativeTensorView]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    levels = {item.level for item in batch}
    if len(levels) != 1:
        raise ValueError("a generative batch cannot mix atom- and unit-level views")
    output: dict[str, object] = {
        "level": batch[0].level,
        "node_type": np.stack([item.node_type for item in batch]),
        "node_mask": np.stack([item.node_mask for item in batch]),
        "node_features": np.stack([item.node_features for item in batch]),
        "edge_type": np.stack([item.edge_type for item in batch]),
        "edge_mask": np.stack([item.edge_mask for item in batch]),
        "composition_logits": np.stack([item.composition_logits for item in batch]),
        "composition_mask": np.stack([item.composition_mask for item in batch]),
        "transition_logits": np.stack([item.transition_logits for item in batch]),
        "transition_mask": np.stack([item.transition_mask for item in batch]),
        "continuous_features": np.stack([item.continuous_features for item in batch]),
        "continuous_feature_mask": np.stack(
            [item.continuous_feature_mask for item in batch]
        ),
    }
    if all(item.coordinates is not None for item in batch):
        output["coordinates"] = np.stack([item.coordinates for item in batch])
        output["coordinate_mask"] = np.stack([item.coordinate_mask for item in batch])
    return output


def collate_transformer_views(
    batch: Sequence[TransformerInputView],
    *,
    pad_token: str = "[PAD]",
) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_length = max(len(item.tokens) for item in batch)
    tokens: list[list[str]] = []
    token_type_ids = np.zeros((len(batch), max_length), dtype=np.int64)
    attention_mask = np.zeros((len(batch), max_length), dtype=bool)
    for batch_index, item in enumerate(batch):
        pad_length = max_length - len(item.tokens)
        tokens.append(item.tokens + [pad_token] * pad_length)
        token_type_ids[batch_index, : len(item.tokens)] = item.token_type_ids
        attention_mask[batch_index, : len(item.tokens)] = item.attention_mask
    return {
        "tokens": tokens,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "continuous_features": np.stack([item.continuous_features for item in batch]),
        "continuous_feature_mask": np.stack(
            [item.continuous_feature_mask for item in batch]
        ),
    }
