from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np

from ....data import MaterialSample, create_dataset
from ...common import require_torch
from ...polymer_utils import (
    PSMILESTokenizer,
    extract_psmiles,
    polymer_graph_arrays,
    polymer_physics_descriptors,
    split_by_family,
)
from ...workspace import MLWorkspace
from .config import PolyPredictionDataConfig, PolyPredictionModelConfig
from .normalizer import MaskedFeatureNormalizer, MaskedTargetNormalizer

torch = require_torch()


def load_polymer_samples(
    config: PolyPredictionDataConfig,
    *,
    workspace: MLWorkspace,
    samples: Sequence[MaterialSample] | None = None,
) -> list[MaterialSample]:
    if samples is not None:
        loaded = list(samples)
    else:
        kwargs = dict(config.dataset_kwargs)
        kwargs.setdefault("root", workspace.dataset_dir(config.dataset))
        source = create_dataset(config.dataset, **kwargs)
        iterator = source.iter_samples()
        loaded = list(islice(iterator, config.limit)) if config.limit else list(iterator)
    usable: list[MaterialSample] = []
    for sample in loaded:
        try:
            extract_psmiles(sample)
        except (TypeError, ValueError):
            continue
        usable.append(sample)
    if not usable:
        raise ValueError(
            "no usable polymer samples; supply PSMILES in metadata or a PolymerRecord structure"
        )
    return usable


def target_arrays(
    samples: Sequence[MaterialSample],
    model_config: PolyPredictionModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(samples), len(model_config.property_specs)), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for row, sample in enumerate(samples):
        for column, spec in enumerate(model_config.property_specs):
            value = sample.labels.get(spec.name)
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values[row, column] = number
                mask[row, column] = True
    return values, mask


class PolymerPropertyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[MaterialSample],
        *,
        tokenizer: PSMILESTokenizer,
        target_normalizer: MaskedTargetNormalizer,
        condition_normalizer: MaskedFeatureNormalizer,
        max_length: int,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.target_normalizer = target_normalizer
        self.condition_normalizer = condition_normalizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        psmiles = extract_psmiles(sample)
        token_ids, attention_mask = self.tokenizer.encode(
            psmiles,
            max_length=self.max_length,
        )
        graph = polymer_graph_arrays(sample)
        targets, target_mask = self.target_normalizer.encode_row(sample.labels)
        conditions, condition_mask = self.condition_normalizer.encode_row(sample.conditions)
        raw_conditions = np.zeros(len(self.condition_normalizer.names), dtype=np.float32)
        for condition_index, name in enumerate(self.condition_normalizer.names):
            if not condition_mask[condition_index]:
                continue
            raw_conditions[condition_index] = float(sample.conditions[name])
        physics = polymer_physics_descriptors(sample)
        return {
            "id": sample.id,
            "psmiles": psmiles,
            "token_ids": torch.as_tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attention_mask, dtype=torch.bool),
            "node_features": torch.as_tensor(graph.node_features),
            "edge_index": torch.as_tensor(graph.edge_index, dtype=torch.long),
            "edge_features": torch.as_tensor(graph.edge_features),
            "node_weights": torch.as_tensor(graph.node_weights),
            "conditions": torch.as_tensor(conditions),
            "raw_conditions": torch.as_tensor(raw_conditions),
            "condition_mask": torch.as_tensor(condition_mask, dtype=torch.bool),
            "targets": torch.as_tensor(targets),
            "target_mask": torch.as_tensor(target_mask, dtype=torch.bool),
            "physics_descriptors": torch.as_tensor(
                [
                    physics["configurational_entropy_R"],
                    physics["bond_type_count"],
                    physics["fluorine_atomic_fraction"],
                    physics["polar_bond_fraction"],
                    physics["high_entropy_margin_R"],
                ],
                dtype=torch.float32,
            ),
        }


def polymer_property_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    node_parts = []
    edge_parts = []
    edge_feature_parts = []
    node_weight_parts = []
    graph_index_parts = []
    offset = 0
    for graph_index, sample in enumerate(batch):
        count = int(sample["node_features"].shape[0])
        node_parts.append(sample["node_features"])
        edge_parts.append(sample["edge_index"] + offset)
        edge_feature_parts.append(sample["edge_features"])
        node_weight_parts.append(sample["node_weights"])
        graph_index_parts.append(torch.full((count,), graph_index, dtype=torch.long))
        offset += count
    return {
        "id": [sample["id"] for sample in batch],
        "psmiles": [sample["psmiles"] for sample in batch],
        "token_ids": torch.stack([sample["token_ids"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
        "node_features": torch.cat(node_parts),
        "edge_index": torch.cat(edge_parts, dim=1),
        "edge_features": torch.cat(edge_feature_parts),
        "node_weights": torch.cat(node_weight_parts),
        "graph_index": torch.cat(graph_index_parts),
        "conditions": torch.stack([sample["conditions"] for sample in batch]),
        "raw_conditions": torch.stack([sample["raw_conditions"] for sample in batch]),
        "condition_mask": torch.stack([sample["condition_mask"] for sample in batch]),
        "targets": torch.stack([sample["targets"] for sample in batch]),
        "target_mask": torch.stack([sample["target_mask"] for sample in batch]),
        "physics_descriptors": torch.stack([sample["physics_descriptors"] for sample in batch]),
    }


@dataclass(slots=True)
class PolyPredictionDataModule:
    train: Any
    valid: Any
    test: Any
    tokenizer: PSMILESTokenizer
    target_normalizer: MaskedTargetNormalizer
    condition_normalizer: MaskedFeatureNormalizer
    split_sizes: dict[str, int]

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test


def prepare_poly_prediction_data(
    data_config: PolyPredictionDataConfig,
    model_config: PolyPredictionModelConfig,
    *,
    workspace: MLWorkspace,
    samples: Sequence[MaterialSample] | None = None,
) -> PolyPredictionDataModule:
    loaded = load_polymer_samples(data_config, workspace=workspace, samples=samples)
    train_samples, valid_samples, test_samples = split_by_family(
        loaded,
        train_ratio=data_config.train_ratio,
        valid_ratio=data_config.valid_ratio,
        test_ratio=data_config.test_ratio,
        seed=data_config.seed,
    )
    if not train_samples:
        raise ValueError("the training split is empty")
    tokenizer = PSMILESTokenizer.fit(
        (extract_psmiles(sample) for sample in train_samples),
        min_frequency=data_config.min_token_frequency,
    )
    model_config.vocab_size = tokenizer.vocab_size
    values, mask = target_arrays(train_samples, model_config)
    if not mask.any():
        names = ", ".join(spec.name for spec in model_config.property_specs)
        raise ValueError(f"the training split has none of the configured targets: {names}")
    target_normalizer = MaskedTargetNormalizer(model_config.property_specs).fit(values, mask)
    condition_normalizer = MaskedFeatureNormalizer(model_config.condition_names).fit(
        [sample.conditions for sample in train_samples]
    )

    def loader(subset: Sequence[MaterialSample], *, shuffle: bool):
        dataset = PolymerPropertyDataset(
            subset,
            tokenizer=tokenizer,
            target_normalizer=target_normalizer,
            condition_normalizer=condition_normalizer,
            max_length=model_config.max_length,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle and bool(subset),
            num_workers=data_config.num_workers,
            collate_fn=polymer_property_collate,
            pin_memory=True,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return PolyPredictionDataModule(
        train=loader(train_samples, shuffle=True),
        valid=loader(valid_samples, shuffle=False),
        test=loader(test_samples, shuffle=False),
        tokenizer=tokenizer,
        target_normalizer=target_normalizer,
        condition_normalizer=condition_normalizer,
        split_sizes={
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples),
        },
    )


__all__ = [
    "PolyPredictionDataModule",
    "PolymerPropertyDataset",
    "load_polymer_samples",
    "polymer_property_collate",
    "prepare_poly_prediction_data",
    "target_arrays",
]
