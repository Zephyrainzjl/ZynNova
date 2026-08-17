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
    split_by_family,
    tokenize_psmiles,
)
from ...prediction.PolyPrediction.normalizer import (
    MaskedFeatureNormalizer,
    MaskedTargetNormalizer,
)
from ...workspace import MLWorkspace
from .config import PolyGenDataConfig, PolyGenModelConfig
from .representation import encode_polymer_generation_sequence

torch = require_torch()


def _load_samples(
    config: PolyGenDataConfig,
    model_config: PolyGenModelConfig,
    *,
    workspace: MLWorkspace,
    samples: Sequence[MaterialSample] | None,
) -> list[MaterialSample]:
    if samples is not None:
        loaded = list(samples)
    else:
        kwargs = dict(config.dataset_kwargs)
        kwargs.setdefault("root", workspace.dataset_dir(config.dataset))
        source = create_dataset(config.dataset, **kwargs)
        iterator = source.iter_samples()
        loaded = list(islice(iterator, config.limit)) if config.limit else list(iterator)
    usable = []
    for sample in loaded:
        try:
            psmiles = extract_psmiles(sample)
            sequence = encode_polymer_generation_sequence(
                psmiles,
                model_config.representation,
            )
            tokens = tokenize_psmiles(sequence)
        except (TypeError, ValueError):
            continue
        if 2 < len(tokens) + 2:
            usable.append(sample)
    if not usable:
        raise ValueError("no usable PSMILES training samples")
    return usable


def _target_arrays(
    samples: Sequence[MaterialSample],
    model_config: PolyGenModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(samples), len(model_config.property_specs)), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for row, sample in enumerate(samples):
        for column, spec in enumerate(model_config.property_specs):
            value = sample.labels.get(spec.name)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values[row, column] = number
                mask[row, column] = True
    return values, mask


class PolymerFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[MaterialSample],
        *,
        tokenizer: PSMILESTokenizer,
        property_normalizer: MaskedTargetNormalizer,
        process_normalizer: MaskedFeatureNormalizer,
        max_length: int,
        representation: str,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.property_normalizer = property_normalizer
        self.process_normalizer = process_normalizer
        self.max_length = int(max_length)
        self.representation = str(representation)
        self.sequences = [
            encode_polymer_generation_sequence(
                extract_psmiles(sample),
                self.representation,
            )
            for sample in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        psmiles = extract_psmiles(sample)
        sequence = self.sequences[index]
        token_ids, attention_mask = self.tokenizer.encode(
            sequence,
            max_length=self.max_length,
        )
        properties, property_mask = self.property_normalizer.encode_row(sample.labels)
        process, process_mask = self.process_normalizer.encode_row(sample.conditions)
        return {
            "id": sample.id,
            "psmiles": psmiles,
            "generation_sequence": sequence,
            "token_ids": torch.as_tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attention_mask, dtype=torch.bool),
            "length": torch.tensor(int(attention_mask.sum()), dtype=torch.long),
            "properties": torch.as_tensor(properties),
            "property_mask": torch.as_tensor(property_mask, dtype=torch.bool),
            "process_conditions": torch.as_tensor(process),
            "process_condition_mask": torch.as_tensor(process_mask, dtype=torch.bool),
        }


def polymer_flow_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [sample["id"] for sample in batch],
        "psmiles": [sample["psmiles"] for sample in batch],
        "generation_sequence": [sample["generation_sequence"] for sample in batch],
        "token_ids": torch.stack([sample["token_ids"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
        "length": torch.stack([sample["length"] for sample in batch]),
        "properties": torch.stack([sample["properties"] for sample in batch]),
        "property_mask": torch.stack([sample["property_mask"] for sample in batch]),
        "process_conditions": torch.stack([sample["process_conditions"] for sample in batch]),
        "process_condition_mask": torch.stack(
            [sample["process_condition_mask"] for sample in batch]
        ),
    }


@dataclass(slots=True)
class PolyGenDataModule:
    train: Any
    valid: Any
    test: Any
    tokenizer: PSMILESTokenizer
    property_normalizer: MaskedTargetNormalizer
    process_normalizer: MaskedFeatureNormalizer
    split_sizes: dict[str, int]

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test


def prepare_poly_gen_data(
    data_config: PolyGenDataConfig,
    model_config: PolyGenModelConfig,
    *,
    workspace: MLWorkspace,
    samples: Sequence[MaterialSample] | None = None,
) -> PolyGenDataModule:
    loaded = [
        sample
        for sample in _load_samples(
            data_config,
            model_config,
            workspace=workspace,
            samples=samples,
        )
        if len(
            tokenize_psmiles(
                encode_polymer_generation_sequence(
                    extract_psmiles(sample),
                    model_config.representation,
                )
            )
        )
        + 2
        <= model_config.max_length
    ]
    if not loaded:
        raise ValueError("all PSMILES strings exceed max_length")
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
        (
            encode_polymer_generation_sequence(
                extract_psmiles(sample),
                model_config.representation,
            )
            for sample in train_samples
        ),
        min_frequency=data_config.min_token_frequency,
    )
    model_config.vocab_size = tokenizer.vocab_size
    values, mask = _target_arrays(train_samples, model_config)
    property_normalizer = MaskedTargetNormalizer(model_config.property_specs).fit(
        values,
        mask,
    )
    process_normalizer = MaskedFeatureNormalizer(model_config.process_condition_names).fit(
        [sample.conditions for sample in train_samples]
    )

    def loader(subset: Sequence[MaterialSample], *, shuffle: bool):
        dataset = PolymerFlowDataset(
            subset,
            tokenizer=tokenizer,
            property_normalizer=property_normalizer,
            process_normalizer=process_normalizer,
            max_length=model_config.max_length,
            representation=model_config.representation,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle and bool(subset),
            num_workers=data_config.num_workers,
            collate_fn=polymer_flow_collate,
            pin_memory=True,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return PolyGenDataModule(
        train=loader(train_samples, shuffle=True),
        valid=loader(valid_samples, shuffle=False),
        test=loader(test_samples, shuffle=False),
        tokenizer=tokenizer,
        property_normalizer=property_normalizer,
        process_normalizer=process_normalizer,
        split_sizes={
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples),
        },
    )


__all__ = [
    "PolyGenDataModule",
    "PolymerFlowDataset",
    "polymer_flow_collate",
    "prepare_poly_gen_data",
]
