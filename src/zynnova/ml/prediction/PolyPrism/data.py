from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ....data import MaterialSample
from ...common import require_torch
from ...polymer_utils import PSMILESTokenizer, extract_psmiles, split_by_family
from ...workspace import MLWorkspace
from ..PolyPrediction.data import (
    PolymerPropertyDataset,
    load_polymer_samples,
    polymer_property_collate,
    target_arrays,
)
from ..PolyPrediction.normalizer import MaskedFeatureNormalizer, MaskedTargetNormalizer
from .config import PolyPrismDataConfig, PolyPrismModelConfig

torch = require_torch()


def infer_fidelity_index(metadata: dict[str, Any], fidelity_names: Sequence[str]) -> int:
    raw = str(
        metadata.get("fidelity")
        or metadata.get("data_fidelity")
        or metadata.get("source_type")
        or "unknown"
    ).strip().lower()
    aliases = {
        "dft": "simulation",
        "md": "simulation",
        "computed": "simulation",
        "paper": "literature",
        "reported": "literature",
        "measured": "experiment",
        "experimental": "experiment",
    }
    normalized = [name.lower() for name in fidelity_names]
    resolved = aliases.get(raw, raw)
    return normalized.index(resolved) if resolved in normalized else 0


class PolyPrismDataset(PolymerPropertyDataset):
    def __init__(self, *args, fidelity_names: Sequence[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fidelity_names = tuple(fidelity_names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["fidelity_index"] = torch.tensor(
            infer_fidelity_index(self.samples[index].metadata, self.fidelity_names),
            dtype=torch.long,
        )
        return item


def poly_prism_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = polymer_property_collate(batch)
    result["fidelity_index"] = torch.stack([item["fidelity_index"] for item in batch])
    return result


@dataclass(slots=True)
class PolyPrismDataModule:
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


def prepare_poly_prism_data(
    data_config: PolyPrismDataConfig,
    model_config: PolyPrismModelConfig,
    *,
    workspace: MLWorkspace,
    samples: Sequence[MaterialSample] | None = None,
) -> PolyPrismDataModule:
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
        dataset = PolyPrismDataset(
            subset,
            tokenizer=tokenizer,
            target_normalizer=target_normalizer,
            condition_normalizer=condition_normalizer,
            max_length=model_config.max_length,
            fidelity_names=model_config.fidelity_names,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle and bool(subset),
            num_workers=data_config.num_workers,
            collate_fn=poly_prism_collate,
            pin_memory=True,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return PolyPrismDataModule(
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
    "PolyPrismDataModule",
    "PolyPrismDataset",
    "infer_fidelity_index",
    "poly_prism_collate",
    "prepare_poly_prism_data",
]
