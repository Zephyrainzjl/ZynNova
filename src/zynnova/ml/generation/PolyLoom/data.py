from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ....data import MaterialSample
from ...common import require_torch
from ...polymer_utils import PSMILESTokenizer
from ...prediction.PolyPrediction.normalizer import (
    MaskedFeatureNormalizer,
    MaskedTargetNormalizer,
)
from ...workspace import MLWorkspace
from ..PolyGen.data import (
    PolymerFlowDataset,
    polymer_flow_collate,
    prepare_poly_gen_data,
)
from .config import PolyLoomDataConfig, PolyLoomModelConfig

torch = require_torch()


class PolyLoomDataset(PolymerFlowDataset):
    """PolyLoom dataset using the standard ZynNova polymer-flow batch contract."""


@dataclass(slots=True)
class PolyLoomDataModule:
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


def prepare_poly_loom_data(
    data_config: PolyLoomDataConfig,
    model_config: PolyLoomModelConfig,
    *,
    workspace: MLWorkspace,
    samples: Sequence[MaterialSample] | None = None,
) -> PolyLoomDataModule:
    prepared = prepare_poly_gen_data(
        data_config,
        model_config,
        workspace=workspace,
        samples=samples,
    )

    def replace(loader, *, shuffle: bool):
        base = loader.dataset
        dataset = PolyLoomDataset(
            base.samples,
            tokenizer=prepared.tokenizer,
            property_normalizer=prepared.property_normalizer,
            process_normalizer=prepared.process_normalizer,
            max_length=model_config.max_length,
            representation=model_config.representation,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle and bool(dataset),
            num_workers=data_config.num_workers,
            collate_fn=polymer_flow_collate,
            pin_memory=True,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return PolyLoomDataModule(
        train=replace(prepared.train, shuffle=True),
        valid=replace(prepared.valid, shuffle=False),
        test=replace(prepared.test, shuffle=False),
        tokenizer=prepared.tokenizer,
        property_normalizer=prepared.property_normalizer,
        process_normalizer=prepared.process_normalizer,
        split_sizes=prepared.split_sizes,
    )


__all__ = [
    "PolyLoomDataModule",
    "PolyLoomDataset",
    "polymer_flow_collate",
    "prepare_poly_loom_data",
]
