from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....data import MaterialSample, create_dataset
from ...common import require_torch
from ...workspace import MLWorkspace
from .config import QM9GeneratorDataConfig, QM9GeneratorModelConfig
from .normalizer import QM9PropertyNormalizer


torch = require_torch()


def center_coordinates(positions, mask):
    weights = mask.to(positions.dtype)[..., None]
    center = (positions * weights).sum(dim=-2, keepdim=True) / weights.sum(
        dim=-2, keepdim=True
    ).clamp_min(1.0)
    return (positions - center) * weights


class QM9PropertyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[MaterialSample],
        *,
        max_atoms: int,
        normalizer: QM9PropertyNormalizer,
    ) -> None:
        self.samples = list(samples)
        self.max_atoms = int(max_atoms)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        structure = sample.structure
        if structure is None:
            raise ValueError(f"QM9 sample {sample.id!r} has no structure")
        count = int(structure.num_atoms)
        if count > self.max_atoms:
            raise ValueError(
                f"QM9 sample {sample.id!r} has {count} atoms, "
                f"max_atoms={self.max_atoms}"
            )
        z = torch.zeros(self.max_atoms, dtype=torch.long)
        positions = torch.zeros((self.max_atoms, 3), dtype=torch.get_default_dtype())
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        z[:count] = torch.as_tensor(structure.atomic_numbers, dtype=torch.long)
        positions[:count] = torch.as_tensor(
            structure.positions,
            dtype=torch.get_default_dtype(),
        )
        mask[:count] = True
        positions = center_coordinates(positions[None, ...], mask[None, ...])[0]

        raw_properties = np.zeros(len(self.normalizer.names), dtype=np.float64)
        property_mask = np.zeros(len(self.normalizer.names), dtype=bool)
        for property_index, name in enumerate(self.normalizer.names):
            value = sample.labels.get(name)
            if value is None:
                continue
            scalar = float(np.asarray(value).reshape(-1)[0])
            if np.isfinite(scalar):
                raw_properties[property_index] = scalar
                property_mask[property_index] = True
        normalized = np.zeros_like(raw_properties)
        normalized[property_mask] = (
            raw_properties[property_mask] - self.normalizer.mean[property_mask]
        ) / self.normalizer.std[property_mask]
        return {
            "id": sample.id,
            "z": z,
            "positions": positions,
            "mask": mask,
            "natoms": torch.tensor(count, dtype=torch.long),
            "properties": torch.as_tensor(normalized, dtype=torch.get_default_dtype()),
            "property_mask": torch.as_tensor(property_mask, dtype=torch.bool),
            "raw_properties": torch.as_tensor(
                raw_properties,
                dtype=torch.get_default_dtype(),
            ),
        }


@dataclass(slots=True)
class QM9GeneratorDataModule:
    train: Any
    valid: Any
    test: Any
    normalizer: QM9PropertyNormalizer
    split_sizes: dict[str, int]

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test


def _usable_sample(
    sample: MaterialSample,
    *,
    property_names: tuple[str, ...],
    max_atoms: int,
    require_all: bool,
) -> bool:
    if sample.structure is None or int(sample.structure.num_atoms) > max_atoms:
        return False
    available = []
    for name in property_names:
        value = sample.labels.get(name)
        try:
            available.append(value is not None and np.isfinite(float(value)))
        except (TypeError, ValueError):
            available.append(False)
    return all(available) if require_all else any(available)


def prepare_qm9_generator_data(
    data_config: QM9GeneratorDataConfig,
    model_config: QM9GeneratorModelConfig,
    *,
    workspace: MLWorkspace,
) -> QM9GeneratorDataModule:
    if data_config.dataset != "qm9":
        raise ValueError("qm9_generator currently targets the QM9 dataset")
    source = create_dataset(
        "qm9",
        root=workspace.dataset_dir("qm9"),
        targets=model_config.property_names,
        limit=data_config.limit,
        local_file=data_config.local_file,
        local_dir=data_config.local_dir,
    )
    samples = [
        sample
        for sample in source.iter_samples()
        if _usable_sample(
            sample,
            property_names=model_config.property_names,
            max_atoms=model_config.max_atoms,
            require_all=data_config.require_all_properties,
        )
    ]
    if len(samples) < 3:
        raise ValueError(
            "qm9_generator needs at least three usable samples after filtering"
        )
    generator = torch.Generator().manual_seed(data_config.seed)
    order = torch.randperm(len(samples), generator=generator).tolist()
    samples = [samples[index] for index in order]

    train_count = max(1, int(len(samples) * data_config.train_ratio))
    valid_count = max(1, int(len(samples) * data_config.valid_ratio))
    if train_count + valid_count >= len(samples):
        valid_count = 1
        train_count = len(samples) - 2
    train_samples = samples[:train_count]
    valid_samples = samples[train_count : train_count + valid_count]
    test_samples = samples[train_count + valid_count :]
    normalizer = QM9PropertyNormalizer.fit(
        train_samples,
        model_config.property_names,
    )

    def loader(subset: Sequence[MaterialSample], *, shuffle: bool):
        return torch.utils.data.DataLoader(
            QM9PropertyDataset(
                subset,
                max_atoms=model_config.max_atoms,
                normalizer=normalizer,
            ),
            batch_size=data_config.batch_size,
            shuffle=shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.num_workers > 0,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return QM9GeneratorDataModule(
        train=loader(train_samples, shuffle=True),
        valid=loader(valid_samples, shuffle=False),
        test=loader(test_samples, shuffle=False),
        normalizer=normalizer,
        split_sizes={
            "train": len(train_samples),
            "valid": len(valid_samples),
            "test": len(test_samples),
        },
    )


__all__ = [
    "QM9GeneratorDataModule",
    "QM9PropertyDataset",
    "center_coordinates",
    "prepare_qm9_generator_data",
]
