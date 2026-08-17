from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....data import MaterialSample, create_dataset
from ...common import require_torch
from ...workspace import MLWorkspace
from .config import QM9FlowDataConfig, QM9FlowModelConfig


torch = require_torch()


def center_coordinates(positions, mask):
    weights = mask.to(positions.dtype)[..., None]
    center = (positions * weights).sum(dim=-2, keepdim=True) / weights.sum(
        dim=-2, keepdim=True
    ).clamp_min(1.0)
    return (positions - center) * weights


class QM9CoordinateDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[MaterialSample], *, max_atoms: int) -> None:
        self.samples = list(samples)
        self.max_atoms = int(max_atoms)

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
                f"QM9 sample {sample.id!r} has {count} atoms, max_atoms={self.max_atoms}"
            )
        z = torch.zeros(self.max_atoms, dtype=torch.long)
        positions = torch.zeros((self.max_atoms, 3), dtype=torch.get_default_dtype())
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        z[:count] = torch.as_tensor(structure.atomic_numbers, dtype=torch.long)
        positions[:count] = torch.as_tensor(structure.positions, dtype=torch.get_default_dtype())
        mask[:count] = True
        positions = center_coordinates(positions[None, ...], mask[None, ...])[0]
        return {
            "id": sample.id,
            "z": z,
            "positions": positions,
            "mask": mask,
            "natoms": torch.tensor(count, dtype=torch.long),
        }


@dataclass(slots=True)
class QM9FlowDataModule:
    train: Any
    valid: Any
    test: Any

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test


def prepare_qm9_flow_data(
    data_config: QM9FlowDataConfig,
    model_config: QM9FlowModelConfig,
    *,
    workspace: MLWorkspace,
) -> QM9FlowDataModule:
    if data_config.dataset != "qm9":
        raise ValueError("the bundled coordinate flow currently targets QM9")
    source = create_dataset(
        "qm9",
        root=workspace.dataset_dir("qm9"),
        targets=("gap",),
        limit=data_config.limit,
    )
    samples = list(source.iter_samples())
    generator = torch.Generator().manual_seed(data_config.seed)
    order = torch.randperm(len(samples), generator=generator).tolist()
    samples = [samples[index] for index in order]
    train_count = int(len(samples) * data_config.train_ratio)
    valid_count = int(len(samples) * data_config.valid_ratio)
    subsets = (
        samples[:train_count],
        samples[train_count : train_count + valid_count],
        samples[train_count + valid_count :],
    )

    def loader(subset, *, shuffle: bool):
        return torch.utils.data.DataLoader(
            QM9CoordinateDataset(subset, max_atoms=model_config.max_atoms),
            batch_size=data_config.batch_size,
            shuffle=shuffle,
            num_workers=data_config.num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return QM9FlowDataModule(
        train=loader(subsets[0], shuffle=True),
        valid=loader(subsets[1], shuffle=False),
        test=loader(subsets[2], shuffle=False),
    )


def empirical_compositions(loader: Any, *, limit: int | None = None) -> list[np.ndarray]:
    compositions: list[np.ndarray] = []
    for batch in loader:
        for z, mask in zip(batch["z"], batch["mask"], strict=True):
            compositions.append(z[mask].cpu().numpy())
            if limit is not None and len(compositions) >= limit:
                return compositions
    return compositions


__all__ = [
    "QM9CoordinateDataset",
    "QM9FlowDataModule",
    "center_coordinates",
    "empirical_compositions",
    "prepare_qm9_flow_data",
]
