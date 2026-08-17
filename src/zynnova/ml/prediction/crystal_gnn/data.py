from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np

from ....data import MaterialSample, create_dataset
from ....structure.crystal import stru2graph
from ...common import require_torch
from ...workspace import MLWorkspace
from .config import CrystalGNNDataConfig, CrystalGNNModelConfig


torch = require_torch()


class CrystalGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[MaterialSample],
        *,
        target: str,
        cutoff_A: float,
        max_neighbors: int | None,
    ) -> None:
        self.samples = list(samples)
        self.target = target
        self.cutoff_A = float(cutoff_A)
        self.max_neighbors = max_neighbors

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if sample.structure is None:
            raise ValueError(f"sample {sample.id!r} has no structure")
        graph = stru2graph(
            sample.structure,
            cutoff=self.cutoff_A,
            max_neighbors=self.max_neighbors,
            neighbor_mode="cutoff",
            directed=True,
            as_pyg=False,
        )
        return {
            "id": sample.id,
            "z": torch.as_tensor(graph.atomic_numbers, dtype=torch.long),
            "edge_index": torch.as_tensor(graph.edge_index, dtype=torch.long),
            "edge_distance": torch.as_tensor(graph.edge_dist, dtype=torch.get_default_dtype()),
            "cell": torch.as_tensor(graph.cell, dtype=torch.get_default_dtype()),
            "target": torch.as_tensor(float(sample.labels[self.target])),
            "natoms": torch.tensor(graph.num_nodes, dtype=torch.long),
        }


def crystal_graph_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    z_parts = []
    edge_parts = []
    distance_parts = []
    batch_index_parts = []
    cells = []
    targets = []
    natoms = []
    offset = 0
    for graph_index, sample in enumerate(batch):
        count = int(sample["z"].shape[0])
        z_parts.append(sample["z"])
        edge_parts.append(sample["edge_index"] + offset)
        distance_parts.append(sample["edge_distance"])
        batch_index_parts.append(torch.full((count,), graph_index, dtype=torch.long))
        cells.append(sample["cell"])
        targets.append(sample["target"])
        natoms.append(sample["natoms"])
        offset += count
    return {
        "id": [sample["id"] for sample in batch],
        "z": torch.cat(z_parts),
        "edge_index": torch.cat(edge_parts, dim=1),
        "edge_distance": torch.cat(distance_parts),
        "batch": torch.cat(batch_index_parts),
        "cell": torch.stack(cells),
        "target": torch.stack(targets).reshape(-1),
        "natoms": torch.stack(natoms).reshape(-1),
    }


@dataclass(slots=True)
class CrystalGNNDataModule:
    train: Any
    valid: Any
    test: Any

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test


def prepare_matbench_data(
    data_config: CrystalGNNDataConfig,
    model_config: CrystalGNNModelConfig,
    *,
    workspace: MLWorkspace,
) -> CrystalGNNDataModule:
    if data_config.dataset != "matbench":
        raise ValueError("the bundled crystal GNN currently targets Matbench")
    source = create_dataset(
        "matbench",
        root=workspace.dataset_dir("matbench"),
        task=data_config.task,
        target_columns=(data_config.target,),
    )
    iterator = source.iter_samples()
    samples = list(islice(iterator, data_config.limit)) if data_config.limit else list(iterator)
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
        dataset = CrystalGraphDataset(
            subset,
            target=data_config.target,
            cutoff_A=model_config.cutoff_A,
            max_neighbors=model_config.max_neighbors,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=shuffle,
            num_workers=data_config.num_workers,
            collate_fn=crystal_graph_collate,
            pin_memory=True,
            generator=torch.Generator().manual_seed(data_config.seed),
        )

    return CrystalGNNDataModule(
        train=loader(subsets[0], shuffle=True),
        valid=loader(subsets[1], shuffle=False),
        test=loader(subsets[2], shuffle=False),
    )


def fit_target_normalization(loader: Any) -> tuple[float, float]:
    values = [batch["target"].double() for batch in loader]
    if not values:
        raise ValueError("cannot fit target normalization from an empty loader")
    target = torch.cat(values)
    return float(target.mean()), max(float(target.std(unbiased=False)), 1.0e-8)


__all__ = [
    "CrystalGNNDataModule",
    "CrystalGraphDataset",
    "crystal_graph_collate",
    "fit_target_normalization",
    "prepare_matbench_data",
]
