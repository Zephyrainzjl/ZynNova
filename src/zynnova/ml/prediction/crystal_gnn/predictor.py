from __future__ import annotations

from pathlib import Path
from typing import Any

from ....structure.crystal import stru2graph
from ...common import load_checkpoint, resolve_device
from .config import CrystalGNNModelConfig
from .data import crystal_graph_collate
from .model import CrystalGNN


def load_crystal_gnn(checkpoint: str | Path, *, device: str = "cpu") -> CrystalGNN:
    payload = load_checkpoint(checkpoint, map_location=device)
    model = CrystalGNN(CrystalGNNModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(resolve_device(device)).eval()
    return model


def predict_crystal_property(
    model: CrystalGNN,
    structure: Any,
    *,
    device: str = "auto",
) -> float:
    import torch

    graph = stru2graph(
        structure,
        cutoff=model.config.cutoff_A,
        max_neighbors=model.config.max_neighbors,
        neighbor_mode="cutoff",
        directed=True,
        as_pyg=False,
    )
    sample = {
        "id": "prediction",
        "z": torch.as_tensor(graph.atomic_numbers, dtype=torch.long),
        "edge_index": torch.as_tensor(graph.edge_index, dtype=torch.long),
        "edge_distance": torch.as_tensor(graph.edge_dist, dtype=torch.get_default_dtype()),
        "cell": torch.as_tensor(graph.cell, dtype=torch.get_default_dtype()),
        "target": torch.tensor(0.0),
        "natoms": torch.tensor(graph.num_nodes),
    }
    batch = crystal_graph_collate([sample])
    resolved = resolve_device(device)
    batch = {
        key: value.to(resolved) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    model = model.to(resolved).eval()
    with torch.no_grad():
        return float(model(batch)[0].cpu())


__all__ = ["load_crystal_gnn", "predict_crystal_property"]
