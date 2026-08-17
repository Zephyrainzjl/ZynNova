"""Periodic crystal GNN trained on Matbench formation energy."""

from ...registry import MODELS
from .config import (
    CrystalGNNConfig,
    CrystalGNNDataConfig,
    CrystalGNNModelConfig,
    CrystalGNNTrainConfig,
)
from .data import (
    CrystalGraphDataset,
    crystal_graph_collate,
    fit_target_normalization,
    prepare_matbench_data,
)
from .model import CrystalGNN
from .predictor import load_crystal_gnn, predict_crystal_property
from .trainer import train_crystal_gnn


@MODELS.register(
    "prediction",
    "crystal_gnn",
    description="Periodic message-passing regressor for Matbench formation energy",
)
def create_crystal_gnn(config: CrystalGNNModelConfig | None = None) -> CrystalGNN:
    return CrystalGNN(config)


__all__ = [
    "CrystalGNN",
    "CrystalGNNConfig",
    "CrystalGNNDataConfig",
    "CrystalGNNModelConfig",
    "CrystalGNNTrainConfig",
    "CrystalGraphDataset",
    "create_crystal_gnn",
    "crystal_graph_collate",
    "fit_target_normalization",
    "load_crystal_gnn",
    "predict_crystal_property",
    "prepare_matbench_data",
    "train_crystal_gnn",
]
