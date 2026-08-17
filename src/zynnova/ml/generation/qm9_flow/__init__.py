"""QM9 composition-conditioned three-dimensional coordinate flow matching."""

from ...registry import MODELS
from .config import (
    QM9FlowConfig,
    QM9FlowDataConfig,
    QM9FlowModelConfig,
    QM9FlowTrainConfig,
)
from .data import QM9CoordinateDataset, center_coordinates, prepare_qm9_flow_data
from .model import QM9EquivariantFlow
from .sampler import load_qm9_flow, sample_qm9_coordinates, save_generated_structures
from .trainer import train_qm9_flow


@MODELS.register(
    "generation",
    "qm9_flow",
    description="Composition-conditioned E(3)-equivariant coordinate flow trained on QM9",
)
def create_qm9_flow(config: QM9FlowModelConfig | None = None) -> QM9EquivariantFlow:
    return QM9EquivariantFlow(config)


__all__ = [
    "QM9CoordinateDataset",
    "QM9EquivariantFlow",
    "QM9FlowConfig",
    "QM9FlowDataConfig",
    "QM9FlowModelConfig",
    "QM9FlowTrainConfig",
    "center_coordinates",
    "create_qm9_flow",
    "load_qm9_flow",
    "prepare_qm9_flow_data",
    "sample_qm9_coordinates",
    "save_generated_structures",
    "train_qm9_flow",
]
