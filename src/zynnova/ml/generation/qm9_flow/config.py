from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class QM9FlowModelConfig:
    max_atomic_number: int = 20
    max_atoms: int = 29
    hidden_dim: int = 128
    num_layers: int = 5
    num_rbf: int = 32
    cutoff_A: float = 8.0
    time_embedding_dim: int = 64


@dataclass(slots=True)
class QM9FlowDataConfig:
    dataset: str = "qm9"
    limit: int | None = 20000
    batch_size: int = 64
    num_workers: int = 0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


@dataclass(slots=True)
class QM9FlowTrainConfig:
    epochs: int = 200
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-6
    gradient_clip_norm: float | None = 10.0
    patience: int = 30
    min_delta: float = 1.0e-6
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None


@dataclass(slots=True)
class QM9FlowConfig:
    model: QM9FlowModelConfig = field(default_factory=QM9FlowModelConfig)
    data: QM9FlowDataConfig = field(default_factory=QM9FlowDataConfig)
    train: QM9FlowTrainConfig = field(default_factory=QM9FlowTrainConfig)


__all__ = [
    "QM9FlowConfig",
    "QM9FlowDataConfig",
    "QM9FlowModelConfig",
    "QM9FlowTrainConfig",
]
