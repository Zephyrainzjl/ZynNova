from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CrystalGNNModelConfig:
    max_atomic_number: int = 100
    hidden_dim: int = 128
    num_layers: int = 4
    num_rbf: int = 48
    cutoff_A: float = 5.0
    max_neighbors: int | None = 64
    target_mean: float = 0.0
    target_std: float = 1.0


@dataclass(slots=True)
class CrystalGNNDataConfig:
    dataset: str = "matbench"
    task: str = "matbench_mp_e_form"
    target: str = "e_form"
    limit: int | None = 20000
    batch_size: int = 32
    num_workers: int = 0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


@dataclass(slots=True)
class CrystalGNNTrainConfig:
    epochs: int = 200
    learning_rate: float = 1.0e-3
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
class CrystalGNNConfig:
    model: CrystalGNNModelConfig = field(default_factory=CrystalGNNModelConfig)
    data: CrystalGNNDataConfig = field(default_factory=CrystalGNNDataConfig)
    train: CrystalGNNTrainConfig = field(default_factory=CrystalGNNTrainConfig)


__all__ = [
    "CrystalGNNConfig",
    "CrystalGNNDataConfig",
    "CrystalGNNModelConfig",
    "CrystalGNNTrainConfig",
]
