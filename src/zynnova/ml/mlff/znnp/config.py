from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ZNNPModelConfig:
    max_atomic_number: int = 100
    hidden_dim: int = 128
    num_interactions: int = 4
    num_rbf: int = 64
    cutoff_A: float = 5.0
    max_neighbors: int | None = 64
    energy_shift_eV_per_atom: float = 0.0
    energy_scale_eV: float = 1.0


@dataclass(slots=True)
class ZNNPDataConfig:
    dataset: str = "rmd17"
    molecule: str = "revised aspirin"
    limit: int = 1000
    selection: str = "random"
    local_file: str | Path | None = None
    prefer_direct_download: bool = True
    batch_size: int = 8
    num_workers: int = 0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


@dataclass(slots=True)
class ZNNPTrainConfig:
    epochs: int = 300
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    energy_weight: float = 1.0
    force_weight: float = 100.0
    gradient_clip_norm: float | None = 10.0
    patience: int = 50
    min_delta: float = 1.0e-6
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None


@dataclass(slots=True)
class ZNNPConfig:
    model: ZNNPModelConfig = field(default_factory=ZNNPModelConfig)
    data: ZNNPDataConfig = field(default_factory=ZNNPDataConfig)
    train: ZNNPTrainConfig = field(default_factory=ZNNPTrainConfig)


__all__ = ["ZNNPConfig", "ZNNPDataConfig", "ZNNPModelConfig", "ZNNPTrainConfig"]
