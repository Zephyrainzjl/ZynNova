from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


QM9_PROPERTY_UNITS: dict[str, str] = {
    "mu": "D",
    "alpha": "Bohr^3",
    "homo": "eV",
    "lumo": "eV",
    "gap": "eV",
    "r2": "Bohr^2",
    "zpve": "eV",
    "u0": "eV",
    "u": "eV",
    "h": "eV",
    "g": "eV",
    "cv": "cal/(mol K)",
    "u0_atom": "eV",
    "u_atom": "eV",
    "h_atom": "eV",
    "g_atom": "eV",
    "a": "GHz",
    "b": "GHz",
    "c": "GHz",
}


@dataclass(slots=True)
class QM9GeneratorModelConfig:
    """Neural architecture for property-conditioned QM9 coordinate generation."""

    property_names: tuple[str, ...] = ("gap", "mu", "alpha")
    max_atomic_number: int = 20
    max_atoms: int = 29
    hidden_dim: int = 128
    num_flow_layers: int = 5
    num_property_layers: int = 4
    num_rbf: int = 32
    cutoff_A: float = 8.0
    time_embedding_dim: int = 64
    condition_hidden_dim: int = 128
    property_head_hidden_dim: int = 128

    def __post_init__(self) -> None:
        self.property_names = tuple(str(name) for name in self.property_names)
        if not self.property_names:
            raise ValueError("property_names cannot be empty")
        unknown = sorted(set(self.property_names) - set(QM9_PROPERTY_UNITS))
        if unknown:
            raise ValueError(f"unknown QM9 properties: {unknown}")
        if len(set(self.property_names)) != len(self.property_names):
            raise ValueError("property_names must be unique")
        for name in (
            "max_atomic_number",
            "max_atoms",
            "hidden_dim",
            "num_flow_layers",
            "num_property_layers",
            "num_rbf",
            "time_embedding_dim",
            "condition_hidden_dim",
            "property_head_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cutoff_A <= 0:
            raise ValueError("cutoff_A must be positive")


@dataclass(slots=True)
class QM9GeneratorDataConfig:
    dataset: str = "qm9"
    limit: int | None = None
    batch_size: int = 64
    num_workers: int = 0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    require_all_properties: bool = True
    local_file: str | Path | None = None
    local_dir: str | Path | None = None
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive or None")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        ratios = (self.train_ratio, self.valid_ratio, self.test_ratio)
        if any(value < 0 for value in ratios):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(ratios) - 1.0) > 1.0e-8:
            raise ValueError("train/valid/test ratios must sum to one")
        if self.local_file is not None and self.local_dir is not None:
            raise ValueError("local_file and local_dir are mutually exclusive")


@dataclass(slots=True)
class QM9GeneratorTrainConfig:
    epochs: int = 300
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-6
    flow_loss_weight: float = 1.0
    property_loss_weight: float = 0.25
    condition_dropout: float = 0.1
    gradient_clip_norm: float | None = 10.0
    patience: int = 40
    min_delta: float = 1.0e-6
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.flow_loss_weight <= 0:
            raise ValueError("flow_loss_weight must be positive")
        if self.property_loss_weight < 0:
            raise ValueError("property_loss_weight cannot be negative")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


@dataclass(slots=True)
class QM9GeneratorSamplingConfig:
    num_candidates: int = 32
    steps: int = 200
    solver: str = "heun"
    noise_scale_A: float = 1.0
    guidance_scale: float = 2.0
    property_guidance_scale: float = 0.25
    property_gradient_clip: float | None = 5.0
    minimum_distance_A: float = 0.55
    collision_guidance_scale: float = 0.2
    bond_scale: float = 1.25
    device: str = "auto"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if self.noise_scale_A <= 0:
            raise ValueError("noise_scale_A must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale cannot be negative")
        if self.property_guidance_scale < 0:
            raise ValueError("property_guidance_scale cannot be negative")
        if self.property_gradient_clip is not None and self.property_gradient_clip <= 0:
            raise ValueError("property_gradient_clip must be positive or None")
        if self.minimum_distance_A <= 0:
            raise ValueError("minimum_distance_A must be positive")
        if self.collision_guidance_scale < 0:
            raise ValueError("collision_guidance_scale cannot be negative")
        if self.bond_scale <= 0:
            raise ValueError("bond_scale must be positive")


@dataclass(slots=True)
class QM9GeneratorConfig:
    model: QM9GeneratorModelConfig = field(default_factory=QM9GeneratorModelConfig)
    data: QM9GeneratorDataConfig = field(default_factory=QM9GeneratorDataConfig)
    train: QM9GeneratorTrainConfig = field(default_factory=QM9GeneratorTrainConfig)


__all__ = [
    "QM9GeneratorConfig",
    "QM9GeneratorDataConfig",
    "QM9GeneratorModelConfig",
    "QM9GeneratorSamplingConfig",
    "QM9GeneratorTrainConfig",
    "QM9_PROPERTY_UNITS",
]
