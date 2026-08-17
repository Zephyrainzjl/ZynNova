from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


PotentialPreset = Literal["condensed", "electrostatic", "reactive"]


@dataclass(slots=True)
class PolymerPotentialConfig:
    """Polymer-specific adapter around the existing JouleWeave trainer."""

    preset: PotentialPreset = "condensed"
    dataset: str = "trajectory"
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_root: str | Path | None = None
    fine_tune_checkpoint: str | Path | None = None
    model_overrides: dict[str, Any] = field(default_factory=dict)
    data_overrides: dict[str, Any] = field(default_factory=dict)
    train_overrides: dict[str, Any] = field(default_factory=dict)
    hidden_dim: int = 192
    num_layers: int = 5
    cutoff_A: float = 6.0
    max_neighbors: int | None = 96
    batch_size: int = 4
    epochs: int = 300
    learning_rate: float = 2.0e-4
    energy_weight: float = 1.0
    force_weight: float = 50.0
    stress_weight: float = 0.1
    force_centric: bool = False
    freeze_backbone_epochs: int = 10
    device: str = "auto"
    dtype: str = "float32"
    workspace_root: str | Path | None = None
    run_name: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.preset not in {"condensed", "electrostatic", "reactive"}:
            raise ValueError(f"unsupported potential preset: {self.preset}")
        if self.hidden_dim < 8 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        if self.cutoff_A <= 0:
            raise ValueError("cutoff_A must be positive")
        if self.max_neighbors is not None and self.max_neighbors < 1:
            raise ValueError("max_neighbors must be positive or None")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if min(self.energy_weight, self.force_weight, self.stress_weight) < 0:
            raise ValueError("loss weights cannot be negative")


@dataclass(slots=True)
class MechanismDiscoveryConfig:
    bootstrap_repeats: int = 256
    confidence_level: float = 0.95
    stability_threshold: float = 0.70
    sign_consistency_threshold: float = 0.80
    environment_invariance_threshold: float = 0.67
    min_samples: int = 20
    min_feature_support: int = 12
    min_environment_samples: int = 8
    elastic_net_l1_ratio: float = 0.85
    alpha_grid: tuple[float, ...] = (
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.40,
    )
    cross_validation_folds: int = 5
    include_squared_terms: bool = True
    include_interactions: bool = True
    max_interaction_features: int = 12
    max_interactions: int = 36
    max_symbolic_terms: int = 4
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.bootstrap_repeats < 20:
            raise ValueError("bootstrap_repeats must be at least 20")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")
        for name in (
            "stability_threshold",
            "sign_consistency_threshold",
            "environment_invariance_threshold",
            "elastic_net_l1_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.min_samples < 6 or self.min_feature_support < 4:
            raise ValueError("sample thresholds are too small for stable discovery")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least 2")
        if not self.alpha_grid or min(self.alpha_grid) <= 0:
            raise ValueError("alpha_grid must contain positive values")


@dataclass(slots=True)
class ActiveLearningConfig:
    batch_size: int = 8
    uncertainty_weight: float = 1.0
    novelty_weight: float = 0.7
    information_weight: float = 0.8
    diversity_weight: float = 0.6
    cost_weight: float = 0.25
    minimum_distance: float = 0.05
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in (
            "uncertainty_weight",
            "novelty_weight",
            "information_weight",
            "diversity_weight",
            "cost_weight",
            "minimum_distance",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class MechanismConstraint:
    feature: str
    lower: float | None = None
    upper: float | None = None
    required: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("constraint lower bound cannot exceed upper bound")
        if self.weight < 0:
            raise ValueError("constraint weight cannot be negative")


@dataclass(slots=True)
class MechanismGenerationConfig:
    num_candidates: int = 32
    oversample_factor: int = 12
    mechanism_weight: float = 0.35
    applicability_weight: float = 0.25
    uncertainty_weight: float = 0.15
    novelty_weight: float = 0.10
    maximum_applicability_distance: float | None = 5.0
    constraints: tuple[MechanismConstraint, ...] = ()
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_candidates < 1 or self.oversample_factor < 1:
            raise ValueError("candidate counts must be positive")
        for name in (
            "mechanism_weight",
            "applicability_weight",
            "uncertainty_weight",
            "novelty_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if (
            self.maximum_applicability_distance is not None
            and self.maximum_applicability_distance <= 0
        ):
            raise ValueError("maximum_applicability_distance must be positive")


__all__ = [
    "ActiveLearningConfig",
    "MechanismConstraint",
    "MechanismDiscoveryConfig",
    "MechanismGenerationConfig",
    "PolymerPotentialConfig",
    "PotentialPreset",
]
