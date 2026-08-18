"""Typed settings for characterization and descriptor-based reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CharacterizationSettings:
    descriptor_types: tuple[str, ...] = ("Correlations", "Variation")
    limit_to: int = 16
    use_multiphase: bool = True
    use_multigrid_descriptors: bool = True
    multigrid_levels: int | None = None
    periodic: bool = True
    slice_mode: str = "full"
    isotropic: bool = False
    descriptor_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    phase_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.descriptor_types:
            raise ValueError("descriptor_types cannot be empty")
        if self.limit_to < 1:
            raise ValueError("limit_to must be >= 1")
        if self.multigrid_levels is not None and self.multigrid_levels < 1:
            raise ValueError("multigrid_levels must be >= 1")
        if self.slice_mode not in {"full", "average", "sample", "sample_surface"}:
            raise ValueError("slice_mode must be full, average, sample, or sample_surface")


@dataclass(frozen=True, slots=True)
class ReconstructionSettings:
    descriptor_types: tuple[str, ...] = ("Correlations", "Variation")
    descriptor_weights: tuple[float, ...] | None = None
    optimizer_type: str = "LBFGSB"
    loss_type: str = "MSE"
    limit_to: int = 16
    use_multiphase: bool = True
    use_multigrid_descriptors: bool = True
    use_multigrid_reconstruction: bool = False
    multigrid_levels: int | None = None
    periodic: bool = True
    slice_mode: str = "full"
    isotropic: bool = False
    learning_rate: float = 0.03
    beta_1: float = 0.9
    beta_2: float = 0.999
    rho: float = 0.9
    momentum: float = 0.0
    max_iter: int = 500
    tolerance: float = 1.0e-8
    seed: int = 0
    device: str = "auto"
    dtype: str = "float64"
    initialization: str = "volume_fraction_noise"
    initial_temperature: float = 1.0e-3
    final_temperature: float | None = None
    cooldown_factor: float = 0.995
    mutation_rule: str = "phase_swap"
    acceptance_distribution: str = "metropolis"
    convergence_data_steps: int = 10
    phase_sum_multiplier: float = 0.0
    out_of_range_multiplier: float = 0.0
    descriptor_kwargs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    phase_ids: tuple[int, ...] | None = None
    postprocess: str = "argmax"
    postprocess_annealing_steps: int = 0

    def __post_init__(self) -> None:
        if not self.descriptor_types:
            raise ValueError("descriptor_types cannot be empty")
        if self.descriptor_weights is not None:
            if len(self.descriptor_weights) != len(self.descriptor_types):
                raise ValueError("descriptor_weights must match descriptor_types")
            if any(weight < 0 for weight in self.descriptor_weights):
                raise ValueError("descriptor_weights must be non-negative")
        if self.slice_mode not in {"full", "average", "sample", "sample_surface"}:
            raise ValueError("slice_mode must be full, average, sample, or sample_surface")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.cooldown_factor <= 1:
            raise ValueError("cooldown_factor must be in (0, 1]")
        if self.initialization not in {"random", "volume_fraction_noise", "provided"}:
            raise ValueError("unsupported initialization")
        if self.postprocess not in {"argmax", "none"}:
            raise ValueError("postprocess must be argmax or none")

    @property
    def weights(self) -> tuple[float, ...]:
        if self.descriptor_weights is not None:
            return tuple(float(item) for item in self.descriptor_weights)
        # Keep the established high default weight for total variation.
        return tuple(100.0 if name.lower() == "variation" else 1.0 for name in self.descriptor_types)


__all__ = ["CharacterizationSettings", "ReconstructionSettings"]
