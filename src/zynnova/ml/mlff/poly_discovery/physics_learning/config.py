from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SymbolicBackendName = Literal["native", "pse", "pysr", "physo", "phye2e"]
OracleBackendName = Literal["auto", "kan", "quadratic", "none"]


@dataclass(slots=True)
class PhysicsLearningConfig:
    """Configuration for neural-symbolic polymer physics discovery."""

    enabled: bool = False
    symbolic_backends: tuple[SymbolicBackendName, ...] = ("native",)
    strict_backend_failures: bool = False
    oracle_backend: OracleBackendName = "auto"
    min_samples: int = 24
    minimum_feature_support_fraction: float = 0.80
    max_features: int = 12
    validation_fraction: float = 0.20
    bootstrap_repeats: int = 64
    random_seed: int = 42
    interaction_sample_count: int = 64
    interaction_step: float = 0.05
    interaction_relative_threshold: float = 0.10
    interaction_quantile: float = 0.70
    complexity_penalty: float = 0.015
    unit_consistency_bonus: float = 0.05
    reject_unit_inconsistent: bool = True
    environment_consistency_weight: float = 0.10
    workspace_root: str | Path | None = None

    # Native standardized-law search.
    native_max_terms: int = 6
    native_max_base_features: int = 12
    native_pair_terms: int = 48

    # PSE/PSRN GPU-parallel symbolic enumeration.
    pse_operators: tuple[str, ...] = (
        "Add",
        "Mul",
        "SemiSub",
        "SemiDiv",
        "Identity",
        "Sin",
        "Cos",
        "Exp",
        "Log",
    )
    pse_symbol_layers: int = 3
    pse_inputs: int | None = None
    pse_downsample: int = 256
    pse_top_k: int = 20
    pse_pareto_candidates: int = 12
    pse_time_limit_seconds: int = 600
    pse_use_constants: bool = True
    pse_use_dr_mask: bool = False
    pse_dr_mask_dir: str | Path | None = None
    pse_device: str = "auto"

    # RBF-KAN oracle used for PhyE2E-style Hessian decomposition.
    kan_hidden_width: int = 16
    kan_grid_size: int = 12
    kan_layers: int = 2
    kan_epochs: int = 500
    kan_learning_rate: float = 2.0e-3
    kan_weight_decay: float = 1.0e-6
    kan_sparsity_weight: float = 1.0e-5
    kan_patience: int = 60
    kan_device: str = "auto"
    kan_dtype: Literal["float32", "float64"] = "float64"
    monotonic_constraints: dict[str, int] = field(default_factory=dict)
    monotonicity_weight: float = 0.05

    # PySR evolutionary Pareto search.
    pysr_iterations: int = 120
    pysr_populations: int = 12
    pysr_population_size: int = 48
    pysr_max_size: int = 30
    pysr_pareto_candidates: int = 12
    pysr_dimensionless_constants_only: bool = True
    pysr_timeout_seconds: int | None = None

    # PhySO unit-guided reinforcement symbolic optimization.
    physo_epochs: int = 120
    physo_device: str = "cpu"
    physo_parallel: bool = False

    # Official PhyE2E adapter. The repository and checkpoint are external.
    phye2e_repository: str | Path | None = None
    phye2e_checkpoint: str | Path | None = None
    phye2e_device: str | None = None
    phye2e_max_points: int = 512
    phye2e_use_divide: bool = True
    phye2e_use_mcts: bool = True
    phye2e_use_gp: bool = True
    phye2e_oracle_epochs: int = 400

    def __post_init__(self) -> None:
        self.symbolic_backends = tuple(
            str(name).lower() for name in self.symbolic_backends
        )
        self.pse_operators = tuple(str(name) for name in self.pse_operators)
        allowed_backends = {"native", "pse", "pysr", "physo", "phye2e"}
        unsupported = set(self.symbolic_backends) - allowed_backends
        if unsupported:
            raise ValueError(f"unsupported symbolic backends: {sorted(unsupported)}")
        if not self.symbolic_backends:
            raise ValueError("at least one symbolic backend is required")
        if self.oracle_backend not in {"auto", "kan", "quadratic", "none"}:
            raise ValueError(f"unsupported oracle backend: {self.oracle_backend}")
        if self.min_samples < 12:
            raise ValueError("min_samples must be at least 12")
        if not 0.5 <= self.minimum_feature_support_fraction <= 1.0:
            raise ValueError(
                "minimum_feature_support_fraction must lie in [0.5, 1]"
            )
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if not 0.05 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in [0.05, 0.5)")
        if self.bootstrap_repeats < 20:
            raise ValueError("bootstrap_repeats must be at least 20")
        if self.interaction_sample_count < 4 or self.interaction_step <= 0:
            raise ValueError("interaction sampling parameters must be positive")
        if not 0.0 <= self.interaction_relative_threshold <= 1.0:
            raise ValueError("interaction_relative_threshold must lie in [0, 1]")
        if not 0.0 <= self.interaction_quantile <= 1.0:
            raise ValueError("interaction_quantile must lie in [0, 1]")
        if min(
            self.complexity_penalty,
            self.unit_consistency_bonus,
            self.environment_consistency_weight,
        ) < 0:
            raise ValueError("equation ranking weights cannot be negative")
        if self.native_max_terms < 1 or self.native_max_base_features < 1:
            raise ValueError("native symbolic search sizes must be positive")
        if (
            not self.pse_operators
            or self.pse_symbol_layers < 1
            or self.pse_downsample < 8
            or self.pse_top_k < 1
            or self.pse_pareto_candidates < 1
            or self.pse_time_limit_seconds < 1
        ):
            raise ValueError("PSE search settings must be positive")
        if self.pse_inputs is not None and self.pse_inputs < 1:
            raise ValueError("pse_inputs must be positive when provided")
        if (
            self.kan_hidden_width < 2
            or self.kan_grid_size < 4
            or self.kan_layers < 1
            or self.kan_epochs < 1
        ):
            raise ValueError("KAN architecture and epochs must be positive")
        if self.kan_learning_rate <= 0 or self.kan_patience < 1:
            raise ValueError("KAN optimizer settings must be positive")
        invalid_monotonicity = {
            name: direction
            for name, direction in self.monotonic_constraints.items()
            if int(direction) not in {-1, 1}
        }
        if invalid_monotonicity:
            raise ValueError(
                "monotonic constraints must use direction -1 or +1: "
                f"{invalid_monotonicity}"
            )
        if (
            self.pysr_iterations < 1
            or self.pysr_max_size < 3
            or self.pysr_pareto_candidates < 1
        ):
            raise ValueError("PySR search sizes must be positive")
        if self.physo_epochs < 1:
            raise ValueError("PhySO epochs must be positive")
        if self.phye2e_max_points < 16 or self.phye2e_oracle_epochs < 1:
            raise ValueError("PhyE2E point and epoch limits must be positive")


__all__ = [
    "OracleBackendName",
    "PhysicsLearningConfig",
    "SymbolicBackendName",
]
