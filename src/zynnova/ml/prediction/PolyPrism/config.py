from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..PolyPrediction.config import (
    ENERGY_STORAGE_CONDITIONS,
    ENERGY_STORAGE_PROPERTIES,
    PropertySpec,
)

UncertaintyMode = Literal["heteroscedastic", "evidential"]


@dataclass(slots=True)
class PolyPrismModelConfig:
    """Configuration for the PolyPrism polymer foundation predictor."""

    property_specs: tuple[PropertySpec, ...] = ENERGY_STORAGE_PROPERTIES
    condition_names: tuple[str, ...] = ENERGY_STORAGE_CONDITIONS
    fidelity_names: tuple[str, ...] = ("unknown", "simulation", "literature", "experiment")
    vocab_size: int = 0
    max_length: int = 256
    node_feature_dim: int = 7
    edge_feature_dim: int = 4
    hidden_dim: int = 256
    sequence_layers: int = 6
    graph_layers: int = 5
    fusion_layers: int = 3
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    num_experts: int = 4
    top_k_experts: int = 2
    dropout: float = 0.10
    uncertainty: UncertaintyMode = "evidential"
    min_log_variance: float = -9.0
    max_log_variance: float = 6.0

    def __post_init__(self) -> None:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.vocab_size and self.vocab_size < 5:
            raise ValueError("vocab_size must contain the five special tokens")
        if not 1 <= self.top_k_experts <= self.num_experts:
            raise ValueError("top_k_experts must lie in [1, num_experts]")
        if self.uncertainty not in {"heteroscedastic", "evidential"}:
            raise ValueError("unknown uncertainty mode")
        if len({item.name for item in self.property_specs}) != len(self.property_specs):
            raise ValueError("property names must be unique")


@dataclass(slots=True)
class PolyPrismDataConfig:
    dataset: str = "transpolymer"
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    batch_size: int = 64
    num_workers: int = 0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    min_token_frequency: int = 1
    seed: int = 42


@dataclass(slots=True)
class PolyPrismTrainConfig:
    epochs: int = 350
    learning_rate: float = 1.5e-4
    weight_decay: float = 1.0e-4
    physics_loss_weight: float = 0.08
    entropy_consistency_weight: float = 0.20
    expert_balance_weight: float = 0.01
    evidential_regularization: float = 1.0e-3
    gradient_clip_norm: float | None = 3.0
    patience: int = 40
    min_delta: float = 1.0e-5
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None


@dataclass(slots=True)
class PolyPrismConfig:
    model: PolyPrismModelConfig = field(default_factory=PolyPrismModelConfig)
    data: PolyPrismDataConfig = field(default_factory=PolyPrismDataConfig)
    train: PolyPrismTrainConfig = field(default_factory=PolyPrismTrainConfig)


def poly_prism_model_config_from_dict(payload: dict[str, Any]) -> PolyPrismModelConfig:
    values = dict(payload)
    values["property_specs"] = tuple(
        spec if isinstance(spec, PropertySpec) else PropertySpec(**spec)
        for spec in values["property_specs"]
    )
    values["condition_names"] = tuple(values["condition_names"])
    values["fidelity_names"] = tuple(values["fidelity_names"])
    return PolyPrismModelConfig(**values)


__all__ = [
    "PolyPrismConfig",
    "PolyPrismDataConfig",
    "PolyPrismModelConfig",
    "PolyPrismTrainConfig",
    "UncertaintyMode",
    "poly_prism_model_config_from_dict",
]
