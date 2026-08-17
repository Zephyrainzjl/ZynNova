from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...prediction.PolyPrediction.config import (
    ENERGY_STORAGE_CONDITIONS,
    ENERGY_STORAGE_PROPERTIES,
    PropertySpec,
)
from ..PolyGen.representation import GenerationRepresentation


@dataclass(slots=True)
class PolyLoomModelConfig:
    property_specs: tuple[PropertySpec, ...] = ENERGY_STORAGE_PROPERTIES
    process_condition_names: tuple[str, ...] = ENERGY_STORAGE_CONDITIONS
    representation: GenerationRepresentation = "polymer_selfies"
    repair_missing_ports: bool = True
    vocab_size: int = 0
    max_length: int = 192
    hidden_dim: int = 384
    num_layers: int = 12
    attention_heads: int = 12
    feedforward_multiplier: int = 4
    dropout: float = 0.10
    self_conditioning: bool = True
    num_experts: int = 4

    def __post_init__(self) -> None:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.vocab_size and self.vocab_size < 5:
            raise ValueError("vocab_size must contain the five special tokens")
        if self.representation not in {"psmiles", "polymer_selfies"}:
            raise ValueError("unsupported representation")


@dataclass(slots=True)
class PolyLoomTrainConfig:
    epochs: int = 450
    learning_rate: float = 1.2e-4
    weight_decay: float = 1.0e-4
    condition_dropout: float = 0.18
    self_condition_probability: float = 0.50
    property_loss_weight: float = 0.25
    length_loss_weight: float = 0.08
    endpoint_loss_weight: float = 0.06
    expert_balance_weight: float = 0.01
    gradient_clip_norm: float | None = 3.0
    patience: int = 45
    min_delta: float = 1.0e-5
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None


@dataclass(slots=True)
class PolyLoomDataConfig:
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
class PolyLoomSamplingConfig:
    num_candidates: int = 32
    oversample_factor: int = 8
    max_sampling_rounds: int = 4
    refinement_steps: int = 64
    guidance_scale: float = 2.0
    condition_z_clip: float | None = 3.0
    temperature: float = 0.85
    top_p: float = 0.95
    minimum_length: int = 8
    maximum_length: int | None = None
    require_two_ports: bool = True
    minimum_configurational_entropy_R: float | None = None
    diversity_weight: float = 0.18
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_candidates < 1 or self.oversample_factor < 1:
            raise ValueError("candidate count and oversample factor must be positive")
        if self.max_sampling_rounds < 1 or self.refinement_steps < 1:
            raise ValueError("sampling rounds and refinement steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale cannot be negative")
        if self.condition_z_clip is not None and self.condition_z_clip <= 0:
            raise ValueError("condition_z_clip must be positive")
        if self.temperature <= 0 or not 0.0 < self.top_p <= 1.0:
            raise ValueError("invalid temperature or top_p")
        if self.minimum_length < 3:
            raise ValueError("minimum_length must leave room for BOS/content/EOS")
        if self.maximum_length is not None and self.maximum_length < self.minimum_length:
            raise ValueError("maximum_length cannot be smaller than minimum_length")


@dataclass(slots=True)
class PolyLoomConfig:
    model: PolyLoomModelConfig = field(default_factory=PolyLoomModelConfig)
    data: PolyLoomDataConfig = field(default_factory=PolyLoomDataConfig)
    train: PolyLoomTrainConfig = field(default_factory=PolyLoomTrainConfig)
    sampling: PolyLoomSamplingConfig = field(default_factory=PolyLoomSamplingConfig)


def poly_loom_model_config_from_dict(payload: dict[str, Any]) -> PolyLoomModelConfig:
    values = dict(payload)
    values.setdefault("representation", "psmiles")
    values.setdefault("repair_missing_ports", False)
    values["property_specs"] = tuple(
        spec if isinstance(spec, PropertySpec) else PropertySpec(**spec)
        for spec in values["property_specs"]
    )
    values["process_condition_names"] = tuple(values["process_condition_names"])
    return PolyLoomModelConfig(**values)


__all__ = [
    "PolyLoomConfig",
    "PolyLoomDataConfig",
    "PolyLoomModelConfig",
    "PolyLoomSamplingConfig",
    "PolyLoomTrainConfig",
    "poly_loom_model_config_from_dict",
]
