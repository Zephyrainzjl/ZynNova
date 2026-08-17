from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...prediction.PolyPrediction.config import (
    ENERGY_STORAGE_CONDITIONS,
    ENERGY_STORAGE_PROPERTIES,
    PropertySpec,
)
from .representation import GenerationRepresentation


@dataclass(slots=True)
class PolyGenModelConfig:
    property_specs: tuple[PropertySpec, ...] = ENERGY_STORAGE_PROPERTIES
    process_condition_names: tuple[str, ...] = ENERGY_STORAGE_CONDITIONS
    representation: GenerationRepresentation = "polymer_selfies"
    repair_missing_ports: bool = True
    vocab_size: int = 0
    max_length: int = 192
    hidden_dim: int = 384
    num_layers: int = 10
    attention_heads: int = 12
    feedforward_multiplier: int = 4
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.representation not in {"psmiles", "polymer_selfies"}:
            raise ValueError("representation must be either 'psmiles' or 'polymer_selfies'")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.vocab_size and self.vocab_size < 5:
            raise ValueError("vocab_size must contain the five special tokens")
        if len({spec.name for spec in self.property_specs}) != len(self.property_specs):
            raise ValueError("property names must be unique")


@dataclass(slots=True)
class PolyGenDataConfig:
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
class PolyGenTrainConfig:
    epochs: int = 400
    learning_rate: float = 1.5e-4
    weight_decay: float = 1.0e-4
    condition_dropout: float = 0.15
    property_loss_weight: float = 0.20
    length_loss_weight: float = 0.08
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
class PolyGenSamplingConfig:
    num_candidates: int = 32
    oversample_factor: int = 8
    max_sampling_rounds: int = 4
    refinement_steps: int = 48
    guidance_scale: float = 1.8
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
        if self.max_sampling_rounds < 1:
            raise ValueError("max_sampling_rounds must be positive")
        if self.refinement_steps < 1:
            raise ValueError("refinement_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale cannot be negative")
        if self.condition_z_clip is not None and self.condition_z_clip <= 0:
            raise ValueError("condition_z_clip must be positive when supplied")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.minimum_length < 3:
            raise ValueError("minimum_length must leave room for BOS, content and EOS")
        if self.maximum_length is not None and self.maximum_length < self.minimum_length:
            raise ValueError("maximum_length cannot be smaller than minimum_length")


@dataclass(slots=True)
class PolyGenConfig:
    model: PolyGenModelConfig = field(default_factory=PolyGenModelConfig)
    data: PolyGenDataConfig = field(default_factory=PolyGenDataConfig)
    train: PolyGenTrainConfig = field(default_factory=PolyGenTrainConfig)
    sampling: PolyGenSamplingConfig = field(default_factory=PolyGenSamplingConfig)


def poly_gen_model_config_from_dict(payload: dict[str, Any]) -> PolyGenModelConfig:
    values = dict(payload)
    # Checkpoints written before Polymer-SELFIES used raw PSMILES. Keep those
    # checkpoints loadable while making the robust representation the new default.
    values.setdefault("representation", "psmiles")
    values.setdefault("repair_missing_ports", False)
    values["property_specs"] = tuple(
        spec if isinstance(spec, PropertySpec) else PropertySpec(**spec)
        for spec in values["property_specs"]
    )
    values["process_condition_names"] = tuple(values["process_condition_names"])
    return PolyGenModelConfig(**values)


__all__ = [
    "PolyGenConfig",
    "PolyGenDataConfig",
    "PolyGenModelConfig",
    "PolyGenSamplingConfig",
    "PolyGenTrainConfig",
    "poly_gen_model_config_from_dict",
]
