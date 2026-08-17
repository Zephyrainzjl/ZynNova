from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TargetTransform = Literal["identity", "log", "log1p", "log10", "logit"]


@dataclass(frozen=True, slots=True)
class PropertySpec:
    name: str
    unit: str
    transform: TargetTransform = "identity"
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("property name cannot be empty")
        if self.transform == "logit":
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError(f"logit property {self.name!r} requires finite bounds")
            if self.lower_bound >= self.upper_bound:
                raise ValueError(f"invalid bounds for property {self.name!r}")


ENERGY_STORAGE_PROPERTIES = (
    PropertySpec(
        "glass_transition_temperature_K",
        "K",
        description="Basic thermal-processability and segmental-mobility descriptor.",
    ),
    PropertySpec(
        "bandgap_eV",
        "eV",
        "log",
        lower_bound=0.0,
        description="Electronic-insulation proxy coupled to leakage and breakdown.",
    ),
    PropertySpec(
        "dielectric_constant",
        "1",
        "log",
        lower_bound=0.0,
        description="Relative permittivity at the supplied temperature and frequency.",
    ),
    PropertySpec(
        "dielectric_loss_tangent",
        "1",
        "log10",
        lower_bound=0.0,
        description="Dielectric loss at the supplied temperature and frequency.",
    ),
    PropertySpec(
        "breakdown_strength_MV_m",
        "MV m^-1",
        "log",
        lower_bound=0.0,
        description="Weibull-scale or specimen-level breakdown strength.",
    ),
    PropertySpec(
        "leakage_current_density_A_m2",
        "A m^-2",
        "log10",
        lower_bound=0.0,
        description="Field-dependent leakage-current density.",
    ),
    PropertySpec(
        "maximum_polarization_C_m2",
        "C m^-2",
        "log1p",
        lower_bound=0.0,
        description="Maximum polarization Pm at the supplied electric field.",
    ),
    PropertySpec(
        "remanent_polarization_C_m2",
        "C m^-2",
        "log1p",
        lower_bound=0.0,
        description="Remanent polarization Pr controlling hysteretic loss.",
    ),
    PropertySpec(
        "recoverable_energy_density_J_cm3",
        "J cm^-3",
        "log1p",
        lower_bound=0.0,
        description="Recoverable energy density Ud from the discharge branch.",
    ),
    PropertySpec(
        "efficiency",
        "1",
        "logit",
        lower_bound=0.0,
        upper_bound=1.0,
        description="Charge-discharge efficiency Ud/(Ud+Uloss).",
    ),
    PropertySpec(
        "configurational_entropy_R",
        "R",
        "log1p",
        lower_bound=0.0,
        description="Bond configurational entropy divided by the gas constant R.",
    ),
    PropertySpec(
        "crystallinity_fraction",
        "1",
        "logit",
        lower_bound=0.0,
        upper_bound=1.0,
        description="Crystalline volume or mass fraction.",
    ),
    PropertySpec(
        "interchain_spacing_A",
        "angstrom",
        "log",
        lower_bound=0.0,
        description="XRD-derived or simulated interchain spacing.",
    ),
    PropertySpec(
        "helix_trans_energy_delta_eV",
        "eV",
        description="DFT energy contrast between 3/1-helix and all-trans conformers.",
    ),
)

ENERGY_STORAGE_CONDITIONS = (
    "temperature_K",
    "log10_frequency_Hz",
    "electric_field_MV_m",
    "proton_dose_Mrad",
    "film_thickness_um",
    "crystallinity_fraction",
    "random_field_strength_MV_m",
    "background_dielectric_constant",
)


@dataclass(slots=True)
class PolyPredictionModelConfig:
    property_specs: tuple[PropertySpec, ...] = ENERGY_STORAGE_PROPERTIES
    condition_names: tuple[str, ...] = ENERGY_STORAGE_CONDITIONS
    vocab_size: int = 0
    max_length: int = 256
    node_feature_dim: int = 7
    edge_feature_dim: int = 4
    hidden_dim: int = 256
    sequence_layers: int = 6
    graph_layers: int = 5
    attention_heads: int = 8
    feedforward_multiplier: int = 4
    dropout: float = 0.12
    min_log_variance: float = -8.0
    max_log_variance: float = 5.0

    def __post_init__(self) -> None:
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.vocab_size and self.vocab_size < 5:
            raise ValueError("vocab_size must contain the five special tokens")
        if len({spec.name for spec in self.property_specs}) != len(self.property_specs):
            raise ValueError("property names must be unique")
        if len(set(self.condition_names)) != len(self.condition_names):
            raise ValueError("condition names must be unique")


@dataclass(slots=True)
class PolyPredictionDataConfig:
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
class PolyPredictionTrainConfig:
    epochs: int = 300
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float | None = 5.0
    physics_loss_weight: float = 0.08
    entropy_consistency_weight: float = 0.20
    patience: int = 35
    min_delta: float = 1.0e-5
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None


@dataclass(slots=True)
class PolyPredictionConfig:
    model: PolyPredictionModelConfig = field(default_factory=PolyPredictionModelConfig)
    data: PolyPredictionDataConfig = field(default_factory=PolyPredictionDataConfig)
    train: PolyPredictionTrainConfig = field(default_factory=PolyPredictionTrainConfig)


def poly_prediction_model_config_from_dict(
    payload: dict[str, Any],
) -> PolyPredictionModelConfig:
    values = dict(payload)
    values["property_specs"] = tuple(
        spec if isinstance(spec, PropertySpec) else PropertySpec(**spec)
        for spec in values["property_specs"]
    )
    values["condition_names"] = tuple(values["condition_names"])
    return PolyPredictionModelConfig(**values)


__all__ = [
    "ENERGY_STORAGE_CONDITIONS",
    "ENERGY_STORAGE_PROPERTIES",
    "PolyPredictionConfig",
    "PolyPredictionDataConfig",
    "PolyPredictionModelConfig",
    "PolyPredictionTrainConfig",
    "PropertySpec",
    "TargetTransform",
    "poly_prediction_model_config_from_dict",
]
