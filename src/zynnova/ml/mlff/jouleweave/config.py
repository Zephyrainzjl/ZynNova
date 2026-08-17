from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ModelMode = Literal["universal", "specialist"]
ReferenceFitMode = Literal["auto", "least_squares", "mean", "none"]
LossKind = Literal["huber", "mse"]
ChargeLabelScheme = Literal[
    "unspecified",
    "bader",
    "ddec3",
    "ddec6",
    "hirshfeld",
    "cm5",
    "custom",
]


@dataclass(slots=True)
class JouleWeaveModelConfig:
    """Architecture and physical-prior configuration for :class:`JouleWeave`.

    ``universal`` and ``specialist`` are training/deployment presets, not two
    incompatible model classes. Checkpoints therefore share one stable forward
    contract and can be fine-tuned between the two regimes.
    """

    mode: ModelMode = "universal"
    max_atomic_number: int = 118
    hidden_dim: int = 192
    num_layers: int = 5
    num_radial: int = 24
    num_attention_heads: int = 8
    num_experts: int = 6
    num_fidelity_heads: int = 1
    interaction_cutoff_A: float = 6.0
    max_neighbors: int | None = 96
    radial_trainable: bool = True
    dropout: float = 0.0
    residual_scale: float = 0.5

    # Optional CHGNet-style magnetic regularization. The atom-wise magnetic
    # moment head supervises the penultimate invariant representation and its
    # constrained latent is injected into the final interaction block.
    use_magmoms: bool = False
    magmom_nonnegative: bool = True
    magmom_condition_scale: float = 0.25

    # Optional redox-aware atom heads. The charge and oxidation-state latent is
    # inferred from the penultimate invariant representation, then gates the
    # final interaction block. Charge conservation is imposed on the reported
    # partition charges without making the local energy path non-local.
    use_charge_head: bool = False
    use_oxidation_states: bool = False
    redox_condition_scale: float = 0.25
    oxidation_state_min: int = -4
    oxidation_state_max: int = 8
    charge_label_scheme: ChargeLabelScheme = "unspecified"
    oxidation_label_method: str | None = None

    # Extensive energy calibration. Element references are in eV.
    atomic_reference_energies: dict[int, float] = field(default_factory=dict)
    residual_shift_eV_per_atom: float = 0.0
    residual_scale_eV: float = 1.0

    # Smooth short-range protection for out-of-distribution collisions.
    use_zbl: bool = True
    zbl_inner_A: float = 0.55
    zbl_outer_A: float = 1.80
    learnable_zbl_scale: bool = True

    # Optional learned, damped pair dispersion. It remains local and is therefore
    # compatible with domain-decomposed ML-IAP evaluation.
    use_dispersion: bool = True
    dispersion_cutoff_A: float = 8.0
    dispersion_init_eV_A6: float = 1.0e-3

    # Optional differentiable constrained charge equilibration. This is a global
    # graph solve and is intentionally disabled by the specialist preset and by
    # the distributed ML-IAP adapter.
    use_qeq: bool = True
    qeq_screening_A_inv: float = 0.20
    qeq_softening_A: float = 0.35
    qeq_min_hardness_eV: float = 0.5
    qeq_max_atoms: int = 512

    @property
    def total_cutoff_A(self) -> float:
        cutoffs = [self.interaction_cutoff_A]
        if self.use_zbl:
            cutoffs.append(self.zbl_outer_A)
        if self.use_dispersion:
            cutoffs.append(self.dispersion_cutoff_A)
        return max(cutoffs)

    def __post_init__(self) -> None:
        if self.mode not in {"universal", "specialist"}:
            raise ValueError("mode must be 'universal' or 'specialist'")
        if self.max_atomic_number < 1:
            raise ValueError("max_atomic_number must be positive")
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.use_magmoms and self.num_layers < 2:
            raise ValueError("use_magmoms=True requires num_layers >= 2")
        if (self.use_charge_head or self.use_oxidation_states) and self.num_layers < 2:
            raise ValueError(
                "charge/oxidation-state heads require num_layers >= 2"
            )
        if self.num_radial < 4:
            raise ValueError("num_radial must be at least 4")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_dim % self.num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if self.num_fidelity_heads < 1:
            raise ValueError("num_fidelity_heads must be positive")
        if self.interaction_cutoff_A <= 0:
            raise ValueError("interaction_cutoff_A must be positive")
        if self.max_neighbors is not None and self.max_neighbors < 1:
            raise ValueError("max_neighbors must be positive or None")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.magmom_condition_scale <= 0:
            raise ValueError("magmom_condition_scale must be positive")
        if self.redox_condition_scale <= 0:
            raise ValueError("redox_condition_scale must be positive")
        if self.oxidation_state_min >= self.oxidation_state_max:
            raise ValueError(
                "oxidation_state_min must be smaller than oxidation_state_max"
            )
        if self.charge_label_scheme not in {
            "unspecified",
            "bader",
            "ddec3",
            "ddec6",
            "hirshfeld",
            "cm5",
            "custom",
        }:
            raise ValueError(f"unsupported charge_label_scheme: {self.charge_label_scheme}")
        if self.residual_scale_eV <= 0:
            raise ValueError("residual_scale_eV must be positive")
        if not 0 < self.zbl_inner_A < self.zbl_outer_A:
            raise ValueError("require 0 < zbl_inner_A < zbl_outer_A")
        if self.use_dispersion and self.dispersion_cutoff_A <= 0:
            raise ValueError("dispersion_cutoff_A must be positive")
        if self.dispersion_init_eV_A6 <= 0:
            raise ValueError("dispersion_init_eV_A6 must be positive")
        if self.qeq_screening_A_inv < 0:
            raise ValueError("qeq_screening_A_inv cannot be negative")
        if self.qeq_softening_A <= 0:
            raise ValueError("qeq_softening_A must be positive")
        if self.qeq_min_hardness_eV <= 0:
            raise ValueError("qeq_min_hardness_eV must be positive")
        if self.qeq_max_atoms < 1:
            raise ValueError("qeq_max_atoms must be positive")
        cleaned: dict[int, float] = {}
        for atomic_number, energy in self.atomic_reference_energies.items():
            z = int(atomic_number)
            if z < 1 or z > self.max_atomic_number:
                raise ValueError(f"invalid reference atomic number: {z}")
            cleaned[z] = float(energy)
        self.atomic_reference_energies = cleaned

    @classmethod
    def universal(cls, **overrides: Any) -> JouleWeaveModelConfig:
        """High-capacity, multi-element preset for foundation-model training."""

        values: dict[str, Any] = {
            "mode": "universal",
            "hidden_dim": 192,
            "num_layers": 5,
            "num_radial": 24,
            "num_attention_heads": 8,
            "num_experts": 6,
            "interaction_cutoff_A": 6.0,
            "max_neighbors": 96,
            "use_magmoms": True,
            "use_charge_head": False,
            "use_oxidation_states": False,
            "use_zbl": True,
            "use_dispersion": True,
            "use_qeq": True,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def cathode(cls, **overrides: Any) -> JouleWeaveModelConfig:
        """Redox-aware high-capacity preset for battery and defect chemistry."""

        values: dict[str, Any] = {
            "mode": "universal",
            "hidden_dim": 192,
            "num_layers": 5,
            "num_radial": 24,
            "num_attention_heads": 8,
            "num_experts": 6,
            "interaction_cutoff_A": 6.0,
            "max_neighbors": 96,
            "use_magmoms": True,
            "use_charge_head": True,
            "use_oxidation_states": True,
            "use_zbl": True,
            "use_dispersion": True,
            "use_qeq": True,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def specialist(cls, **overrides: Any) -> JouleWeaveModelConfig:
        """Lower-cost preset for one chemistry, phase, or reaction family."""

        values: dict[str, Any] = {
            "mode": "specialist",
            "hidden_dim": 128,
            "num_layers": 4,
            "num_radial": 16,
            "num_attention_heads": 4,
            "num_experts": 2,
            "interaction_cutoff_A": 5.5,
            "max_neighbors": 64,
            "use_magmoms": False,
            "use_charge_head": False,
            "use_oxidation_states": False,
            "use_zbl": True,
            "use_dispersion": False,
            "use_qeq": False,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(slots=True)
class JouleWeaveDataConfig:
    """Dataset-to-potential task mapping.

    Any ZynNova :class:`~zynnova.data.DatasetSource` can be supplied directly
    to :func:`train_jouleweave`. When no source is supplied, ``dataset`` and
    ``dataset_kwargs`` are passed to the public dataset registry.
    """

    dataset: str = "rmd17"
    dataset_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "molecule": "revised aspirin",
            "limit": 1000,
            "selection": "random",
            "prefer_direct_download": True,
            "convert_to_ev": True,
        }
    )
    dataset_root: str | Path | None = None
    energy_source: str = "labels.energy"
    forces_source: str = "labels.forces"
    stress_source: str | None = None
    magmoms_source: str | None = None
    charges_source: str | None = None
    oxidation_states_source: str | None = None
    dipole_source: str | None = None
    total_charge_source: str | None = None
    spin_source: str | None = None
    fidelity_source: str | None = None
    charge_label_scheme: ChargeLabelScheme = "unspecified"
    oxidation_label_method: str | None = None
    material_types: tuple[str, ...] = ()
    batch_size: int = 8
    num_workers: int = 0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset cannot be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.charge_label_scheme not in {
            "unspecified",
            "bader",
            "ddec3",
            "ddec6",
            "hirshfeld",
            "cm5",
            "custom",
        }:
            raise ValueError(f"unsupported charge_label_scheme: {self.charge_label_scheme}")
        ratios = (self.train_ratio, self.valid_ratio, self.test_ratio)
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1.0e-8:
            raise ValueError("train/valid/test ratios must be non-negative and sum to 1")


@dataclass(slots=True)
class JouleWeaveTrainConfig:
    epochs: int = 300
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 1.0e-6
    warmup_epochs: int = 10
    weight_decay: float = 1.0e-6
    energy_weight: float = 1.0
    force_weight: float = 50.0
    stress_weight: float = 0.1
    magmom_weight: float = 0.0
    charge_weight: float = 0.0
    oxidation_state_weight: float = 0.0
    charge_qeq_consistency_weight: float = 0.0
    dipole_weight: float = 0.0
    oxidation_label_smoothing: float = 0.0
    strict_electronic_labels: bool = False
    loss: LossKind = "huber"
    huber_delta: float = 0.01
    gradient_clip_norm: float | None = 10.0
    gradient_accumulation: int = 1
    ema_decay: float = 0.999
    patience: int = 50
    min_delta: float = 1.0e-6
    reference_fit: ReferenceFitMode = "auto"
    reference_ridge: float = 1.0e-8
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    deterministic: bool = False
    workspace_root: str | Path | None = None
    run_name: str | None = None
    fine_tune_checkpoint: str | Path | None = None
    strict_fine_tune: bool = False
    freeze_backbone_epochs: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and initial rate positive")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        for name in (
            "energy_weight",
            "force_weight",
            "stress_weight",
            "magmom_weight",
            "charge_weight",
            "oxidation_state_weight",
            "charge_qeq_consistency_weight",
            "dipole_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.loss not in {"huber", "mse"}:
            raise ValueError("loss must be 'huber' or 'mse'")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0.0 <= self.oxidation_label_smoothing < 1.0:
            raise ValueError("oxidation_label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.reference_fit not in {"auto", "least_squares", "mean", "none"}:
            raise ValueError("invalid reference_fit mode")
        if self.reference_ridge < 0:
            raise ValueError("reference_ridge cannot be negative")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs cannot be negative")


@dataclass(slots=True)
class JouleWeaveConfig:
    model: JouleWeaveModelConfig = field(default_factory=JouleWeaveModelConfig.universal)
    data: JouleWeaveDataConfig = field(default_factory=JouleWeaveDataConfig)
    train: JouleWeaveTrainConfig = field(default_factory=JouleWeaveTrainConfig)


def jouleweave_model_config_from_dict(payload: dict[str, Any]) -> JouleWeaveModelConfig:
    values = dict(payload)
    references = values.get("atomic_reference_energies", {})
    values["atomic_reference_energies"] = {
        int(atomic_number): float(energy) for atomic_number, energy in dict(references).items()
    }
    return JouleWeaveModelConfig(**values)


__all__ = [
    "JouleWeaveConfig",
    "JouleWeaveDataConfig",
    "JouleWeaveModelConfig",
    "JouleWeaveTrainConfig",
    "ChargeLabelScheme",
    "LossKind",
    "ModelMode",
    "ReferenceFitMode",
    "jouleweave_model_config_from_dict",
]
