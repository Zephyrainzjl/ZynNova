"""Validated configuration for the variational ZIVAR electro-spin core.

The version-2 production path minimizes one constrained ``q/p/Q/m`` energy
functional.  The fixed-depth polar-density and direct-head implementations are
kept only as explicit legacy alternatives; they are never constructed beside
the variational core by an implicit fallback.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

ZIVAR_VERSION = "0.2.0"
ARCHITECTURE_REVISION = "zivar-variational-electrospin-2"
NUMERICS_REVISION = "variational-scf-pme.1"
LEGACY_ARCHITECTURE_REVISIONS: tuple[str, ...] = (
    "zivar-electrospin-density-0.1.0",
)
SUPPORTED_MACE_SERIES = "0.3.16"


@dataclass(frozen=True, slots=True)
class SCFConfig:
    """Fail-closed settings for the constrained variational solve.

    Residuals are measured after projecting out all equality constraints.  A
    production evaluation always raises when these tolerances are not reached;
    returning the last iterate is intentionally not configurable.
    """

    solver: Literal["pcg"] = "pcg"
    preconditioner: Literal["onsite", "none"] = "onsite"
    atol: float = 1.0e-10
    rtol: float = 1.0e-8
    energy_atol_eV_per_atom: float = 1.0e-12
    max_iter: int = 200
    warm_start: bool = True
    negative_curvature_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.solver != "pcg":
            raise ValueError("the production variational core requires solver='pcg'")
        if self.preconditioner not in {"onsite", "none"}:
            raise ValueError("invalid SCF preconditioner")
        if min(self.atol, self.rtol, self.energy_atol_eV_per_atom) <= 0:
            raise ValueError("SCF tolerances must be positive")
        if self.max_iter < 1:
            raise ValueError("SCF max_iter must be positive")
        if self.negative_curvature_tolerance < 0:
            raise ValueError("negative-curvature tolerance must be nonnegative")


@dataclass(frozen=True, slots=True)
class ElectrostaticsConfig:
    """Boundary and error contract for direct Ewald and mesh electrostatics."""

    boundary: Literal["auto", "periodic_3d", "isolated"] = "auto"
    method: Literal["pme", "direct_ewald"] = "pme"
    error_target: float = 1.0e-6
    real_cutoff_A: float | None = None
    alpha_per_A: float | None = None
    mesh: tuple[int, int, int] | None = None
    interpolation_order: int = 4
    neutralizing_background: bool = True
    surface: Literal["tinfoil"] = "tinfoil"
    direct_reference_tolerance: float = 1.0e-11
    isolated_direct_max_atoms: int = 8_192

    def __post_init__(self) -> None:
        if self.boundary not in {"auto", "periodic_3d", "isolated"}:
            raise ValueError("invalid electrostatic boundary")
        if self.method not in {"pme", "direct_ewald"}:
            raise ValueError("invalid electrostatic method")
        if not 0 < self.error_target < 1:
            raise ValueError("electrostatic error_target must lie in (0, 1)")
        if self.real_cutoff_A is not None and self.real_cutoff_A <= 0:
            raise ValueError("real_cutoff_A must be positive")
        if self.alpha_per_A is not None and self.alpha_per_A <= 0:
            raise ValueError("alpha_per_A must be positive")
        if self.mesh is not None and (
            len(self.mesh) != 3 or any(int(value) < 4 for value in self.mesh)
        ):
            raise ValueError("mesh must contain three dimensions of at least four")
        if self.interpolation_order not in {2, 4, 6}:
            raise ValueError("interpolation_order must be one of 2, 4 or 6")
        if self.surface != "tinfoil":
            raise ValueError("version 2 currently supports only tinfoil surfaces")
        if self.direct_reference_tolerance <= 0 or self.isolated_direct_max_atoms < 1:
            raise ValueError("invalid electrostatic reference settings")


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    """Immutable local conservative backbone.

    The backbone owns local geometry features and short-range energy only.
    Electronic density, electrostatics and magnetic Hamiltonians are external
    modules and use the same contract for every registered backbone.
    """

    kind: str = "mace"
    cutoff_A: float = 5.0
    atomic_numbers: tuple[int, ...] = tuple(range(1, 95))
    atomic_energies_eV: tuple[float, ...] | None = None
    channels: int = 96
    hidden_irreps: str | None = None
    mlp_irreps: str = "32x0e"
    num_interactions: int = 2
    correlation: int | tuple[int, ...] = 3
    max_ell: int = 3
    num_bessel: int = 8
    cutoff_polynomial_order: int = 5
    radial_mlp: tuple[int, ...] = (64, 64, 64)
    average_num_neighbors: float = 24.0
    pair_repulsion: bool = True
    use_reduced_cg: bool = False
    use_edge_irreps_first: bool = True
    backend: Literal["auto", "e3nn", "cueq", "oeq", "hybrid"] = "auto"
    compile_mode: Literal[
        "none", "default", "reduce-overhead", "max-autotune"
    ] = "none"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", self.kind) is None:
            raise ValueError("kind must be a lowercase registry identifier")
        if self.cutoff_A <= 0:
            raise ValueError("cutoff_A must be positive")
        if (
            not self.atomic_numbers
            or tuple(sorted(set(self.atomic_numbers))) != self.atomic_numbers
        ):
            raise ValueError("atomic_numbers must be a sorted unique tuple")
        if any(value < 1 or value > 118 for value in self.atomic_numbers):
            raise ValueError("atomic numbers must lie in [1, 118]")
        if self.atomic_energies_eV is not None and len(
            self.atomic_energies_eV
        ) != len(self.atomic_numbers):
            raise ValueError("atomic_energies_eV must match atomic_numbers")
        if self.channels < 8:
            raise ValueError("channels must be at least 8")
        if not 1 <= self.num_interactions <= 4:
            raise ValueError("num_interactions must lie in [1, 4]")
        if not 0 <= self.max_ell <= 4:
            raise ValueError("max_ell must lie in [0, 4]")
        if len(self.correlations) != self.num_interactions or any(
            value < 1 or value > 4 for value in self.correlations
        ):
            raise ValueError("correlation requires one value in [1, 4] per interaction")
        if self.num_bessel < 2 or self.cutoff_polynomial_order < 2:
            raise ValueError("radial basis sizes must be at least two")
        if not self.radial_mlp or any(width < 1 for width in self.radial_mlp):
            raise ValueError("radial_mlp must contain positive widths")
        if self.average_num_neighbors <= 0:
            raise ValueError("average_num_neighbors must be positive")
        if self.backend not in {"auto", "e3nn", "cueq", "oeq", "hybrid"}:
            raise ValueError("invalid equivariant execution backend")
        if self.compile_mode not in {"none", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("invalid torch.compile mode")
        if not isinstance(self.options, dict) or any(
            not isinstance(key, str) for key in self.options
        ):
            raise ValueError("options must be a string-keyed dictionary")

    @property
    def resolved_hidden_irreps(self) -> str:
        return self.hidden_irreps or f"{self.channels}x0e + {self.channels}x1o"

    @property
    def correlations(self) -> tuple[int, ...]:
        if isinstance(self.correlation, int):
            return (self.correlation,) * self.num_interactions
        return self.correlation


@dataclass(frozen=True, slots=True)
class OxidationConfig:
    """Supervised formal oxidation-state model.

    Partial charges are never converted into formal labels. Enabling exact
    inference requires an explicitly named formal-label source so an untrained
    classifier cannot be exposed as a chemical result.
    """

    enabled: bool = False
    minimum_state: int = -4
    maximum_state: int = 8
    hidden: tuple[int, ...] = (128, 96, 64)
    exact_inference: bool = True
    label_source: str = "none"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_state >= self.maximum_state:
            raise ValueError("oxidation-state interval is empty")
        if self.minimum_state < -8 or self.maximum_state > 12:
            raise ValueError("oxidation-state interval is outside the supported range")
        if not self.hidden or any(width < 4 for width in self.hidden):
            raise ValueError("oxidation hidden widths must be at least four")
        if self.temperature <= 0:
            raise ValueError("oxidation scales are invalid")
        forbidden = {"", "none", "partial_charge", "rounded_charge"}
        if self.enabled and self.label_source.strip().lower() in forbidden:
            raise ValueError(
                "enabled oxidation prediction requires a named formal-label source; "
                "partial or rounded charges are forbidden"
            )


@dataclass(frozen=True, slots=True)
class SpinConfig:
    """Time-reversal-equivariant magnetic potential configuration.

    ``spin_lattice`` models an explicit time-odd axial spin ``S`` and an
    induced axial moment ``m`` inside the same scalar energy functional.
    Magnitude and collinear modes are auxiliary compatibility heads and never
    claim non-collinear spin physics.
    """

    mode: Literal[
        "spin_lattice", "collinear_density", "magnitude_auxiliary", "disabled"
    ] = "spin_lattice"
    soc: bool = True
    induced_magnetization: bool = True
    constrain_total_magnetization: bool = False
    hidden: tuple[int, ...] = (128, 128, 64)
    require_spin_input: bool = True
    exchange: bool = True
    biquadratic_exchange: bool = True
    anisotropy: bool = True
    dmi: bool = True
    neural_high_order: bool = True
    onsite_landau: bool = True
    external_field: bool = True
    minimum_moment: float = 1.0e-8
    energy_scale_eV: float = 0.10
    dmi_scale_eV: float = 0.02
    anisotropy_scale_eV: float = 0.02
    bohr_magneton_eV_per_T: float = 5.7883818060e-5
    llg_gyromagnetic_ratio: float = 1.76085963023e11
    llg_damping: float = 0.05
    inverse_susceptibility_floor_eV_per_muB2: float = 0.10

    def __post_init__(self) -> None:
        allowed = {"spin_lattice", "collinear_density", "magnitude_auxiliary", "disabled"}
        if self.mode not in allowed:
            raise ValueError("invalid magnetic mode")
        if not self.hidden or any(width < 4 for width in self.hidden):
            raise ValueError("spin hidden widths must be at least four")
        positive = (
            self.minimum_moment, self.energy_scale_eV, self.dmi_scale_eV,
            self.anisotropy_scale_eV, self.bohr_magneton_eV_per_T,
            self.llg_gyromagnetic_ratio,
            self.inverse_susceptibility_floor_eV_per_muB2,
        )
        if min(positive) <= 0 or self.llg_damping < 0:
            raise ValueError("spin scales are invalid")
        if self.mode == "spin_lattice" and not self.require_spin_input:
            raise ValueError(
                "production spin_lattice mode requires explicit spin vectors; "
                "geometry alone cannot select a time-odd state"
            )
        if not self.soc and (self.dmi or self.anisotropy):
            raise ValueError(
                "SOC-off mode requires dmi=False and anisotropy=False; "
                "spin-lattice anisotropy cannot be enabled implicitly"
            )
        if self.constrain_total_magnetization and not self.induced_magnetization:
            raise ValueError(
                "a total-magnetization constraint requires induced magnetization"
            )


@dataclass(frozen=True, slots=True)
class ElectronicConfig:
    """Variational charge, polarization, quadrupole and induced-spin settings.

    ``variational`` is the only default path and is controlled by residual and
    energy tolerances.  ``polar``, ``qeq`` and direct heads remain explicitly
    selectable legacy alternatives; none is constructed alongside the
    variational model.
    """

    method: Literal[
        "variational", "polar", "qeq", "direct", "fukui_auxiliary"
    ] = "variational"
    energy_coupling: Literal["auxiliary", "learned", "electrostatic", "full"] = "full"
    density_lmax: int = 2
    potential_lmax: int = 2
    hidden: tuple[int, ...] = (192, 192, 128)
    polarization_updates: int = 0
    radial_basis: int = 12
    fukui_floor: float = 1.0e-4
    gaussian_width_A: float = 0.75
    reciprocal_kmax: int = 5
    direct_pair_block: int = 16_384
    coulomb_constant_eV_A: float = 14.3996454784255
    hardness_floor_eV: float = 0.50
    qeq_jitter_eV: float = 1.0e-6
    qeq_max_atoms: int = 2_048
    learned_energy_scale_eV: float = 0.10
    potential_scale_eV: float = 10.0
    multipole_update_scale: float = 0.20
    constraint_tolerance: float = 1.0e-6
    boundary_mode: Literal["fixed_charge", "fixed_potential", "mixed"] = "fixed_charge"
    periodic_background: bool = True
    variational_envelope_forces: bool = True
    enforce_spin_each_update: bool = True
    include_external_field_in_updates: bool = True
    oxidation: OxidationConfig = field(default_factory=OxidationConfig)
    dipole_stiffness_floor_eV_per_eA2: float = 0.25
    quadrupole_stiffness_floor_eV_per_eA4: float = 0.10
    magnetic_stiffness_floor_eV_per_muB2: float = 0.10

    def __post_init__(self) -> None:
        if self.method not in {
            "variational", "polar", "qeq", "direct", "fukui_auxiliary"
        }:
            raise ValueError("invalid electronic method")
        if self.energy_coupling not in {"auxiliary", "learned", "electrostatic", "full"}:
            raise ValueError("invalid electronic energy coupling")
        if self.method in {"variational", "polar"} and self.energy_coupling != "full":
            raise ValueError(
                "variational and legacy polar density require energy_coupling='full'"
            )
        if self.method == "qeq" and self.energy_coupling not in {"electrostatic", "full"}:
            raise ValueError("QEq energy must include its variational electrostatic term")
        if self.method in {"direct", "fukui_auxiliary"} and self.energy_coupling not in {
            "auxiliary",
            "learned",
        }:
            raise ValueError(
                "direct/Fukui alternatives do not implement explicit electrostatic "
                "energy coupling"
            )
        if not 0 <= self.density_lmax <= 4 or not 0 <= self.potential_lmax <= 4:
            raise ValueError("multipole angular ranks must lie in [0, 4]")
        if self.potential_lmax < self.density_lmax:
            raise ValueError("potential_lmax must cover density_lmax")
        if not self.hidden or any(width < 4 for width in self.hidden):
            raise ValueError("electronic hidden widths must be at least four")
        if not 0 <= self.polarization_updates <= 8:
            raise ValueError("polarization_updates must lie in [0, 8]")
        if self.method in {"polar", "fukui_auxiliary"} and self.polarization_updates < 1:
            raise ValueError("polar electronic methods require at least one update")
        if self.method == "variational" and self.polarization_updates != 0:
            raise ValueError(
                "the variational core uses SCF convergence, not polarization_updates; "
                "set polarization_updates=0"
            )
        if self.radial_basis < 2:
            raise ValueError("radial_basis must be at least two")
        positive = (
            self.fukui_floor, self.gaussian_width_A, self.coulomb_constant_eV_A,
            self.hardness_floor_eV, self.qeq_jitter_eV,
            self.learned_energy_scale_eV, self.potential_scale_eV,
            self.multipole_update_scale,
            self.constraint_tolerance,
            self.dipole_stiffness_floor_eV_per_eA2,
            self.quadrupole_stiffness_floor_eV_per_eA4,
            self.magnetic_stiffness_floor_eV_per_muB2,
        )
        if min(positive) <= 0:
            raise ValueError("electronic stability scales must be positive")
        if self.reciprocal_kmax < 1 or self.direct_pair_block < 1 or self.qeq_max_atoms < 1:
            raise ValueError("invalid long-range resolution")
        if self.boundary_mode not in {"fixed_charge", "fixed_potential", "mixed"}:
            raise ValueError("invalid electronic boundary mode")

    @property
    def multipole_lmax(self) -> int:
        return self.density_lmax

    @property
    def multipole_dim(self) -> int:
        return (self.density_lmax + 1) ** 2


@dataclass(frozen=True, slots=True)
class ZIVARConfig:
    """Complete architecture and label-semantics contract."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    electronic: ElectronicConfig = field(default_factory=ElectronicConfig)
    spin: SpinConfig = field(default_factory=SpinConfig)
    scf: SCFConfig = field(default_factory=SCFConfig)
    electrostatics: ElectrostaticsConfig = field(default_factory=ElectrostaticsConfig)
    dft_level: str = "unspecified"
    charge_label_scheme: str = "DDEC6_partial_charge_optional"
    spin_label_scheme: str = "noncollinear_spin_vectors_and_effective_fields"
    oxidation_label_scheme: str = "formal_oxidation_state_optional"
    energy_reference_scheme: str = "isolated_atomic_energies"

    def __post_init__(self) -> None:
        if not self.dft_level.strip():
            raise ValueError("dft_level must be nonempty")
        schemes = (
            self.charge_label_scheme, self.spin_label_scheme,
            self.oxidation_label_scheme, self.energy_reference_scheme,
        )
        if any(not value.strip() for value in schemes):
            raise ValueError("label and reference schemes must be nonempty")
        if self.spin.mode == "spin_lattice" and self.electronic.density_lmax < 1:
            raise ValueError("spin-lattice production mode requires density_lmax >= 1")
        if self.electronic.method == "variational" and not self.spin.induced_magnetization:
            raise ValueError(
                "the variational q/p/Q/m core requires induced_magnetization=True"
            )
        if (
            self.electronic.method == "variational"
            and self.spin.mode == "magnitude_auxiliary"
        ):
            raise ValueError(
                "the scalar magnitude auxiliary head cannot run beside the "
                "variational q/p/Q/m core; select an explicit legacy electronic "
                "configuration or disable the auxiliary head"
            )
        if (
            self.electronic.method == "variational"
            and self.electronic.boundary_mode != "fixed_charge"
        ):
            raise ValueError(
                "the variational core currently implements only the exact "
                "fixed_charge constraint; fixed_potential and mixed remain "
                "explicit legacy electronic boundary modes"
            )

    @classmethod
    def production(cls, **overrides: Any) -> ZIVARConfig:
        return _override_config(cls(), overrides)

    @classmethod
    def chgnet_compatible(cls, **overrides: Any) -> ZIVARConfig:
        """Auxiliary-moment preset for scalar DFT moments only."""
        cfg = cls(
            electronic=ElectronicConfig(
                method="polar", energy_coupling="full", polarization_updates=2
            ),
            spin=SpinConfig(mode="magnitude_auxiliary", require_spin_input=False),
            spin_label_scheme="Bader_magnetic_moment_magnitude",
        )
        return _override_config(cfg, overrides)

    @classmethod
    def qeq(cls, **overrides: Any) -> ZIVARConfig:
        cfg = cls(
            electronic=ElectronicConfig(
                method="qeq", energy_coupling="full", density_lmax=0,
                potential_lmax=0
            ),
            spin=SpinConfig(mode="magnitude_auxiliary", require_spin_input=False),
        )
        return _override_config(cfg, overrides)

    @classmethod
    def direct_heads(cls, **overrides: Any) -> ZIVARConfig:
        cfg = cls(
            electronic=ElectronicConfig(
                method="direct", energy_coupling="auxiliary", density_lmax=0,
                potential_lmax=0, polarization_updates=0
            ),
            spin=SpinConfig(mode="magnitude_auxiliary", require_spin_input=False),
        )
        return _override_config(cfg, overrides)

    @classmethod
    def convolution(cls, **overrides: Any) -> ZIVARConfig:
        cfg = cls(
            backbone=BackboneConfig(
                kind="convolution", channels=192, max_ell=0, correlation=1,
                num_interactions=4, radial_mlp=(192, 192, 128)
            ),
            electronic=ElectronicConfig(),
        )
        return _override_config(cfg, overrides)

    @classmethod
    def small(cls, **overrides: Any) -> ZIVARConfig:
        cfg = cls(
            backbone=BackboneConfig(channels=64, max_ell=2, correlation=3),
            electronic=ElectronicConfig(
                density_lmax=1, potential_lmax=1, hidden=(96, 96),
                reciprocal_kmax=4
            ),
            spin=SpinConfig(hidden=(96, 64)),
        )
        return _override_config(cfg, overrides)

    @classmethod
    def balanced(cls, **overrides: Any) -> ZIVARConfig:
        return cls.production(**overrides)

    @classmethod
    def large(cls, **overrides: Any) -> ZIVARConfig:
        cfg = cls(
            backbone=BackboneConfig(
                channels=128,
                hidden_irreps="128x0e + 128x1o + 128x2e",
                num_interactions=3, correlation=3, max_ell=4, num_bessel=12,
                radial_mlp=(128, 128, 128),
            ),
            electronic=ElectronicConfig(
                density_lmax=3, potential_lmax=3, hidden=(256, 256, 128),
                reciprocal_kmax=6,
            ),
            spin=SpinConfig(hidden=(256, 192, 128)),
        )
        return _override_config(cfg, overrides)

    @classmethod
    def maximal(cls, **overrides: Any) -> ZIVARConfig:
        cfg = cls.large(
            backbone__hidden_irreps="128x0e + 128x1o + 128x2e + 128x3o + 128x4e",
            backbone__correlation=4,
            electronic__density_lmax=4,
            electronic__potential_lmax=4,
        )
        return _override_config(cfg, overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ZIVARConfig:
        values = dict(payload)
        backbone = dict(values.get("backbone", {}))
        for key in ("atomic_numbers", "atomic_energies_eV", "radial_mlp"):
            if backbone.get(key) is not None:
                backbone[key] = tuple(backbone[key])
        if isinstance(backbone.get("correlation"), list):
            backbone["correlation"] = tuple(backbone["correlation"])
        electronic = dict(values.get("electronic", {}))
        if electronic.get("hidden") is not None:
            electronic["hidden"] = tuple(electronic["hidden"])
        oxidation = dict(electronic.get("oxidation", {}))
        if oxidation.get("hidden") is not None:
            oxidation["hidden"] = tuple(oxidation["hidden"])
        electronic["oxidation"] = OxidationConfig(**oxidation)
        spin = dict(values.get("spin", {}))
        if spin.get("hidden") is not None:
            spin["hidden"] = tuple(spin["hidden"])
        scf = dict(values.get("scf", {}))
        electrostatics = dict(values.get("electrostatics", {}))
        if electrostatics.get("mesh") is not None:
            electrostatics["mesh"] = tuple(electrostatics["mesh"])
        values["backbone"] = BackboneConfig(**backbone)
        values["electronic"] = ElectronicConfig(**electronic)
        values["spin"] = SpinConfig(**spin)
        values["scf"] = SCFConfig(**scf)
        values["electrostatics"] = ElectrostaticsConfig(**electrostatics)
        return cls(**values)


def _override_config(config: ZIVARConfig, overrides: dict[str, Any]) -> ZIVARConfig:
    backbone_values: dict[str, Any] = {}
    electronic_values: dict[str, Any] = {}
    oxidation_values: dict[str, Any] = {}
    spin_values: dict[str, Any] = {}
    scf_values: dict[str, Any] = {}
    electrostatics_values: dict[str, Any] = {}
    top_values: dict[str, Any] = {}
    for key, value in overrides.items():
        if key.startswith("backbone__"):
            backbone_values[key.removeprefix("backbone__")] = value
        elif key.startswith("electronic__oxidation__"):
            oxidation_values[key.removeprefix("electronic__oxidation__")] = value
        elif key.startswith("electronic__"):
            electronic_values[key.removeprefix("electronic__")] = value
        elif key.startswith("spin__"):
            spin_values[key.removeprefix("spin__")] = value
        elif key.startswith("scf__"):
            scf_values[key.removeprefix("scf__")] = value
        elif key.startswith("electrostatics__"):
            electrostatics_values[key.removeprefix("electrostatics__")] = value
        else:
            top_values[key] = value
    electronic = replace(
        config.electronic,
        oxidation=replace(config.electronic.oxidation, **oxidation_values),
        **electronic_values,
    )
    return replace(
        config,
        backbone=replace(config.backbone, **backbone_values),
        electronic=electronic,
        spin=replace(config.spin, **spin_values),
        scf=replace(config.scf, **scf_values),
        electrostatics=replace(config.electrostatics, **electrostatics_values),
        **top_values,
    )


__all__ = [
    "ARCHITECTURE_REVISION", "BackboneConfig", "ElectronicConfig",
    "ElectrostaticsConfig",
    "LEGACY_ARCHITECTURE_REVISIONS", "NUMERICS_REVISION", "OxidationConfig",
    "SCFConfig", "SUPPORTED_MACE_SERIES", "SpinConfig", "ZIVARConfig",
    "ZIVAR_VERSION",
]
