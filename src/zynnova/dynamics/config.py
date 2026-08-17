from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import math
from typing import Any

from .exceptions import ConfigurationError


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Ensemble(_StrEnum):
    NVE = "nve"
    NVT_LANGEVIN = "nvt_langevin"
    NVT_BERENDSEN = "nvt_berendsen"
    NVT_BUSSI = "nvt_bussi"
    NVT_ANDERSEN = "nvt_andersen"
    NVT_NOSE_HOOVER = "nvt_nose_hoover"
    NPT_BERENDSEN = "npt_berendsen"
    NPT_BERENDSEN_MASKED = "npt_berendsen_masked"
    NPT_MTK_ISOTROPIC = "npt_mtk_isotropic"
    NPT_MTK_FULL = "npt_mtk_full"
    NPT_MTK_MASKED = "npt_mtk_masked"
    NPT_LANGEVIN_BAOAB = "npt_langevin_baoab"
    NPT_MELCHIONNA = "npt_melchionna"


class OptimizerMethod(_StrEnum):
    FIRE2 = "fire2"
    FIRE = "fire"
    BFGS = "bfgs"
    LBFGS = "lbfgs"
    LBFGS_LINESEARCH = "lbfgs_linesearch"
    BFGS_LINESEARCH = "bfgs_linesearch"
    MDMIN = "mdmin"
    GPMIN = "gpmin"
    SCIPY_BFGS = "scipy_bfgs"
    SCIPY_CG = "scipy_cg"
    CELL_AWARE_BFGS = "cell_aware_bfgs"


class CellMode(_StrEnum):
    FIXED = "fixed"
    FRECHET = "frechet"
    UNIT_CELL = "unit_cell"
    EXPONENTIAL = "exponential"
    STRAIN = "strain"
    CELL_AWARE = "cell_aware"


class VelocityMode(_StrEnum):
    KEEP = "keep"
    MAXWELL_BOLTZMANN = "maxwell_boltzmann"
    ZERO = "zero"


@dataclass(slots=True)
class VelocityConfig:
    mode: VelocityMode | str = VelocityMode.MAXWELL_BOLTZMANN
    temperature_K: float | None = None
    force_temperature: bool = False
    remove_translation: bool = True
    remove_rotation: bool = False
    preserve_temperature_after_cleanup: bool = True
    seed: int | None = 0

    def validate(self, target_temperature_K: float | None = None) -> None:
        self.mode = VelocityMode(self.mode)
        temperature = (
            self.temperature_K if self.temperature_K is not None else target_temperature_K
        )
        if self.seed is not None and int(self.seed) != self.seed:
            raise ConfigurationError("velocity seed must be an integer or None")
        if self.mode is VelocityMode.MAXWELL_BOLTZMANN:
            if temperature is None or not math.isfinite(float(temperature)) or temperature <= 0:
                raise ConfigurationError(
                    "Maxwell-Boltzmann initialization requires a positive temperature"
                )


@dataclass(slots=True)
class OutputConfig:
    directory: str | Path = "zynnova-md"
    trajectory_filename: str = "trajectory.traj"
    thermo_filename: str = "thermo.csv"
    metadata_filename: str = "run.json"
    checkpoint_filename: str = "checkpoint.traj"
    checkpoint_state_filename: str = "checkpoint.json"
    trajectory_interval: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 1000
    append: bool = False
    overwrite: bool = False
    store_in_memory: bool = True

    def validate(self) -> None:
        self.directory = Path(self.directory)
        for name in (
            "trajectory_filename",
            "thermo_filename",
            "metadata_filename",
            "checkpoint_filename",
            "checkpoint_state_filename",
        ):
            if not str(getattr(self, name)).strip():
                raise ConfigurationError(f"{name} cannot be empty")
        for name in ("trajectory_interval", "log_interval", "checkpoint_interval"):
            value = int(getattr(self, name))
            if value < 0:
                raise ConfigurationError(f"{name} must be non-negative")
            setattr(self, name, value)


@dataclass(slots=True)
class SafetyConfig:
    check_interval: int = 10
    stop_on_nonfinite: bool = True
    max_force_eV_per_A: float | None = 1.0e4
    max_temperature_K: float | None = 1.0e6
    min_temperature_K: float | None = None
    minimum_distance_A: float | None = 0.2
    max_energy_per_atom_eV: float | None = 1.0e6
    max_energy_jump_per_atom_eV: float | None = 1.0e4
    min_volume_A3: float | None = 1.0e-6
    wrap_positions: bool = False

    def validate(self) -> None:
        if self.check_interval <= 0:
            raise ConfigurationError("safety check_interval must be positive")
        for name in (
            "max_force_eV_per_A",
            "max_temperature_K",
            "minimum_distance_A",
            "max_energy_per_atom_eV",
            "max_energy_jump_per_atom_eV",
            "min_volume_A3",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or value <= 0):
                raise ConfigurationError(f"{name} must be finite and positive when provided")
        if self.min_temperature_K is not None:
            if not math.isfinite(float(self.min_temperature_K)) or self.min_temperature_K < 0:
                raise ConfigurationError("min_temperature_K must be finite and nonnegative")
        if (
            self.min_temperature_K is not None
            and self.max_temperature_K is not None
            and self.min_temperature_K > self.max_temperature_K
        ):
            raise ConfigurationError("minimum temperature exceeds maximum temperature")


@dataclass(slots=True)
class MDConfig:
    ensemble: Ensemble | str = Ensemble.NVE
    steps: int = 1000
    timestep_fs: float = 1.0
    temperature_K: float | None = 300.0
    pressure_GPa: float | None = None
    friction_per_fs: float = 0.01
    thermostat_time_fs: float = 100.0
    barostat_time_fs: float = 1000.0
    compressibility_GPa_inv: float = 4.57e-5
    andersen_probability: float = 0.01
    thermostat_chain_length: int = 3
    barostat_chain_length: int = 3
    thermostat_substeps: int = 1
    barostat_substeps: int = 1
    cell_mask: tuple[bool, bool, bool] = (True, True, True)
    pressure_mask: tuple[int, int, int] = (1, 1, 1)
    remove_com_interval: int = 0
    random_seed: int | None = 0
    extra_integrator_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.ensemble = Ensemble(self.ensemble)
        if int(self.steps) != self.steps or self.steps < 0:
            raise ConfigurationError("steps must be a non-negative integer")
        self.steps = int(self.steps)
        if not math.isfinite(float(self.timestep_fs)) or self.timestep_fs <= 0:
            raise ConfigurationError("timestep_fs must be finite and positive")
        if self.ensemble is not Ensemble.NVE:
            if (
                self.temperature_K is None
                or not math.isfinite(float(self.temperature_K))
                or self.temperature_K <= 0
            ):
                raise ConfigurationError(f"{self.ensemble.value} requires temperature_K > 0")
        if self.ensemble.value.startswith("npt"):
            if self.pressure_GPa is None or not math.isfinite(float(self.pressure_GPa)):
                raise ConfigurationError(f"{self.ensemble.value} requires finite pressure_GPa")
            if not math.isfinite(float(self.barostat_time_fs)) or self.barostat_time_fs <= 0:
                raise ConfigurationError("barostat_time_fs must be finite and positive")
            if (
                not math.isfinite(float(self.compressibility_GPa_inv))
                or self.compressibility_GPa_inv <= 0
            ):
                raise ConfigurationError("compressibility_GPa_inv must be finite and positive")
        if not math.isfinite(float(self.thermostat_time_fs)) or self.thermostat_time_fs <= 0:
            raise ConfigurationError("thermostat_time_fs must be finite and positive")
        if not math.isfinite(float(self.friction_per_fs)) or self.friction_per_fs < 0:
            raise ConfigurationError("friction_per_fs must be finite and non-negative")
        if (
            not math.isfinite(float(self.andersen_probability))
            or not 0 <= self.andersen_probability <= 1
        ):
            raise ConfigurationError("andersen_probability must lie in [0, 1]")
        for name in (
            "thermostat_chain_length",
            "barostat_chain_length",
            "thermostat_substeps",
            "barostat_substeps",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if int(self.remove_com_interval) != self.remove_com_interval or self.remove_com_interval < 0:
            raise ConfigurationError("remove_com_interval must be a non-negative integer")
        self.remove_com_interval = int(self.remove_com_interval)
        if self.random_seed is not None and int(self.random_seed) != self.random_seed:
            raise ConfigurationError("random_seed must be an integer or None")
        if len(self.cell_mask) != 3 or len(self.pressure_mask) != 3:
            raise ConfigurationError("cell_mask and pressure_mask must contain three values")
        self.cell_mask = tuple(bool(value) for value in self.cell_mask)
        if any(int(value) not in (0, 1) for value in self.pressure_mask):
            raise ConfigurationError("pressure_mask values must be 0 or 1")
        self.pressure_mask = tuple(int(value) for value in self.pressure_mask)


@dataclass(slots=True)
class RelaxationConfig:
    optimizer: OptimizerMethod | str = OptimizerMethod.FIRE2
    cell_mode: CellMode | str = CellMode.FIXED
    fmax_eV_per_A: float = 0.05
    smax_eV_per_A3: float = 0.005
    max_steps: int = 1000
    maxstep_A: float = 0.2
    restart_filename: str | None = None
    trajectory_filename: str | None = "relaxation.traj"
    logfile: str | None = "relaxation.log"
    append_trajectory: bool = False
    cell_mask: tuple[bool, bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        True,
        True,
        True,
    )
    hydrostatic_strain: bool = False
    constant_volume: bool = False
    scalar_pressure_GPa: float = 0.0
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    filter_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.optimizer = OptimizerMethod(self.optimizer)
        self.cell_mode = CellMode(self.cell_mode)
        if not math.isfinite(float(self.fmax_eV_per_A)) or self.fmax_eV_per_A <= 0:
            raise ConfigurationError("fmax_eV_per_A must be finite and positive")
        if not math.isfinite(float(self.smax_eV_per_A3)) or self.smax_eV_per_A3 <= 0:
            raise ConfigurationError("smax_eV_per_A3 must be finite and positive")
        if self.max_steps <= 0:
            raise ConfigurationError("max_steps must be positive")
        if self.maxstep_A <= 0:
            raise ConfigurationError("maxstep_A must be positive")
        if len(self.cell_mask) != 6:
            raise ConfigurationError("cell_mask must have six Voigt components")
        if self.optimizer is OptimizerMethod.CELL_AWARE_BFGS:
            self.cell_mode = CellMode.CELL_AWARE


@dataclass(slots=True)
class RunConfig:
    md: MDConfig = field(default_factory=MDConfig)
    velocities: VelocityConfig = field(default_factory=VelocityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def validate(self) -> None:
        self.md.validate()
        self.velocities.validate(self.md.temperature_K)
        self.output.validate()
        self.safety.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        return normalize(data)
