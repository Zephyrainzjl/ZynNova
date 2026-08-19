from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .exceptions import DFTConfigurationError


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AIMDEnsemble(_StrEnum):
    NVE = "nve"
    NVT_LANGEVIN = "nvt_langevin"


@dataclass(slots=True)
class ElectronicConfig:
    """Common settings for an electronic-structure calculator.

    The strongly typed fields are consumed by the native PySCF and GPAW
    builders. Other ASE calculators receive ``backend_kwargs`` unchanged.
    """

    backend: str = "pyscf"
    xc: str = "PBE"
    basis: str = "def2-svp"
    charge: int = 0
    spin: int = 0
    scf_tolerance: float = 1.0e-9
    max_scf_cycles: int = 100
    grid_level: int = 4
    density_fit: bool = True
    auxiliary_basis: str | None = None
    use_gpu: bool = False
    newton_on_failure: bool = True
    plane_wave_cutoff_eV: float = 500.0
    kpoints: tuple[int, int, int] = (1, 1, 1)
    mode: str = "auto"
    txt: str | None = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.backend.strip():
            raise DFTConfigurationError("electronic backend cannot be empty")
        if not self.xc.strip():
            raise DFTConfigurationError("xc cannot be empty")
        if int(self.charge) != self.charge:
            raise DFTConfigurationError("charge must be an integer")
        if int(self.spin) != self.spin or self.spin < 0:
            raise DFTConfigurationError(
                "spin must be a non-negative integer (PySCF uses 2S)"
            )
        self.charge = int(self.charge)
        self.spin = int(self.spin)
        if self.scf_tolerance <= 0:
            raise DFTConfigurationError("scf_tolerance must be positive")
        if int(self.max_scf_cycles) != self.max_scf_cycles or self.max_scf_cycles <= 0:
            raise DFTConfigurationError("max_scf_cycles must be a positive integer")
        if int(self.grid_level) != self.grid_level or self.grid_level < 0:
            raise DFTConfigurationError("grid_level must be a non-negative integer")
        self.max_scf_cycles = int(self.max_scf_cycles)
        self.grid_level = int(self.grid_level)
        if self.plane_wave_cutoff_eV <= 0:
            raise DFTConfigurationError("plane_wave_cutoff_eV must be positive")
        if len(self.kpoints) != 3 or any(int(value) <= 0 for value in self.kpoints):
            raise DFTConfigurationError("kpoints must contain three positive integers")
        self.kpoints = tuple(int(value) for value in self.kpoints)
        self.backend_kwargs = dict(self.backend_kwargs)


@dataclass(slots=True)
class AIMDOutputConfig:
    directory: str | Path = "zynnova-aimd"
    trajectory_filename: str = "trajectory.traj"
    thermo_filename: str = "thermo.csv"
    checkpoint_filename: str = "checkpoint.npz"
    metadata_filename: str = "run.json"
    trajectory_interval: int = 1
    log_interval: int = 1
    checkpoint_interval: int = 10
    overwrite: bool = False
    append: bool = False
    store_in_memory: bool = True

    def validate(self) -> None:
        self.directory = Path(self.directory)
        for name in ("trajectory_interval", "log_interval", "checkpoint_interval"):
            value = int(getattr(self, name))
            if value < 0:
                raise DFTConfigurationError(f"{name} must be non-negative")
            setattr(self, name, value)


@dataclass(slots=True)
class AIMDSafetyConfig:
    check_interval: int = 1
    stop_on_nonfinite: bool = True
    max_force_eV_per_A: float | None = 1.0e4
    max_temperature_K: float | None = 1.0e6
    min_distance_A: float | None = 0.15
    max_energy_jump_per_atom_eV: float | None = 100.0

    def validate(self) -> None:
        if self.check_interval <= 0:
            raise DFTConfigurationError("safety check_interval must be positive")
        for name in (
            "max_force_eV_per_A",
            "max_temperature_K",
            "min_distance_A",
            "max_energy_jump_per_atom_eV",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise DFTConfigurationError(f"{name} must be positive when provided")


@dataclass(slots=True)
class AIMDConfig:
    ensemble: AIMDEnsemble | str = AIMDEnsemble.NVE
    steps: int = 100
    timestep_fs: float = 0.5
    temperature_K: float = 300.0
    initial_temperature_K: float | None = None
    friction_per_fs: float = 0.01
    random_seed: int = 0
    initialize_velocities: bool = True
    remove_center_of_mass: bool = True
    exact_initial_temperature: bool = True
    fixed_indices: tuple[int, ...] = ()
    wrap_positions: bool = True
    integrator_backend: str = "auto"
    output: AIMDOutputConfig = field(default_factory=AIMDOutputConfig)
    safety: AIMDSafetyConfig = field(default_factory=AIMDSafetyConfig)

    def validate(self) -> None:
        self.ensemble = AIMDEnsemble(self.ensemble)
        if int(self.steps) != self.steps or self.steps < 0:
            raise DFTConfigurationError("steps must be a non-negative integer")
        self.steps = int(self.steps)
        if self.timestep_fs <= 0:
            raise DFTConfigurationError("timestep_fs must be positive")
        if self.temperature_K <= 0:
            raise DFTConfigurationError("temperature_K must be positive")
        if self.initial_temperature_K is not None and self.initial_temperature_K <= 0:
            raise DFTConfigurationError("initial_temperature_K must be positive")
        if self.friction_per_fs < 0:
            raise DFTConfigurationError("friction_per_fs must be non-negative")
        if int(self.random_seed) != self.random_seed or self.random_seed < 0:
            raise DFTConfigurationError("random_seed must be a non-negative integer")
        self.random_seed = int(self.random_seed)
        if self.integrator_backend not in {"auto", "python", "cpp"}:
            raise DFTConfigurationError("integrator_backend must be 'auto', 'python', or 'cpp'")
        normalized_indices = tuple(sorted({int(index) for index in self.fixed_indices}))
        if any(index < 0 for index in normalized_indices):
            raise DFTConfigurationError("fixed_indices cannot contain negative indices")
        self.fixed_indices = normalized_indices
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
