from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ....dynamics.adapters import to_ase_atoms
from ....dynamics.config import RelaxationConfig
from .electronic import (
    ChargeOxidationCalibrator,
    OxidationStateAssignment,
    OxidationStateResolver,
)
from .materials import _resolve_calculator, optimize_jouleweave_structure
from .ncm import (
    NCMCompositionEnumerator,
    NCMEnumerationConfig,
    ncm_mixing_statistics,
)


@dataclass(slots=True)
class CathodeCyclingConfig:
    li_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    antisite_pairs: tuple[int, ...] = (0, 1)
    max_structures_per_x: int = 16
    relax_structures: bool = True
    relax_cell: bool = True
    relaxation_config: RelaxationConfig | None = None
    compute_voltages: bool = True
    resolve_oxidation_states: bool = True
    oxygen_redox: bool = False
    variable_co_mn: bool = False
    require_unique_oxidation_states: bool = False
    total_charge_e: int = 0

    def __post_init__(self) -> None:
        self.li_fractions = tuple(float(value) for value in self.li_fractions)
        if not self.li_fractions:
            raise ValueError("li_fractions cannot be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.li_fractions):
            raise ValueError("all Li fractions must lie in [0, 1]")
        self.antisite_pairs = tuple(sorted({int(value) for value in self.antisite_pairs}))
        if any(value < 0 for value in self.antisite_pairs):
            raise ValueError("antisite pair counts cannot be negative")
        if self.max_structures_per_x < 1:
            raise ValueError("max_structures_per_x must be positive")


@dataclass(slots=True)
class CathodePhaseRecord:
    identifier: str
    x_li: float
    n_li: int
    n_transition_metals: int
    energy_eV: float
    energy_eV_per_atom: float
    converged: bool | None
    antisite_pair_count: int
    mixing_statistics: dict[str, float | int]
    structure: Any = field(repr=False)
    magmoms_mu_B: np.ndarray | None = field(default=None, repr=False)
    charges_e: np.ndarray | None = field(default=None, repr=False)
    oxidation_states: np.ndarray | None = field(default=None, repr=False)
    oxidation_labels: tuple[str, ...] | None = None
    oxidation_unique: bool | None = None
    oxidation_ambiguity_gap: float | None = None
    oxidation_error: str | None = None

    def to_dict(self, *, include_atom_arrays: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "identifier": self.identifier,
            "x_li": self.x_li,
            "n_li": self.n_li,
            "n_transition_metals": self.n_transition_metals,
            "energy_eV": self.energy_eV,
            "energy_eV_per_atom": self.energy_eV_per_atom,
            "converged": self.converged,
            "antisite_pair_count": self.antisite_pair_count,
            "mixing_statistics": dict(self.mixing_statistics),
            "oxidation_labels": (
                None if self.oxidation_labels is None else list(self.oxidation_labels)
            ),
            "oxidation_unique": self.oxidation_unique,
            "oxidation_ambiguity_gap": self.oxidation_ambiguity_gap,
            "oxidation_error": self.oxidation_error,
        }
        if include_atom_arrays:
            payload.update(
                {
                    "magmoms_mu_B": (
                        None if self.magmoms_mu_B is None else self.magmoms_mu_B.tolist()
                    ),
                    "charges_e": (None if self.charges_e is None else self.charges_e.tolist()),
                    "oxidation_states": (
                        None if self.oxidation_states is None else self.oxidation_states.tolist()
                    ),
                }
            )
        return payload


@dataclass(slots=True)
class VoltageStep:
    x_high: float
    x_low: float
    lithium_removed: int
    average_voltage_V: float
    high_phase_identifier: str
    low_phase_identifier: str


@dataclass(slots=True)
class CathodeCyclingResult:
    phases: tuple[CathodePhaseRecord, ...]
    ground_states: tuple[CathodePhaseRecord, ...]
    voltage_steps: tuple[VoltageStep, ...]
    lithium_reference_energy_eV_per_atom: float | None
    output_directory: Path

    def to_dict(self, *, include_atom_arrays: bool = True) -> dict[str, Any]:
        return {
            "lithium_reference_energy_eV_per_atom": (self.lithium_reference_energy_eV_per_atom),
            "phases": [
                phase.to_dict(include_atom_arrays=include_atom_arrays) for phase in self.phases
            ],
            "ground_states": [
                phase.to_dict(include_atom_arrays=include_atom_arrays)
                for phase in self.ground_states
            ],
            "voltage_steps": [
                {
                    "x_high": step.x_high,
                    "x_low": step.x_low,
                    "lithium_removed": step.lithium_removed,
                    "average_voltage_V": step.average_voltage_V,
                    "high_phase_identifier": step.high_phase_identifier,
                    "low_phase_identifier": step.low_phase_identifier,
                }
                for step in self.voltage_steps
            ],
        }

    def write_json(
        self,
        path: str | Path | None = None,
        *,
        include_atom_arrays: bool = True,
    ) -> Path:
        target = self.output_directory / "cycling-summary.json" if path is None else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(include_atom_arrays=include_atom_arrays),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return target


def _calculator_property(atoms: Any, name: str) -> np.ndarray | None:
    calculator = atoms.calc
    if calculator is None:
        return None
    if name not in set(getattr(calculator, "implemented_properties", ())):
        return None
    return np.asarray(calculator.get_property(name, atoms), dtype=float)


def _state_values_from_calculator(
    calculator: Any,
    class_count: int,
) -> np.ndarray:
    wrapped = getattr(calculator, "model", None)
    model = getattr(wrapped, "model", wrapped)
    config = getattr(model, "config", None)
    minimum = int(getattr(config, "oxidation_state_min", -4))
    maximum = int(getattr(config, "oxidation_state_max", minimum + class_count - 1))
    values = np.arange(minimum, maximum + 1, dtype=int)
    if len(values) != class_count:
        raise ValueError("calculator oxidation-state class count does not match its model config")
    return values


def _model_config_from_calculator(calculator: Any) -> Any | None:
    wrapped = getattr(calculator, "model", None)
    model = getattr(wrapped, "model", wrapped)
    return getattr(model, "config", None)


class CathodeCyclingWorkflow:
    """Delithiate, relax, rank, voltage, redox-resolve, and quantify mixing."""

    def __init__(
        self,
        potential: Any,
        *,
        device: str = "auto",
        dtype: str = "float32",
        compile_model: bool = False,
        oxidation_resolver: OxidationStateResolver | None = None,
        charge_calibrator: ChargeOxidationCalibrator | None = None,
    ) -> None:
        self.potential = potential
        self.device = device
        self.dtype = dtype
        self.compile_model = bool(compile_model)
        self.oxidation_resolver = oxidation_resolver or OxidationStateResolver()
        self.charge_calibrator = charge_calibrator
        self._calculator_cache: dict[bool, Any] = {}

    def _calculator(self, *, require_stress: bool = False) -> Any:
        key = bool(require_stress)
        if key not in self._calculator_cache:
            self._calculator_cache[key] = _resolve_calculator(
                self.potential,
                require_stress=key,
                device=self.device,
                dtype=self.dtype,
                compile_model=self.compile_model,
            )
        return self._calculator_cache[key]

    def _lithium_reference(
        self,
        *,
        energy_eV_per_atom: float | None,
        structure: Any | None,
    ) -> float | None:
        if energy_eV_per_atom is not None:
            return float(energy_eV_per_atom)
        if structure is None:
            return None
        atoms = to_ase_atoms(structure).copy()
        atoms.calc = self._calculator()
        return float(atoms.get_potential_energy()) / len(atoms)

    def _electronic_assignment(
        self,
        atoms: Any,
        probabilities: np.ndarray | None,
        charges: np.ndarray | None,
        config: CathodeCyclingConfig,
    ) -> OxidationStateAssignment | None:
        if probabilities is None or not config.resolve_oxidation_states:
            return None
        if self.charge_calibrator is not None:
            model_config = _model_config_from_calculator(atoms.calc)
            model_scheme = getattr(
                model_config,
                "charge_label_scheme",
                "unspecified",
            )
            if (
                model_scheme != "unspecified"
                and model_scheme != self.charge_calibrator.scheme
            ):
                raise ValueError(
                    "charge calibrator scheme does not match the model charge head: "
                    f"{self.charge_calibrator.scheme!r} != {model_scheme!r}"
                )
        values = _state_values_from_calculator(atoms.calc, probabilities.shape[-1])
        return self.oxidation_resolver.resolve_ncm(
            atoms.get_atomic_numbers(),
            probabilities,
            values,
            target_total_charge=config.total_charge_e,
            oxygen_redox=config.oxygen_redox,
            variable_co_mn=config.variable_co_mn,
            partition_charges_e=charges,
            calibrator=self.charge_calibrator,
        )

    def run(
        self,
        parent_structure: Any,
        *,
        config: CathodeCyclingConfig | None = None,
        enumeration_config: NCMEnumerationConfig | None = None,
        lithium_reference_energy_eV_per_atom: float | None = None,
        lithium_reference_structure: Any | None = None,
        output_directory: str | Path = "jouleweave-cathode-cycling",
    ) -> CathodeCyclingResult:
        config = config or CathodeCyclingConfig()
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        enumerator = NCMCompositionEnumerator(
            parent_structure,
            enumeration_config,
        )
        candidates = enumerator.enumerate(
            config.li_fractions,
            antisite_pairs=config.antisite_pairs,
            max_structures_per_x=config.max_structures_per_x,
        )
        if not candidates:
            raise ValueError("NCM enumeration produced no structures")
        li_reference = self._lithium_reference(
            energy_eV_per_atom=lithium_reference_energy_eV_per_atom,
            structure=lithium_reference_structure,
        )
        if config.compute_voltages and li_reference is None:
            raise ValueError(
                "voltage calculation requires lithium_reference_energy_eV_per_atom "
                "or a lithium_reference_structure evaluated by the same checkpoint"
            )

        phases: list[CathodePhaseRecord] = []
        for index, candidate in enumerate(candidates):
            identifier = (
                f"x-{candidate.x_li:.6f}-a-{candidate.antisite_pair_count:02d}-c-{index:05d}"
            )
            phase_directory = output / identifier
            phase_directory.mkdir(parents=True, exist_ok=True)
            atoms = candidate.atoms.copy()
            atoms.info["total_charge"] = float(config.total_charge_e)
            converged: bool | None = None
            if config.relax_structures:
                relaxation = optimize_jouleweave_structure(
                    atoms,
                    self._calculator(require_stress=config.relax_cell),
                    relax_cell=config.relax_cell,
                    config=config.relaxation_config,
                    output_directory=phase_directory / "relax",
                    device=self.device,
                    dtype=self.dtype,
                    compile_model=self.compile_model,
                )
                atoms = to_ase_atoms(relaxation.final_structure)
                atoms.info["total_charge"] = float(config.total_charge_e)
                converged = bool(relaxation.converged)
            atoms.calc = self._calculator()
            energy = float(atoms.get_potential_energy())
            magmoms = _calculator_property(atoms, "magmoms")
            charges = _calculator_property(atoms, "charges")
            probabilities = _calculator_property(
                atoms,
                "oxidation_state_probabilities",
            )
            if config.require_unique_oxidation_states and probabilities is None:
                raise RuntimeError(
                    f"{identifier}: calculator does not provide oxidation-state "
                    "probabilities"
                )
            assignment = None
            assignment_error = None
            try:
                assignment = self._electronic_assignment(
                    atoms,
                    probabilities,
                    charges,
                    config,
                )
            except ValueError as exc:
                assignment_error = str(exc)
                if config.require_unique_oxidation_states:
                    raise
            if (
                assignment is not None
                and config.require_unique_oxidation_states
                and not assignment.is_unique
            ):
                raise RuntimeError(
                    f"{identifier}: oxidation-state assignment is ambiguous "
                    f"(gap={assignment.ambiguity_gap:.6g})"
                )
            try:
                from ase.io import write

                write(str(phase_directory / "relaxed.cif"), atoms)
            except ImportError:
                pass
            phases.append(
                CathodePhaseRecord(
                    identifier=identifier,
                    x_li=candidate.x_li,
                    n_li=candidate.n_li,
                    n_transition_metals=candidate.n_transition_metals,
                    energy_eV=energy,
                    energy_eV_per_atom=energy / len(atoms),
                    converged=converged,
                    antisite_pair_count=candidate.antisite_pair_count,
                    mixing_statistics=ncm_mixing_statistics(atoms),
                    structure=atoms.copy(),
                    magmoms_mu_B=(None if magmoms is None else magmoms.reshape(-1)),
                    charges_e=None if charges is None else charges.reshape(-1),
                    oxidation_states=(None if assignment is None else assignment.states.copy()),
                    oxidation_labels=(None if assignment is None else assignment.labels),
                    oxidation_unique=(None if assignment is None else assignment.is_unique),
                    oxidation_ambiguity_gap=(
                        None if assignment is None else assignment.ambiguity_gap
                    ),
                    oxidation_error=assignment_error,
                )
            )

        ground_states = []
        for x_value in sorted({phase.x_li for phase in phases}, reverse=True):
            group = [phase for phase in phases if phase.x_li == x_value]
            ground_states.append(min(group, key=lambda phase: phase.energy_eV))

        voltage_steps: list[VoltageStep] = []
        if config.compute_voltages:
            assert li_reference is not None
            for high, low in zip(ground_states[:-1], ground_states[1:], strict=True):
                lithium_removed = high.n_li - low.n_li
                if lithium_removed <= 0:
                    raise ValueError("ground-state Li counts must decrease with x")
                voltage = (
                    low.energy_eV + lithium_removed * li_reference - high.energy_eV
                ) / lithium_removed
                voltage_steps.append(
                    VoltageStep(
                        x_high=high.x_li,
                        x_low=low.x_li,
                        lithium_removed=lithium_removed,
                        average_voltage_V=float(voltage),
                        high_phase_identifier=high.identifier,
                        low_phase_identifier=low.identifier,
                    )
                )
        result = CathodeCyclingResult(
            phases=tuple(phases),
            ground_states=tuple(ground_states),
            voltage_steps=tuple(voltage_steps),
            lithium_reference_energy_eV_per_atom=li_reference,
            output_directory=output,
        )
        result.write_json()
        return result


__all__ = [
    "CathodeCyclingConfig",
    "CathodeCyclingResult",
    "CathodeCyclingWorkflow",
    "CathodePhaseRecord",
    "VoltageStep",
]
