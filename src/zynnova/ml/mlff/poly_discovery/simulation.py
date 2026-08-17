from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .observables import DielectricEstimate, dielectric_from_dipole_fluctuations
from .potential import LoadedPolymerPotential, load_polymer_potential


@dataclass(frozen=True, slots=True)
class EnergyProfile:
    energies_eV: np.ndarray
    relative_energies_eV: np.ndarray
    coordinate: np.ndarray
    barrier_eV: float
    minimum_index: int
    maximum_index: int


@dataclass(frozen=True, slots=True)
class DipoleProbeResult:
    dipoles_eA: np.ndarray
    volumes_A3: np.ndarray
    dielectric: DielectricEstimate
    frame_count: int
    warning: str | None


class PolymerMechanismSimulator:
    """Use JouleWeave/ASE workflows to produce mechanism-level observables."""

    def __init__(
        self,
        potential: LoadedPolymerPotential | str | Path | Any,
        *,
        device: str = "auto",
        dtype: str = "float32",
        analytic_stress: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.analytic_stress = bool(analytic_stress)
        if isinstance(potential, LoadedPolymerPotential):
            self.loaded = potential
            self.calculator = potential.calculator
            self.potential = potential.model
        elif isinstance(potential, (str, Path)):
            self.loaded = load_polymer_potential(
                potential,
                device=device,
                dtype=dtype,
                analytic_stress=analytic_stress,
            )
            self.calculator = self.loaded.calculator
            self.potential = self.loaded.model
        elif hasattr(potential, "implemented_properties"):
            self.loaded = None
            self.calculator = potential
            self.potential = potential
        else:
            from ..jouleweave import jouleweave_calculator

            self.loaded = None
            self.potential = potential
            self.calculator = jouleweave_calculator(
                potential,
                device=device,
                dtype=dtype,
                analytic_stress=analytic_stress,
            )

    def relax(
        self,
        structure: Any,
        *,
        config: Any | None = None,
        output_directory: str | Path = "poly-discovery-relax",
    ) -> Any:
        from ....dynamics import relax

        return relax(
            structure,
            self.calculator,
            config,
            output_directory=output_directory,
        )

    def run_md(
        self,
        structure: Any,
        *,
        config: Any | None = None,
        resume: bool = False,
    ) -> Any:
        from ....dynamics import run_md

        return run_md(structure, self.calculator, config, resume=resume)

    def neb(
        self,
        initial: Any,
        final: Any,
        *,
        config: Any | None = None,
        output_directory: str | Path = "poly-discovery-neb",
    ) -> Any:
        from ..jouleweave import JouleWeaveNEB

        workflow = JouleWeaveNEB(
            self.potential,
            device=self.device,
            dtype=self.dtype,
        )
        return workflow.run(
            initial,
            final,
            config=config,
            output_directory=output_directory,
        )

    def energy_profile(
        self,
        frames: Sequence[Any],
        *,
        coordinate: Sequence[float] | None = None,
    ) -> EnergyProfile:
        if len(frames) < 2:
            raise ValueError("an energy profile requires at least two frames")
        energies = []
        for frame in frames:
            atoms = frame.copy()
            atoms.calc = self.calculator
            energies.append(float(atoms.get_potential_energy()))
        energy = np.asarray(energies, dtype=float)
        relative = energy - float(np.min(energy))
        if coordinate is None:
            resolved_coordinate = _path_coordinate(frames)
        else:
            resolved_coordinate = np.asarray(coordinate, dtype=float).reshape(-1)
            if resolved_coordinate.shape != energy.shape:
                raise ValueError("coordinate must contain one value per frame")
        minimum = int(np.argmin(energy))
        maximum = int(np.argmax(energy))
        return EnergyProfile(
            energies_eV=energy,
            relative_energies_eV=relative,
            coordinate=resolved_coordinate,
            barrier_eV=float(energy[maximum] - energy[0]),
            minimum_index=minimum,
            maximum_index=maximum,
        )

    def conformer_energy_difference(self, first: Any, second: Any) -> float:
        profile = self.energy_profile((first, second), coordinate=(0.0, 1.0))
        return float(profile.energies_eV[1] - profile.energies_eV[0])

    def dipole_probe(
        self,
        frames: Sequence[Any],
        *,
        temperature_K: float,
        block_count: int = 5,
    ) -> DipoleProbeResult:
        if len(frames) < 2:
            raise ValueError("a dipole probe requires at least two frames")
        if "dipole" not in getattr(self.calculator, "implemented_properties", ()):
            raise ValueError(
                "the potential does not expose dipoles; train the electrostatic preset "
                "with charge/dipole labels or use a validated electronic-structure source"
            )
        dipoles = []
        volumes = []
        for frame in frames:
            atoms = frame.copy()
            atoms.calc = self.calculator
            value = atoms.calc.get_property("dipole", atoms)
            dipoles.append(np.asarray(value, dtype=float).reshape(-1, 3)[0])
            volumes.append(float(atoms.get_volume()))
        dipole_array = np.asarray(dipoles, dtype=float)
        volume_array = np.asarray(volumes, dtype=float)
        estimate = dielectric_from_dipole_fluctuations(
            dipole_array,
            temperature_K=temperature_K,
            volume_A3=volume_array,
            unit="eA",
            block_count=block_count,
        )
        warning = estimate.warning
        if any(not bool(np.all(frame.pbc)) for frame in frames):
            warning = (
                "Static bulk dielectric estimates require a fully periodic equilibrated "
                "cell; at least one supplied frame is not fully periodic."
            )
        return DipoleProbeResult(
            dipoles_eA=dipole_array,
            volumes_A3=volume_array,
            dielectric=estimate,
            frame_count=len(frames),
            warning=warning,
        )


def _path_coordinate(frames: Sequence[Any]) -> np.ndarray:
    result = [0.0]
    for previous, current in zip(frames[:-1], frames[1:], strict=True):
        left = np.asarray(previous.get_positions(), dtype=float)
        right = np.asarray(current.get_positions(), dtype=float)
        if left.shape != right.shape:
            raise ValueError("all path frames must contain the same atom ordering")
        rms = float(np.sqrt(np.mean(np.sum((right - left) ** 2, axis=1))))
        result.append(result[-1] + rms)
    return np.asarray(result, dtype=float)


__all__ = [
    "DipoleProbeResult",
    "EnergyProfile",
    "PolymerMechanismSimulator",
]
