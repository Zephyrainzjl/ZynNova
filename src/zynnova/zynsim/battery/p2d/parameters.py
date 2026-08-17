"""Validated Doyle–Fuller–Newman/P2D parameter objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ...constants import FARADAY
from ..properties import ArrheniusLaw, PropertyValue, evaluate_property


OCPFunction = Callable[[np.ndarray | float, float], np.ndarray | float]


def graphite_ocp(stoichiometry: np.ndarray | float, temperature_K: float) -> np.ndarray:
    """Smooth graphite open-circuit potential fit in volts vs Li/Li+."""

    del temperature_K
    x = np.clip(np.asarray(stoichiometry, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return (
        0.194
        + 1.5 * np.exp(-120.0 * x)
        + 0.0351 * np.tanh((x - 0.286) / 0.083)
        - 0.0045 * np.tanh((x - 0.849) / 0.119)
        - 0.0350 * np.tanh((x - 0.9233) / 0.050)
        - 0.0147 * np.tanh((x - 0.500) / 0.034)
        - 0.1020 * np.tanh((x - 0.194) / 0.142)
        - 0.0220 * np.tanh((x - 0.900) / 0.0164)
        - 0.0110 * np.tanh((x - 0.124) / 0.0226)
        + 0.0155 * np.tanh((x - 0.105) / 0.029)
    )


def nmc811_ocp(stoichiometry: np.ndarray | float, temperature_K: float) -> np.ndarray:
    """Smooth NMC811 open-circuit potential fit in volts vs Li/Li+."""

    del temperature_K
    x = np.clip(np.asarray(stoichiometry, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return (
        -0.8090 * x
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (x - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (x - 0.3120))
    )


def zero_entropic_coefficient(
    stoichiometry: np.ndarray | float, temperature_K: float
) -> np.ndarray:
    del temperature_K
    return np.zeros_like(np.asarray(stoichiometry, dtype=float))


@dataclass(slots=True)
class ElectrodeParameters:
    name: str
    thickness_m: float
    porosity: float
    active_volume_fraction: float
    bruggeman: float
    particle_radius_m: float
    maximum_concentration_mol_m3: float
    solid_diffusivity_m2_s: PropertyValue
    electronic_conductivity_S_m: PropertyValue
    reaction_rate_m2p5_mol_m0p5_s: PropertyValue
    stoichiometry_at_soc0: float
    stoichiometry_at_soc1: float
    ocp_V: OCPFunction
    entropic_coefficient_V_K: OCPFunction = zero_entropic_coefficient
    charge_transfer_coefficient: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            self.thickness_m,
            self.particle_radius_m,
            self.maximum_concentration_mol_m3,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError(f"{self.name}: geometric and concentration values must be positive")
        if not 0.0 < self.porosity < 1.0:
            raise ValueError(f"{self.name}: porosity must lie in (0, 1)")
        if not 0.0 < self.active_volume_fraction < 1.0 - self.porosity + 1.0e-12:
            raise ValueError(f"{self.name}: active volume fraction is inconsistent")
        if self.bruggeman <= 0.0:
            raise ValueError(f"{self.name}: Bruggeman exponent must be positive")
        if not 0.0 < self.charge_transfer_coefficient < 1.0:
            raise ValueError(f"{self.name}: charge-transfer coefficient must lie in (0, 1)")
        for value in (self.stoichiometry_at_soc0, self.stoichiometry_at_soc1):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{self.name}: stoichiometric endpoints must lie in (0, 1)")

    @property
    def specific_surface_area_m2_m3(self) -> float:
        return 3.0 * self.active_volume_fraction / self.particle_radius_m

    def stoichiometry(self, soc: float) -> float:
        return float(
            self.stoichiometry_at_soc0
            + np.clip(soc, 0.0, 1.0)
            * (self.stoichiometry_at_soc1 - self.stoichiometry_at_soc0)
        )

    def diffusivity(self, soc: float, temperature_K: float) -> float:
        return evaluate_property(
            self.solid_diffusivity_m2_s,
            soc,
            temperature_K,
            name=f"{self.name}.solid_diffusivity_m2_s",
        )

    def conductivity(self, soc: float, temperature_K: float) -> float:
        return evaluate_property(
            self.electronic_conductivity_S_m,
            soc,
            temperature_K,
            name=f"{self.name}.electronic_conductivity_S_m",
        )

    def reaction_rate(self, soc: float, temperature_K: float) -> float:
        return evaluate_property(
            self.reaction_rate_m2p5_mol_m0p5_s,
            soc,
            temperature_K,
            name=f"{self.name}.reaction_rate",
        )


@dataclass(slots=True)
class SeparatorParameters:
    thickness_m: float
    porosity: float
    bruggeman: float = 1.5

    def __post_init__(self) -> None:
        if self.thickness_m <= 0.0 or not 0.0 < self.porosity < 1.0:
            raise ValueError("separator geometry is invalid")
        if self.bruggeman <= 0.0:
            raise ValueError("separator Bruggeman exponent must be positive")


@dataclass(slots=True)
class ElectrolyteParameters:
    initial_concentration_mol_m3: float = 1000.0
    diffusivity_m2_s: PropertyValue = 1.7694e-10
    ionic_conductivity_S_m: PropertyValue = 0.9487
    transference_number: float = 0.2594
    thermodynamic_factor: PropertyValue = 1.0

    def __post_init__(self) -> None:
        if self.initial_concentration_mol_m3 <= 0.0:
            raise ValueError("initial electrolyte concentration must be positive")
        if not 0.0 < self.transference_number < 1.0:
            raise ValueError("transference number must lie in (0, 1)")


@dataclass(slots=True)
class ThermalParameters:
    density_heat_capacity_J_m3_K: float = 2.5e6
    heat_transfer_coefficient_W_m2_K: float = 8.0
    cooling_area_m2: float = 0.2
    ambient_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if any(
            value < 0.0
            for value in (
                self.density_heat_capacity_J_m3_K,
                self.heat_transfer_coefficient_W_m2_K,
                self.cooling_area_m2,
            )
        ):
            raise ValueError("thermal parameters cannot be negative")


@dataclass(slots=True)
class P2DDiscretization:
    negative_cells: int = 12
    separator_cells: int = 8
    positive_cells: int = 12
    negative_particle_cells: int = 16
    positive_particle_cells: int = 16

    def __post_init__(self) -> None:
        if min(
            self.negative_cells,
            self.separator_cells,
            self.positive_cells,
            self.negative_particle_cells,
            self.positive_particle_cells,
        ) < 2:
            raise ValueError("every P2D discretization count must be at least two")


@dataclass(slots=True)
class P2DParameters:
    negative: ElectrodeParameters
    separator: SeparatorParameters
    positive: ElectrodeParameters
    electrolyte: ElectrolyteParameters = field(default_factory=ElectrolyteParameters)
    thermal: ThermalParameters = field(default_factory=ThermalParameters)
    discretization: P2DDiscretization = field(default_factory=P2DDiscretization)
    area_m2: float = 0.1027
    initial_temperature_K: float = 298.15
    minimum_concentration_mol_m3: float = 1.0e-9
    nonlinear_tolerance: float = 1.0e-8
    nonlinear_max_evaluations: int = 300
    coupling_tolerance: float = 1.0e-7
    coupling_max_iterations: int = 12
    coupling_relaxation: float = 0.7

    def __post_init__(self) -> None:
        if self.area_m2 <= 0.0 or self.initial_temperature_K <= 0.0:
            raise ValueError("cell area and temperature must be positive")
        if self.minimum_concentration_mol_m3 <= 0.0:
            raise ValueError("minimum concentration must be positive")
        if self.nonlinear_tolerance <= 0.0 or self.coupling_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.nonlinear_max_evaluations < 1 or self.coupling_max_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if not 0.0 < self.coupling_relaxation <= 1.0:
            raise ValueError("coupling relaxation must lie in (0, 1]")

    @property
    def thickness_m(self) -> float:
        return (
            self.negative.thickness_m
            + self.separator.thickness_m
            + self.positive.thickness_m
        )

    def theoretical_capacity_Ah(self) -> float:
        negative = (
            self.area_m2
            * self.negative.thickness_m
            * self.negative.active_volume_fraction
            * self.negative.maximum_concentration_mol_m3
            * abs(
                self.negative.stoichiometry_at_soc1
                - self.negative.stoichiometry_at_soc0
            )
            * FARADAY
            / 3600.0
        )
        positive = (
            self.area_m2
            * self.positive.thickness_m
            * self.positive.active_volume_fraction
            * self.positive.maximum_concentration_mol_m3
            * abs(
                self.positive.stoichiometry_at_soc1
                - self.positive.stoichiometry_at_soc0
            )
            * FARADAY
            / 3600.0
        )
        return float(min(negative, positive))


def reference_graphite_nmc811_parameters(
    *,
    area_m2: float = 0.1027,
    discretization: P2DDiscretization | None = None,
) -> P2DParameters:
    """Return a literature-scale graphite/NMC811 validation parameterization.

    This is a reproducible starting point, not a universal cell definition.
    Geometry, OCP, kinetic, thermal, and transport data must be re-identified
    for quantitative prediction of a particular manufactured cell.
    """

    negative = ElectrodeParameters(
        name="graphite",
        thickness_m=85.2e-6,
        porosity=0.25,
        active_volume_fraction=0.75,
        bruggeman=1.5,
        particle_radius_m=5.86e-6,
        maximum_concentration_mol_m3=33133.0,
        solid_diffusivity_m2_s=ArrheniusLaw(3.3e-14, 30300.0),
        electronic_conductivity_S_m=215.0,
        reaction_rate_m2p5_mol_m0p5_s=ArrheniusLaw(1.764e-11, 35000.0),
        stoichiometry_at_soc0=0.0263,
        stoichiometry_at_soc1=0.9100,
        ocp_V=graphite_ocp,
    )
    positive = ElectrodeParameters(
        name="NMC811",
        thickness_m=75.6e-6,
        porosity=0.335,
        active_volume_fraction=0.665,
        bruggeman=1.5,
        particle_radius_m=5.22e-6,
        maximum_concentration_mol_m3=63104.0,
        solid_diffusivity_m2_s=ArrheniusLaw(4.0e-15, 25000.0),
        electronic_conductivity_S_m=0.18,
        reaction_rate_m2p5_mol_m0p5_s=ArrheniusLaw(6.667e-11, 17800.0),
        stoichiometry_at_soc0=0.9360,
        stoichiometry_at_soc1=0.2630,
        ocp_V=nmc811_ocp,
    )
    return P2DParameters(
        negative=negative,
        separator=SeparatorParameters(thickness_m=12.0e-6, porosity=0.47),
        positive=positive,
        area_m2=area_m2,
        discretization=discretization or P2DDiscretization(),
    )


__all__ = [
    "ElectrodeParameters",
    "ElectrolyteParameters",
    "OCPFunction",
    "P2DDiscretization",
    "P2DParameters",
    "SeparatorParameters",
    "ThermalParameters",
    "graphite_ocp",
    "nmc811_ocp",
    "reference_graphite_nmc811_parameters",
    "zero_entropic_coefficient",
]
