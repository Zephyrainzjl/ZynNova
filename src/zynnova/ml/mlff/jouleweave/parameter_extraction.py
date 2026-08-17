"""Transport and electrochemical-reaction parameter extraction from atomistic data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


BOLTZMANN_J_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19
FARADAY_C_MOL = 96485.33212
GAS_CONSTANT_J_MOL_K = 8.314462618


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    name: str
    value: float
    unit: str
    standard_uncertainty: float
    soc: float
    temperature_K: float
    electrode_potential_V: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or not np.isfinite(self.standard_uncertainty):
            raise ValueError("parameter estimate must be finite")
        if self.standard_uncertainty < 0.0:
            raise ValueError("parameter uncertainty cannot be negative")
        if not 0.0 <= self.soc <= 1.0 or self.temperature_K <= 0.0:
            raise ValueError("parameter SOC/temperature is invalid")


@dataclass(slots=True)
class AtomisticParameterSurface:
    """Uncertainty-aware inverse-distance interpolation over SOC and temperature."""

    estimates: Sequence[ParameterEstimate]
    soc_scale: float = 0.1
    temperature_scale_K: float = 20.0
    potential_scale_V: float = 0.2

    def __post_init__(self) -> None:
        self.estimates = tuple(self.estimates)
        if not self.estimates:
            raise ValueError("parameter surface requires at least one estimate")
        names = {estimate.name for estimate in self.estimates}
        units = {estimate.unit for estimate in self.estimates}
        if len(names) != 1 or len(units) != 1:
            raise ValueError("one parameter surface can contain only one name and unit")
        if min(self.soc_scale, self.temperature_scale_K, self.potential_scale_V) <= 0.0:
            raise ValueError("parameter-surface scales must be positive")

    @property
    def name(self) -> str:
        return self.estimates[0].name

    @property
    def unit(self) -> str:
        return self.estimates[0].unit

    def evaluate(
        self,
        soc: float,
        temperature_K: float,
        electrode_potential_V: float | None = None,
    ) -> ParameterEstimate:
        if not 0.0 <= soc <= 1.0 or temperature_K <= 0.0:
            raise ValueError("query SOC/temperature is invalid")
        distances: list[float] = []
        for estimate in self.estimates:
            distance = ((soc - estimate.soc) / self.soc_scale) ** 2
            distance += ((temperature_K - estimate.temperature_K) / self.temperature_scale_K) ** 2
            if electrode_potential_V is not None and estimate.electrode_potential_V is not None:
                distance += (
                    (electrode_potential_V - estimate.electrode_potential_V)
                    / self.potential_scale_V
                ) ** 2
            distances.append(float(np.sqrt(distance)))
        distance_array = np.asarray(distances)
        exact = np.flatnonzero(distance_array <= 1.0e-12)
        if len(exact):
            return self.estimates[int(exact[0])]
        weights = 1.0 / np.maximum(distance_array, 1.0e-12) ** 2
        weights /= np.sum(weights)
        values = np.asarray([estimate.value for estimate in self.estimates])
        sigmas = np.asarray(
            [estimate.standard_uncertainty for estimate in self.estimates]
        )
        mean = float(weights @ values)
        variance = float(weights @ (sigmas**2 + (values - mean) ** 2))
        return ParameterEstimate(
            name=self.name,
            value=mean,
            unit=self.unit,
            standard_uncertainty=float(np.sqrt(max(variance, 0.0))),
            soc=float(soc),
            temperature_K=float(temperature_K),
            electrode_potential_V=electrode_potential_V,
            metadata={"interpolation": "uncertainty-weighted inverse distance"},
        )



@dataclass(frozen=True, slots=True)
class AtomisticExtractionInput:
    """Standardized transport/reaction observables from MD and rare-event runs."""

    time_s: np.ndarray
    msd_m2: Mapping[str, np.ndarray]
    concentrations_mol_m3: Mapping[str, float]
    charge_numbers: Mapping[str, float]
    reaction_barriers_eV: Mapping[str, float] = field(default_factory=dict)
    barrier_uncertainty_eV: Mapping[str, float] = field(default_factory=dict)
    reaction_concentrations_mol_m3: Mapping[str, tuple[float, float]] = field(
        default_factory=dict
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float).reshape(-1)
        if len(time) < 6 or np.any(np.diff(time) <= 0.0):
            raise ValueError("atomistic extraction time grid is invalid")
        object.__setattr__(self, "time_s", time)
        species = set(self.msd_m2)
        if not species:
            raise ValueError("at least one species MSD is required")
        if species - set(self.concentrations_mol_m3):
            raise ValueError("every MSD species requires a concentration")
        if species - set(self.charge_numbers):
            raise ValueError("every MSD species requires a charge number")
        for name, values in self.msd_m2.items():
            array = np.asarray(values, dtype=float).reshape(-1)
            if array.shape != time.shape or not np.isfinite(array).all():
                raise ValueError(f"MSD for {name!r} is not aligned and finite")
        if any(float(value) < 0.0 for value in self.concentrations_mol_m3.values()):
            raise ValueError("species concentrations cannot be negative")
        if any(float(value) < 0.0 for value in self.reaction_barriers_eV.values()):
            raise ValueError("reaction barriers cannot be negative")
        if any(float(value) < 0.0 for value in self.barrier_uncertainty_eV.values()):
            raise ValueError("barrier uncertainties cannot be negative")


@dataclass(slots=True)
class AutomaticParameterExtractor:
    """Convert one atomistic campaign point into continuum-ready parameters."""

    diffusivity_names: Mapping[str, str]
    conductivity_name: str = "electrolyte_conductivity"
    transference_name: str = "electrolyte_transference_number"
    cation_species: str = "Li+"
    reaction_rate_names: Mapping[str, str] = field(default_factory=dict)
    exchange_current_names: Mapping[str, str] = field(default_factory=dict)
    dimensionality: int = 3
    fit_window: tuple[float, float] = (0.4, 0.9)
    blocks: int = 5
    correlation_factor: float = 1.0
    attempt_frequency_Hz: float = 1.0e13
    transfer_coefficient: float = 0.5
    electron_number: int = 1

    def extract(
        self,
        data: AtomisticExtractionInput,
        *,
        soc: float,
        temperature_K: float,
        electrode_potential_V: float | None = None,
        overpotential_V: float = 0.0,
    ) -> tuple[ParameterEstimate, ...]:
        if not 0.0 <= soc <= 1.0 or temperature_K <= 0.0:
            raise ValueError("extraction SOC/temperature is invalid")
        diffusion: dict[str, float] = {}
        diffusion_sigma: dict[str, float] = {}
        estimates: list[ParameterEstimate] = []
        for species, curve in data.msd_m2.items():
            value, sigma = diffusion_from_msd(
                data.time_s,
                curve,
                dimensionality=self.dimensionality,
                fit_window=self.fit_window,
                blocks=self.blocks,
            )
            diffusion[species] = value
            diffusion_sigma[species] = sigma
            name = self.diffusivity_names.get(species)
            if name is not None:
                estimates.append(
                    ParameterEstimate(
                        name=name,
                        value=value,
                        unit="m2 s-1",
                        standard_uncertainty=sigma,
                        soc=float(soc),
                        temperature_K=float(temperature_K),
                        electrode_potential_V=electrode_potential_V,
                        metadata={
                            **dict(data.metadata),
                            "species": species,
                            "method": "Einstein MSD slope with block uncertainty",
                        },
                    )
                )

        ordered_species = tuple(data.msd_m2)
        concentrations = [data.concentrations_mol_m3[name] for name in ordered_species]
        diffusivities = [diffusion[name] for name in ordered_species]
        charges = [data.charge_numbers[name] for name in ordered_species]
        conductivity = nernst_einstein_conductivity(
            concentrations,
            diffusivities,
            charges,
            temperature_K,
            correlation_factor=self.correlation_factor,
        )
        coefficients = (
            self.correlation_factor
            * FARADAY_C_MOL**2
            * np.asarray(concentrations, dtype=float)
            * np.asarray(charges, dtype=float) ** 2
            / (GAS_CONSTANT_J_MOL_K * temperature_K)
        )
        conductivity_sigma = float(
            np.sqrt(
                np.sum(
                    (
                        coefficients
                        * np.asarray(
                            [diffusion_sigma[name] for name in ordered_species],
                            dtype=float,
                        )
                    )
                    ** 2
                )
            )
        )
        estimates.append(
            ParameterEstimate(
                self.conductivity_name,
                conductivity,
                "S m-1",
                conductivity_sigma,
                float(soc),
                float(temperature_K),
                electrode_potential_V,
                {**dict(data.metadata), "method": "Nernst-Einstein"},
            )
        )

        if self.cation_species in diffusion:
            transference = cation_transference_number(
                data.concentrations_mol_m3[self.cation_species],
                diffusion[self.cation_species],
                data.charge_numbers[self.cation_species],
                concentrations,
                diffusivities,
                charges,
            )
            transference_sigma = _transference_uncertainty(
                ordered_species,
                data,
                diffusion,
                diffusion_sigma,
                self.cation_species,
            )
            estimates.append(
                ParameterEstimate(
                    self.transference_name,
                    transference,
                    "1",
                    transference_sigma,
                    float(soc),
                    float(temperature_K),
                    electrode_potential_V,
                    {**dict(data.metadata), "method": "Nernst-Einstein mobility ratio"},
                )
            )

        for reaction, barrier in data.reaction_barriers_eV.items():
            rate = reaction_rate_from_barrier(
                float(barrier),
                temperature_K,
                attempt_frequency_Hz=self.attempt_frequency_Hz,
                transfer_coefficient=self.transfer_coefficient,
                overpotential_V=overpotential_V,
                electron_number=self.electron_number,
            )
            barrier_sigma = float(data.barrier_uncertainty_eV.get(reaction, 0.0))
            sensitivity = (
                ELEMENTARY_CHARGE_C / (BOLTZMANN_J_K * temperature_K)
            )
            rate_sigma = abs(rate * sensitivity * barrier_sigma)
            rate_name = self.reaction_rate_names.get(reaction)
            if rate_name is not None:
                estimates.append(
                    ParameterEstimate(
                        rate_name,
                        rate,
                        "s-1",
                        rate_sigma,
                        float(soc),
                        float(temperature_K),
                        electrode_potential_V,
                        {
                            **dict(data.metadata),
                            "reaction": reaction,
                            "barrier_eV": float(barrier),
                            "method": "transition-state theory",
                        },
                    )
                )
            exchange_name = self.exchange_current_names.get(reaction)
            concentrations_pair = data.reaction_concentrations_mol_m3.get(reaction)
            if exchange_name is not None and concentrations_pair is not None:
                # A length scale supplied by the caller can convert the event
                # frequency to an effective interfacial rate constant.
                length_m = float(data.metadata.get("reaction_length_m", 1.0e-10))
                rate_constant = rate * length_m
                exchange = exchange_current_density(
                    rate_constant,
                    concentrations_pair[0],
                    concentrations_pair[1],
                    transfer_coefficient=self.transfer_coefficient,
                    electron_number=self.electron_number,
                )
                exchange_sigma = (
                    0.0 if rate <= 0.0 else abs(exchange * rate_sigma / rate)
                )
                estimates.append(
                    ParameterEstimate(
                        exchange_name,
                        exchange,
                        "A m-2",
                        exchange_sigma,
                        float(soc),
                        float(temperature_K),
                        electrode_potential_V,
                        {
                            **dict(data.metadata),
                            "reaction": reaction,
                            "method": "TST rate plus Butler-Volmer concentration factor",
                        },
                    )
                )
        return tuple(estimates)


def _transference_uncertainty(
    species: Sequence[str],
    data: AtomisticExtractionInput,
    diffusion: Mapping[str, float],
    diffusion_sigma: Mapping[str, float],
    cation_species: str,
) -> float:
    mobility = np.asarray(
        [
            data.concentrations_mol_m3[name]
            * data.charge_numbers[name] ** 2
            * diffusion[name]
            for name in species
        ],
        dtype=float,
    )
    mobility_sigma = np.asarray(
        [
            data.concentrations_mol_m3[name]
            * data.charge_numbers[name] ** 2
            * diffusion_sigma[name]
            for name in species
        ],
        dtype=float,
    )
    denominator = float(np.sum(mobility))
    if denominator <= 0.0:
        return 0.0
    cation_index = species.index(cation_species)
    numerator = mobility[cation_index]
    gradient = np.full(len(species), -numerator / denominator**2)
    gradient[cation_index] += 1.0 / denominator
    return float(np.sqrt(np.sum((gradient * mobility_sigma) ** 2)))

def diffusion_from_msd(
    time_s: np.ndarray,
    msd_m2: np.ndarray,
    *,
    dimensionality: int = 3,
    fit_window: tuple[float, float] = (0.4, 0.9),
    blocks: int = 5,
) -> tuple[float, float]:
    """Estimate tracer diffusivity and block uncertainty from an MSD curve."""

    time = np.asarray(time_s, dtype=float).reshape(-1)
    msd = np.asarray(msd_m2, dtype=float).reshape(-1)
    if time.shape != msd.shape or len(time) < 6 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time/MSD arrays are invalid")
    if dimensionality < 1 or not 0.0 <= fit_window[0] < fit_window[1] <= 1.0:
        raise ValueError("invalid diffusivity fit controls")
    start = int(np.floor(fit_window[0] * (len(time) - 1)))
    stop = int(np.ceil(fit_window[1] * len(time)))
    x = time[start:stop]
    y = msd[start:stop]
    slope = float(np.polyfit(x, y, 1)[0])
    diffusivity = max(slope / (2.0 * dimensionality), 0.0)
    chunks = [chunk for chunk in np.array_split(np.arange(len(x)), blocks) if len(chunk) >= 2]
    block_values = [
        max(float(np.polyfit(x[chunk], y[chunk], 1)[0]) / (2.0 * dimensionality), 0.0)
        for chunk in chunks
    ]
    uncertainty = float(np.std(block_values, ddof=1)) if len(block_values) > 1 else 0.0
    return diffusivity, uncertainty


def nernst_einstein_conductivity(
    concentrations_mol_m3: Sequence[float],
    diffusivities_m2_s: Sequence[float],
    charge_numbers: Sequence[float],
    temperature_K: float,
    *,
    correlation_factor: float = 1.0,
) -> float:
    concentration = np.asarray(concentrations_mol_m3, dtype=float)
    diffusion = np.asarray(diffusivities_m2_s, dtype=float)
    charge = np.asarray(charge_numbers, dtype=float)
    if not (concentration.shape == diffusion.shape == charge.shape) or temperature_K <= 0.0:
        raise ValueError("Nernst-Einstein inputs are invalid")
    if np.any(concentration < 0.0) or np.any(diffusion < 0.0) or correlation_factor < 0.0:
        raise ValueError("transport inputs cannot be negative")
    return float(
        correlation_factor
        * FARADAY_C_MOL**2
        * np.sum(concentration * charge**2 * diffusion)
        / (GAS_CONSTANT_J_MOL_K * temperature_K)
    )


def cation_transference_number(
    cation_concentration_mol_m3: float,
    cation_diffusivity_m2_s: float,
    cation_charge: float,
    species_concentrations_mol_m3: Sequence[float],
    species_diffusivities_m2_s: Sequence[float],
    species_charges: Sequence[float],
) -> float:
    numerator = (
        cation_concentration_mol_m3 * cation_diffusivity_m2_s * cation_charge**2
    )
    denominator = float(
        np.sum(
            np.asarray(species_concentrations_mol_m3)
            * np.asarray(species_diffusivities_m2_s)
            * np.asarray(species_charges) ** 2
        )
    )
    if denominator <= 0.0:
        raise ValueError("transference-number denominator must be positive")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def arrhenius_fit(
    temperature_K: Sequence[float],
    values: Sequence[float],
) -> tuple[float, float, float]:
    """Return prefactor, activation energy [J/mol], and log-space residual sigma."""

    temperature = np.asarray(temperature_K, dtype=float)
    y = np.asarray(values, dtype=float)
    if temperature.shape != y.shape or len(y) < 2 or np.any(temperature <= 0.0) or np.any(y <= 0.0):
        raise ValueError("Arrhenius samples must be positive and aligned")
    x = 1.0 / temperature
    slope, intercept = np.polyfit(x, np.log(y), 1)
    prediction = intercept + slope * x
    sigma = float(np.sqrt(np.mean((np.log(y) - prediction) ** 2)))
    return float(np.exp(intercept)), float(-slope * GAS_CONSTANT_J_MOL_K), sigma


def reaction_rate_from_barrier(
    barrier_eV: float,
    temperature_K: float,
    *,
    attempt_frequency_Hz: float = 1.0e13,
    transfer_coefficient: float = 0.5,
    overpotential_V: float = 0.0,
    electron_number: int = 1,
) -> float:
    if barrier_eV < 0.0 or temperature_K <= 0.0 or attempt_frequency_Hz <= 0.0:
        raise ValueError("reaction-rate inputs are invalid")
    effective_barrier_eV = barrier_eV - transfer_coefficient * electron_number * overpotential_V
    exponent = -effective_barrier_eV * ELEMENTARY_CHARGE_C / (BOLTZMANN_J_K * temperature_K)
    return float(attempt_frequency_Hz * np.exp(np.clip(exponent, -700.0, 100.0)))


def exchange_current_density(
    rate_constant_m_s: float,
    concentration_oxidized_mol_m3: float,
    concentration_reduced_mol_m3: float,
    *,
    transfer_coefficient: float = 0.5,
    electron_number: int = 1,
) -> float:
    if rate_constant_m_s < 0.0 or min(
        concentration_oxidized_mol_m3, concentration_reduced_mol_m3
    ) < 0.0:
        raise ValueError("exchange-current inputs cannot be negative")
    alpha = float(transfer_coefficient)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("transfer coefficient must lie in [0,1]")
    return float(
        electron_number
        * FARADAY_C_MOL
        * rate_constant_m_s
        * concentration_oxidized_mol_m3 ** (1.0 - alpha)
        * concentration_reduced_mol_m3**alpha
    )


__all__ = [
    "AtomisticExtractionInput",
    "AtomisticParameterSurface",
    "AutomaticParameterExtractor",
    "ParameterEstimate",
    "arrhenius_fit",
    "cation_transference_number",
    "diffusion_from_msd",
    "exchange_current_density",
    "nernst_einstein_conductivity",
    "reaction_rate_from_barrier",
]
