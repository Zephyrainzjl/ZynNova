"""Phase-field crystal, Swift--Hohenberg, Ohta--Kawasaki, and MBE models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fields import FieldSpec
from .base import PhaseFieldModel


@dataclass(slots=True)
class PhaseFieldCrystalModel(PhaseFieldModel):
    """Conserved phase-field crystal model on diffusive time scales."""

    mobility: float = 1.0
    undercooling: float = -0.25
    preferred_wave_number: float = 1.0
    quartic: float = 1.0
    field_name: str = "psi"
    name: str = "phase-field-crystal"

    @property
    def field_specs(self):
        return (FieldSpec(self.field_name, conserved=True),)

    def chemical_potential(self, psi, operators):
        q2 = self.preferred_wave_number**2
        lap = operators.laplacian(psi)
        squared_operator = operators.laplacian(lap + 2.0 * q2 * psi) + q2**2 * psi
        return self.undercooling * psi + squared_operator + self.quartic * psi**3

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        psi = fields[self.field_name]
        return {self.field_name: self.mobility * operators.laplacian(self.chemical_potential(psi, operators))}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        q2 = self.preferred_wave_number**2
        return -self.mobility * k2 * (
            self.undercooling + (q2 - k2) ** 2
        )

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        psi = fields[self.field_name]
        q2 = self.preferred_wave_number**2
        shifted = operators.laplacian(psi) + q2 * psi
        return 0.5 * self.undercooling * psi**2 + 0.5 * shifted**2 + 0.25 * self.quartic * psi**4


@dataclass(slots=True)
class SwiftHohenbergModel(PhaseFieldModel):
    """Non-conserved Swift--Hohenberg pattern-forming equation."""

    growth_rate: float = 0.3
    preferred_wave_number: float = 1.0
    cubic: float = 1.0
    mobility: float = 1.0
    field_name: str = "u"
    name: str = "swift-hohenberg"

    @property
    def field_specs(self):
        return (FieldSpec(self.field_name, conserved=False),)

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        u = fields[self.field_name]
        q2 = self.preferred_wave_number**2
        shifted = operators.laplacian(u) + q2 * u
        return {self.field_name: self.mobility * (self.growth_rate * u - operators.laplacian(shifted) - q2 * shifted - self.cubic * u**3)}

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        q2 = self.preferred_wave_number**2
        return self.mobility * (self.growth_rate - (q2 - k2) ** 2)

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        u = fields[self.field_name]
        shifted = operators.laplacian(u) + self.preferred_wave_number**2 * u
        return -0.5 * self.growth_rate * u**2 + 0.5 * shifted**2 + 0.25 * self.cubic * u**4


@dataclass(slots=True)
class OhtaKawasakiModel(PhaseFieldModel):
    """Ohta--Kawasaki model for diblock-copolymer microphase separation."""

    mobility: float = 1.0
    quadratic: float = -1.0
    quartic: float = 1.0
    gradient_coefficient: float = 1.0
    long_range_strength: float = 0.1
    mean_composition: float = 0.0
    field_name: str = "phi"
    name: str = "ohta-kawasaki"

    @property
    def field_specs(self):
        return (FieldSpec(self.field_name, conserved=True),)

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        phi = fields[self.field_name]
        chemical_potential = (
            self.quadratic * phi
            + self.quartic * phi**3
            - self.gradient_coefficient * operators.laplacian(phi)
        )
        # Applying Laplacian to the non-local inverse-Laplacian contribution yields
        # -alpha(phi-mean), avoiding a separate Poisson solve in the evolution step.
        return {
            self.field_name: self.mobility * operators.laplacian(chemical_potential)
            - self.mobility * self.long_range_strength * (phi - self.mean_composition)
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        return -self.mobility * (
            k2 * (self.quadratic + self.gradient_coefficient * k2)
            + self.long_range_strength
        )

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        phi = fields[self.field_name]
        local = (
            0.5 * self.quadratic * phi**2
            + 0.25 * self.quartic * phi**4
            + 0.5 * self.gradient_coefficient * operators.grad_squared(phi)
        )
        # Periodic spectral inverse Laplacian for the nonlocal term.
        if hasattr(operators, "k2"):
            transformed = operators._fft(phi - self.mean_composition)
            k2 = operators.k2
            if hasattr(k2, "to"):
                k2 = k2.to(device=phi.device, dtype=phi.real.dtype)
            denominator = k2 + (k2 == 0) * 1.0
            potential = operators._ifft(transformed / denominator)
        else:
            from ..operators import numpy_wave_numbers

            k2, _ = numpy_wave_numbers(operators.grid)
            transformed = np.fft.fftn(np.asarray(phi) - self.mean_composition)
            denominator = np.where(k2 == 0.0, 1.0, k2)
            potential = np.fft.ifftn(transformed / denominator).real
        return local + 0.5 * self.long_range_strength * (phi - self.mean_composition) * potential


@dataclass(slots=True)
class MolecularBeamEpitaxyModel(PhaseFieldModel):
    """Fourth-order molecular-beam epitaxy surface-growth model."""

    mobility: float = 1.0
    surface_diffusion: float = 1.0
    slope_strength: float = 1.0
    deposition_rate: float = 0.0
    field_name: str = "height"
    name: str = "molecular-beam-epitaxy"
    gradient_flow: bool = False

    @property
    def field_specs(self):
        return (FieldSpec(self.field_name, conserved=True),)

    def rhs(self, fields, grid, operators, *, xp=np, time=0.0):
        del grid, time
        height = fields[self.field_name]
        gradient = operators.gradient(height)
        norm_squared = 0.0
        for component in gradient:
            norm_squared = norm_squared + component**2
        nonlinear_flux = tuple((1.0 - norm_squared) * component for component in gradient)
        return {
            self.field_name: -self.mobility * self.surface_diffusion * operators.biharmonic(height)
            - self.mobility * self.slope_strength * operators.divergence(nonlinear_flux)
            + self.deposition_rate
        }

    def linear_symbol(self, field_name, grid, k2, *, xp=np):
        del grid, xp
        return -self.mobility * self.surface_diffusion * k2**2

    def free_energy_density(self, fields, grid, operators, *, xp=np):
        del grid, xp
        height = fields[self.field_name]
        grad2 = operators.grad_squared(height)
        return 0.5 * self.surface_diffusion * operators.laplacian(height) ** 2 + 0.25 * self.slope_strength * (1.0 - grad2) ** 2


__all__ = [
    "MolecularBeamEpitaxyModel",
    "OhtaKawasakiModel",
    "PhaseFieldCrystalModel",
    "SwiftHohenbergModel",
]
