"""Unified production polar density and audited alternative electronic paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .config import ElectronicConfig
from .electrostatics import electrostatic_potential_features
from .multipoles import SpinMultipoleState, coefficients_to_cartesian, multipole_slice
from .oxidation import OxidationPrediction, OxidationStateHead
from .polar import (
    PolarDensityModel,
    constrain_charge_monopoles,
    graph_count,
    scatter_sum,
)
from .qeq import QEqResult, solve_qeq

torch = require_torch()
nn = torch.nn
functional = torch.nn.functional


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> Any:
    layers: list[Any] = []
    current = input_dim
    for width in hidden:
        layers.extend((nn.Linear(current, width), nn.SiLU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _zero_last(module: Any) -> None:
    for item in reversed(list(module.modules())):
        if isinstance(item, nn.Linear):
            nn.init.zeros_(item.weight)
            if item.bias is not None:
                nn.init.zeros_(item.bias)
            return


@dataclass(slots=True)
class ElectronicPrediction:
    state: SpinMultipoleState
    magmoms: Any
    collinear_spins: Any
    energy: Any
    atomic_energy: Any
    learned_energy: Any
    coulomb_energy: Any
    external_energy: Any
    fukui_weights: Any
    spin_fukui_weights: Any
    potential: Any
    potential_coefficients: Any
    electric_field: Any
    residual: Any
    spin_residual: Any
    converged: bool
    method: str
    electrostatic_backend: tuple[str, ...]
    oxidation: OxidationPrediction | None
    electronegativity: Any
    hardness: Any


class AlternativeElectronicModel(nn.Module):
    """Deliberately compact direct/QEq/auxiliary electronic alternatives."""

    def __init__(self, feature_dim: int, config: ElectronicConfig) -> None:
        super().__init__()
        self.config = config
        self.local = _mlp(feature_dim, config.hidden, 5)
        self.update = nn.ModuleList(
            _mlp(feature_dim + 2, config.hidden, 2)
            for _ in range(config.polarization_updates)
        )
        self.energy_readout = _mlp(feature_dim + 3, config.hidden, 1)
        _zero_last(self.local)
        for item in self.update:
            _zero_last(item)
        _zero_last(self.energy_readout)

    def _state(self, charge: Any, spin_vectors: Any | None, moment: Any) -> SpinMultipoleState:
        dimension = self.config.multipole_dim
        density = charge.new_zeros((charge.shape[0], dimension))
        density[:, 0] = charge
        spin = charge.new_zeros((charge.shape[0], 3, dimension))
        if spin_vectors is not None:
            spin[:, :, 0] = spin_vectors
        else:
            spin[:, 2, 0] = moment
        return SpinMultipoleState(density, spin)

    def forward(
        self,
        features: Any,
        *,
        positions: Any,
        batch: Any,
        cell: Any | None,
        pbc: Any | None,
        conditions: dict[str, Any],
        spin_vectors: Any | None,
    ) -> tuple[
        SpinMultipoleState, Any, Any, Any, Any, Any, Any, tuple[str, ...],
        Any, Any, Any, Any,
    ]:
        raw = self.local(features)
        charge_raw, charge_logits = raw[:, 0], raw[:, 1]
        moment = functional.softplus(raw[:, 2])
        electronegativity = raw[:, 3]
        hardness = functional.softplus(raw[:, 4]) + self.config.hardness_floor_eV
        qeq: QEqResult | None = None
        if self.config.method == "qeq":
            qeq = solve_qeq(
                electronegativity, hardness, positions, batch, cell=cell, pbc=pbc,
                conditions=conditions, config=self.config,
            )
            charge = qeq.charges
            weights = functional.softplus(charge_logits) + self.config.fukui_floor
            target = None
        else:
            charge, weights, target = constrain_charge_monopoles(
                charge_raw, charge_logits, batch, conditions, self.config
            )
        state = self._state(charge, spin_vectors, moment)
        potential = positions.new_zeros((positions.shape[0], self.config.multipole_dim))
        backends: tuple[str, ...] = ()
        if self.config.method == "fukui_auxiliary":
            for update in self.update:
                field = electrostatic_potential_features(
                    state, positions, batch, cell, pbc, self.config,
                    conditions=conditions,
                )
                potential, backends = field.coefficients, field.backend
                delta = update(torch.cat((features, charge[:, None], potential[:, :1]), dim=-1))
                charge, weights, target = constrain_charge_monopoles(
                    charge + delta[:, 0], charge_logits + delta[:, 1], batch,
                    conditions, self.config,
                )
                state = self._state(charge, spin_vectors, moment)
        if self.config.energy_coupling in {"learned", "full"}:
            if not backends:
                field = electrostatic_potential_features(
                    state, positions, batch, cell, pbc, self.config,
                    conditions=conditions,
                )
                potential, backends = field.coefficients, field.backend
            learned_atomic = self.config.learned_energy_scale_eV * self.energy_readout(
                torch.cat((features, charge[:, None], moment[:, None], potential[:, :1]), dim=-1)
            ).reshape(-1)
        else:
            learned_atomic = charge.new_zeros(charge.shape)
        if qeq is not None:
            coulomb_atomic, coulomb_graph, residual = (
                qeq.atomic_energy, qeq.graph_energy, qeq.residual
            )
        else:
            coulomb_atomic = charge.new_zeros(charge.shape)
            coulomb_graph = charge.new_zeros(graph_count(batch))
            residual = (
                charge.new_zeros(graph_count(batch))
                if target is None else scatter_sum(charge, batch, graph_count(batch)) - target
            )
        atomic = learned_atomic + coulomb_atomic
        energy = scatter_sum(atomic, batch, graph_count(batch))
        return (
            state, moment, energy, atomic,
            scatter_sum(learned_atomic, batch, graph_count(batch)), coulomb_graph,
            residual, backends, weights, electronegativity, hardness, potential,
        )


class StableElectronicModel(nn.Module):
    """Production polar density with explicitly labelled alternative paths."""

    def __init__(
        self,
        feature_dim: int,
        atomic_numbers: tuple[int, ...],
        config: ElectronicConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.production = (
            PolarDensityModel(feature_dim, config)
            if config.method == "polar"
            else None
        )
        self.alternative = (
            AlternativeElectronicModel(feature_dim, config)
            if config.method != "polar"
            else None
        )
        oxidation_input = feature_dim + 2
        self.oxidation = (
            OxidationStateHead(oxidation_input, atomic_numbers, config.oxidation)
            if config.oxidation.enabled else None
        )

    def _oxidation(
        self,
        features: Any,
        state: SpinMultipoleState,
        atomic_numbers: Any,
        batch: Any,
        conditions: dict[str, Any],
    ) -> OxidationPrediction | None:
        if self.oxidation is None:
            return None
        count = graph_count(batch)
        total = conditions.get("formal_total_charge", conditions.get("total_charge"))
        if total is None:
            total = state.charges.new_zeros(count)
        total = total.to(device=state.charge.device, dtype=state.charge.dtype)
        if total.ndim == 0 or total.shape == (1,):
            total = total.reshape(1).expand(count)
        exact = bool(
            not self.training
            and conditions.get("infer_oxidation_states", False)
            and self.config.oxidation.exact_inference
            and self.config.boundary_mode == "fixed_charge"
        )
        magnitude2 = state.magnetic_moments.square().sum(-1, keepdim=True)
        context = torch.cat((features, state.charges[:, None], magnitude2), dim=-1)
        return self.oxidation(context, atomic_numbers, batch, total, exact=exact)

    def forward(
        self,
        features: Any,
        *,
        positions: Any,
        atomic_numbers: Any,
        batch: Any,
        edge_index: Any,
        shifts: Any | None,
        cell: Any | None,
        pbc: Any | None,
        cutoff_A: float,
        conditions: dict[str, Any] | None = None,
        spin_vectors: Any | None = None,
    ) -> ElectronicPrediction:
        conditions = dict(conditions or {})
        count = graph_count(batch)
        if self.production is not None:
            result = self.production(
                features, positions=positions, batch=batch, edge_index=edge_index,
                shifts=shifts, cell=cell, pbc=pbc, cutoff_A=cutoff_A,
                conditions=conditions, spin_vectors=spin_vectors,
            )
            state = result.state
            magnitude = result.auxiliary_magnitudes
            energy, atomic = result.energy, result.atomic_energy
            learned, coulomb, external = (
                result.learned_energy, result.coulomb_energy, result.external_energy
            )
            residual, spin_residual = result.residual, result.spin_residual
            backends = result.electrostatic_backend
            weights, spin_weights = result.charge_fukui, result.spin_fukui
            potential_coefficients = result.potential_coefficients
            electronegativity, hardness = result.electronegativity, result.hardness
        else:
            assert self.alternative is not None
            (
                state, magnitude, energy, atomic, learned, coulomb, residual,
                backends, weights, electronegativity, hardness,
                potential_coefficients,
            ) = self.alternative(
                features, positions=positions, batch=batch, cell=cell, pbc=pbc,
                conditions=conditions, spin_vectors=spin_vectors,
            )
            external = energy.new_zeros(count)
            spin_residual = energy.new_zeros((count, 3))
            spin_weights = weights
        if potential_coefficients.shape[1] >= 4:
            electric_field = -coefficients_to_cartesian(
                potential_coefficients[:, multipole_slice(1)], 1
            )
        else:
            electric_field = positions.new_zeros((positions.shape[0], 3))
        oxidation = self._oxidation(
            features, state, atomic_numbers, batch, conditions
        )
        return ElectronicPrediction(
            state=state,
            magmoms=magnitude,
            collinear_spins=state.magnetic_moments[:, 2],
            energy=energy,
            atomic_energy=atomic,
            learned_energy=learned,
            coulomb_energy=coulomb,
            external_energy=external,
            fukui_weights=weights,
            spin_fukui_weights=spin_weights,
            potential=potential_coefficients[:, 0],
            potential_coefficients=potential_coefficients,
            electric_field=electric_field,
            residual=residual,
            spin_residual=spin_residual,
            converged=bool(
                torch.isfinite(residual).all().detach()
                and torch.isfinite(spin_residual).all().detach()
                and torch.all(residual.abs() <= self.config.constraint_tolerance).detach()
                and torch.all(spin_residual.abs() <= self.config.constraint_tolerance).detach()
            ),
            method=self.config.method,
            electrostatic_backend=backends,
            oxidation=oxidation,
            electronegativity=electronegativity,
            hardness=hardness,
        )


ElectronicFunctional = StableElectronicModel


__all__ = [
    "ElectronicFunctional", "ElectronicPrediction", "StableElectronicModel",
    "graph_count", "scatter_sum",
]
