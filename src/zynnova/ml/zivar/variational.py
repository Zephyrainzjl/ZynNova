"""Default constrained variational ``q/p/Q/m`` model.

This module is the only production electronic path.  A local equivariant
backbone parameterises one convex electro-spin functional; a projected,
matrix-free PCG solve minimises that functional exactly enough to satisfy the
configured residual and charge constraints.  The converged state is then
inserted back into the same scalar functional.  Envelope/Hellmann--Feynman
derivatives of that scalar provide model gradients, forces and stress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .config import ElectronicConfig, ElectrostaticsConfig, SCFConfig
from .ewald_reference import EwaldParameters, ewald_energy, plan_ewald
from .functional import (
    ElectroSpinFunctional,
    ElectroSpinParameters,
    LinearConstraints,
)
from .mesh import plan_mesh, validate_boundary
from .operators import CallableLinearOperator, DiagonalPreconditioner
from .pme import PMEPlan, plan_pme, pme_energy
from .scf import SCFReport, SCFSolverConfig, solve_quadratic_scf
from .types import Conditions, ElectronicState, EnergyBreakdown, project_stf

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
    for item in reversed(tuple(module.modules())):
        if isinstance(item, nn.Linear):
            nn.init.zeros_(item.weight)
            if item.bias is not None:
                nn.init.zeros_(item.bias)
            return


def _scatter_sum(values: Any, index: Any, count: int) -> Any:
    output = values.new_zeros((count, *values.shape[1:]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


def _isolated_energy(
    positions: Any,
    charges: Any,
    *,
    coulomb_constant_eV_A: float,
    pair_block: int,
) -> Any:
    """Blocked, differentiable open-boundary Coulomb energy in either dtype."""

    count = int(charges.numel())
    if count < 2:
        return charges.new_zeros(())
    pair = torch.triu_indices(count, count, offset=1, device=charges.device)
    pieces = []
    for start in range(0, pair.shape[1], pair_block):
        local = pair[:, start : start + pair_block]
        vector = positions[local[0]] - positions[local[1]]
        squared = vector.square().sum(-1)
        if bool(torch.any(squared <= torch.finfo(positions.dtype).eps).detach()):
            raise ValueError("distinct point charges occupy a coincident position")
        pieces.append(charges[local[0]] * charges[local[1]] / torch.sqrt(squared))
    return float(coulomb_constant_eV_A) * torch.cat(pieces).sum()


class ElectrostaticQuadraticOperator:
    """Matrix-free monopole Coulomb Hessian for Ewald/PME or open boundaries.

    Standard self-subtracted Coulomb matrices need not be positive
    semidefinite by themselves.  Strict convexity is checked for the complete
    onsite-plus-Coulomb Hessian by the SCF solver's curvature gate.
    """

    symmetric = True
    positive_semidefinite = False

    def __init__(
        self,
        positions: Any,
        batch: Any,
        cell: Any | None,
        pbc: Any | None,
        electronic: ElectronicConfig,
        electrostatics: ElectrostaticsConfig,
        *,
        differentiable: bool,
    ) -> None:
        self.positions = positions
        self.batch = batch
        self.cell = cell
        self.pbc = pbc
        self.electronic = electronic
        self.config = electrostatics
        self.differentiable = bool(differentiable)
        self.graph_count = int(batch.max().detach().item()) + 1 if batch.numel() else 0
        self.plans: list[PMEPlan | EwaldParameters | None] = []
        for graph in range(self.graph_count):
            flags = self._flags(graph)
            periodic = bool(torch.all(flags).detach())
            validate_boundary(flags, periodic=periodic)
            if electrostatics.boundary == "periodic_3d" and not periodic:
                raise ValueError(
                    "electrostatics boundary='periodic_3d' requires pbc=[True,True,True]"
                )
            if electrostatics.boundary == "isolated" and periodic:
                raise ValueError(
                    "electrostatics boundary='isolated' requires pbc=[False,False,False]"
                )
            if not periodic:
                self.plans.append(None)
                continue
            if cell is None:
                raise ValueError("periodic electrostatics requires a cell")
            local_cell = cell[graph]
            if electrostatics.method == "pme":
                self.plans.append(self._pme_plan(local_cell))
            else:
                self.plans.append(self._ewald_plan(local_cell))
        self.stabilizer = self._build_stabilizer()

    def _flags(self, graph: int) -> Any:
        if self.pbc is None:
            return torch.zeros(3, device=self.positions.device, dtype=torch.bool)
        return self.pbc.reshape(self.graph_count, 3)[graph]

    def _ewald_plan(self, cell: Any) -> EwaldParameters:
        selected = plan_ewald(
            cell.detach(),
            self.config.error_target,
            real_cutoff_A=self.config.real_cutoff_A,
        )
        if self.config.alpha_per_A is None:
            return selected
        alpha = float(self.config.alpha_per_A)
        decay = math.sqrt(-math.log(self.config.error_target * 0.1))
        real_cutoff = self.config.real_cutoff_A or decay / alpha
        return EwaldParameters(
            alpha_inv_A=alpha,
            real_cutoff_A=float(real_cutoff),
            reciprocal_cutoff_inv_A=2.0 * alpha * decay,
            error_target=self.config.error_target,
        )

    def _pme_plan(self, cell: Any) -> PMEPlan:
        if self.config.alpha_per_A is None:
            return plan_pme(
                cell.detach(),
                self.config.error_target,
                real_cutoff_A=self.config.real_cutoff_A,
                interpolation_order=self.config.interpolation_order,
                mesh_shape=self.config.mesh,
            )
        ewald = self._ewald_plan(cell)
        mesh = plan_mesh(
            cell.detach(),
            self.config.error_target,
            ewald.alpha_inv_A,
            interpolation_order=self.config.interpolation_order,
            shape=self.config.mesh,
        )
        return PMEPlan(ewald=ewald, mesh=mesh)

    def _build_stabilizer(self) -> Any:
        """Return a differentiable atomic self hardness for strict convexity.

        For an isolated point-charge matrix this is its exact Gershgorin row
        sum plus a small margin.  For periodic Ewald it also compensates the
        signed point self correction; the reciprocal mesh contribution is a
        positive quadratic form.  PCG still verifies the complete operator
        and fails closed if the conservative periodic bound is insufficient.
        """

        diagonal = self.positions.new_zeros(self.positions.shape[0])
        constant = float(self.electronic.coulomb_constant_eV_A)
        margin = max(float(self.electronic.qeq_jitter_eV), 1.0e-6)
        for graph in range(self.graph_count):
            atom_indices = torch.nonzero(self.batch == graph, as_tuple=False).reshape(-1)
            atom_count = int(atom_indices.numel())
            if atom_count < 2:
                local_diagonal = self.positions.new_full((atom_count,), margin)
            else:
                local_positions = self.positions[atom_indices]
                pair = torch.triu_indices(
                    atom_count, atom_count, offset=1, device=self.positions.device
                )
                local_diagonal = self.positions.new_zeros(atom_count)
                periodic = bool(torch.all(self._flags(graph)).detach())
                for start in range(0, pair.shape[1], self.electronic.direct_pair_block):
                    local = pair[:, start : start + self.electronic.direct_pair_block]
                    if periodic:
                        if self.cell is None:
                            raise ValueError("periodic electrostatics requires a cell")
                        inverse = torch.linalg.inv(self.cell[graph])
                        fractional = local_positions @ inverse
                        delta = fractional[local[0]] - fractional[local[1]]
                        delta = delta - torch.round(delta)
                        vector = delta @ self.cell[graph]
                    else:
                        vector = local_positions[local[0]] - local_positions[local[1]]
                    squared = vector.square().sum(-1)
                    if bool(torch.any(squared <= torch.finfo(vector.dtype).eps).detach()):
                        raise ValueError("distinct point charges occupy a coincident position")
                    distance = torch.sqrt(squared)
                    if periodic:
                        plan = self.plans[graph]
                        alpha = (
                            plan.ewald.alpha_inv_A
                            if isinstance(plan, PMEPlan)
                            else plan.alpha_inv_A
                        )
                        weight = constant * torch.special.erfc(alpha * distance) / distance
                    else:
                        weight = constant / distance
                    local_diagonal = local_diagonal.index_add(0, local[0], weight)
                    local_diagonal = local_diagonal.index_add(0, local[1], weight)
                if periodic:
                    plan = self.plans[graph]
                    alpha = (
                        plan.ewald.alpha_inv_A
                        if isinstance(plan, PMEPlan)
                        else plan.alpha_inv_A
                    )
                    local_diagonal = local_diagonal + (
                        2.0 * constant * alpha / math.sqrt(math.pi)
                    )
                local_diagonal = local_diagonal + margin
            diagonal = diagonal.index_add(0, atom_indices, local_diagonal)
        return diagonal

    def energy(self, charges: Any) -> Any:
        values = 0.5 * (self.stabilizer * charges.square()).sum()
        for graph in range(self.graph_count):
            mask = self.batch == graph
            local_positions = self.positions[mask]
            local_charges = charges[mask]
            flags = self._flags(graph)
            periodic = bool(torch.all(flags).detach())
            if periodic:
                if self.cell is None:
                    raise ValueError("periodic electrostatics requires a cell")
                plan = self.plans[graph]
                if self.config.method == "pme":
                    if not isinstance(plan, PMEPlan):
                        raise TypeError("invalid PME plan")
                    values = values + pme_energy(
                        local_positions,
                        local_charges,
                        self.cell[graph],
                        flags,
                        plan,
                        neutralizing_background=self.config.neutralizing_background,
                        coulomb_constant_eV_A=self.electronic.coulomb_constant_eV_A,
                    ).energy
                else:
                    if not isinstance(plan, EwaldParameters):
                        raise TypeError("invalid direct-Ewald plan")
                    values = values + ewald_energy(
                        local_positions,
                        local_charges,
                        self.cell[graph],
                        flags,
                        plan,
                        neutralizing_background=self.config.neutralizing_background,
                        coulomb_constant_eV_A=self.electronic.coulomb_constant_eV_A,
                    ).energy
            else:
                atom_count = int(local_charges.numel())
                if atom_count > self.config.isolated_direct_max_atoms:
                    raise ValueError(
                        "isolated direct electrostatics exceeds isolated_direct_max_atoms; "
                        "an FMM backend is required for a larger open system"
                    )
                values = values + _isolated_energy(
                    local_positions,
                    local_charges,
                    coulomb_constant_eV_A=self.electronic.coulomb_constant_eV_A,
                    pair_block=self.electronic.direct_pair_block,
                )
        return values

    def matvec(
        self,
        state: ElectronicState,
        *,
        batch: Any,
        conditions: Conditions,
    ) -> ElectronicState:
        del conditions
        if batch.shape != self.batch.shape or not bool(torch.equal(batch, self.batch)):
            raise ValueError("electrostatic operator batch differs from its geometry")
        with torch.enable_grad():
            charges = state.q
            if not charges.requires_grad:
                charges = charges.detach().requires_grad_(True)
            energy = self.energy(charges)
            potential = torch.autograd.grad(
                energy,
                charges,
                create_graph=self.differentiable,
                retain_graph=self.differentiable,
            )[0]
        zero_vector = state.p.new_zeros(state.p.shape)
        zero_rank2 = state.Q.new_zeros(state.Q.shape)
        return ElectronicState(potential, zero_vector, zero_rank2, zero_vector.clone())


class _ImplicitMatrixFreeSolve(torch.autograd.Function):
    """Differentiate a constrained linear solve with an adjoint PCG solve.

    The operator and preconditioner are evaluated at the converged stationary
    state and deliberately treated as constants here.  Their parameter
    derivatives enter through the live stationarity residual supplied as
    ``rhs``; differentiating the inverse itself would multiply a zero residual
    and is therefore unnecessary at the stationary point.
    """

    @staticmethod
    def forward(
        ctx: Any,
        rhs: Any,
        operator: Any,
        constraint: Any,
        preconditioner: Any,
        solver_config: SCFSolverConfig,
    ) -> Any:
        target = rhs.new_zeros(constraint.shape[0])
        result = solve_quadratic_scf(
            operator,
            -rhs,
            constraint=constraint,
            target=target,
            preconditioner=preconditioner,
            config=solver_config,
        )
        ctx.operator = operator
        ctx.constraint = constraint
        ctx.preconditioner = preconditioner
        ctx.solver_config = solver_config
        return result.solution

    @staticmethod
    def backward(ctx: Any, gradient_output: Any) -> tuple[Any, None, None, None, None]:
        target = gradient_output.new_zeros(ctx.constraint.shape[0])
        result = solve_quadratic_scf(
            ctx.operator,
            -gradient_output,
            constraint=ctx.constraint,
            target=target,
            preconditioner=ctx.preconditioner,
            config=ctx.solver_config,
        )
        return result.solution, None, None, None, None


@dataclass(slots=True)
class VariationalPrediction:
    state: ElectronicState
    parameters: ElectroSpinParameters
    breakdown: EnergyBreakdown
    report: SCFReport
    constraints: LinearConstraints
    electrostatic_potential: Any
    electrostatic_backend: tuple[str, ...]


class VariationalElectroSpinModel(nn.Module):
    """Local coefficient model plus fail-closed constrained SCF minimisation."""

    def __init__(
        self,
        feature_dim: int,
        electronic: ElectronicConfig,
        scf: SCFConfig,
        electrostatics: ElectrostaticsConfig,
        *,
        bohr_magneton_eV_per_T: float = 5.7883818060e-5,
        constrain_total_magnetization: bool = False,
    ) -> None:
        super().__init__()
        if electronic.method != "variational":
            raise ValueError("VariationalElectroSpinModel requires method='variational'")
        self.electronic_config = electronic
        self.scf_config = scf
        self.electrostatics_config = electrostatics
        self.constrain_total_magnetization = bool(constrain_total_magnetization)
        self.local = _mlp(feature_dim, electronic.hidden, 7)
        self.edge = _mlp(2 * feature_dim + 1, electronic.hidden, 2)
        _zero_last(self.local)
        _zero_last(self.edge)
        self.functional = ElectroSpinFunctional(
            positive_floor=0.0,
            bohr_magneton_eV_per_T=bohr_magneton_eV_per_T,
        )

    def _edge_drives(
        self,
        features: Any,
        positions: Any,
        edge_index: Any,
        shifts: Any | None,
        cutoff_A: float,
    ) -> tuple[Any, Any]:
        atom_count = int(features.shape[0])
        source, target = edge_index[0], edge_index[1]
        if source.numel() == 0:
            return (
                positions.new_zeros((atom_count, 3)),
                positions.new_zeros((atom_count, 3, 3)),
            )
        vector = positions[target] - positions[source]
        if shifts is not None:
            vector = vector + shifts
        distance = torch.linalg.vector_norm(vector, dim=-1).clamp_min(
            torch.finfo(positions.dtype).eps
        )
        direction = vector / distance[:, None]
        x = distance / float(cutoff_A)
        envelope = torch.where(
            x < 1.0,
            (1.0 - x.square()).clamp_min(0.0).square(),
            torch.zeros_like(x),
        )
        edge_features = torch.cat(
            (
                features[source] + features[target],
                (features[source] - features[target]).abs(),
                x[:, None],
            ),
            dim=-1,
        )
        coefficients = self.edge(edge_features) * envelope[:, None]
        dipole = _scatter_sum(
            coefficients[:, :1] * direction,
            source,
            atom_count,
        )
        identity = torch.eye(3, device=positions.device, dtype=positions.dtype)
        dyadic = direction[:, :, None] * direction[:, None, :] - identity[None] / 3.0
        quadrupole = _scatter_sum(
            coefficients[:, 1, None, None] * dyadic,
            source,
            atom_count,
        )
        return dipole, project_stf(quadrupole)

    def _make_parameters(
        self,
        features: Any,
        positions: Any,
        edge_index: Any,
        shifts: Any | None,
        cutoff_A: float,
    ) -> ElectroSpinParameters:
        raw = self.local(features)
        dipole, quadrupole = self._edge_drives(
            features, positions, edge_index, shifts, cutoff_A
        )
        cfg = self.electronic_config
        return ElectroSpinParameters(
            reference_atomic_energy=cfg.learned_energy_scale_eV * raw[:, 0],
            electronegativity=raw[:, 1],
            hardness=functional.softplus(raw[:, 2]) + cfg.hardness_floor_eV,
            inverse_polarizability=(
                functional.softplus(raw[:, 3])
                + cfg.dipole_stiffness_floor_eV_per_eA2
            ),
            inverse_quadrupole_polarizability=(
                functional.softplus(raw[:, 4])
                + cfg.quadrupole_stiffness_floor_eV_per_eA4
            ),
            inverse_magnetic_susceptibility=(
                functional.softplus(raw[:, 5])
                + cfg.magnetic_stiffness_floor_eV_per_muB2
            ),
            dipole_drive=cfg.potential_scale_eV * dipole,
            quadrupole_drive=cfg.potential_scale_eV * quadrupole,
            magnetic_drive=features.new_zeros((features.shape[0], 3)),
            spin_coupling=cfg.learned_energy_scale_eV * torch.tanh(raw[:, 6]),
        )

    @staticmethod
    def _detached_parameters(parameters: ElectroSpinParameters) -> ElectroSpinParameters:
        def detached(value: Any | None) -> Any | None:
            return None if value is None else value.detach()

        return ElectroSpinParameters(
            electronegativity=detached(parameters.electronegativity),
            hardness=detached(parameters.hardness),
            inverse_polarizability=detached(parameters.inverse_polarizability),
            inverse_quadrupole_polarizability=detached(
                parameters.inverse_quadrupole_polarizability
            ),
            inverse_magnetic_susceptibility=detached(
                parameters.inverse_magnetic_susceptibility
            ),
            reference_atomic_energy=detached(parameters.reference_atomic_energy),
            dipole_drive=detached(parameters.dipole_drive),
            quadrupole_drive=detached(parameters.quadrupole_drive),
            magnetic_drive=detached(parameters.magnetic_drive),
            spin_coupling=detached(parameters.spin_coupling),
        )

    @staticmethod
    def _zero_state(reference: Any, atom_count: int) -> ElectronicState:
        return ElectronicState(
            q=reference.new_zeros(atom_count),
            p=reference.new_zeros((atom_count, 3)),
            Q=reference.new_zeros((atom_count, 3, 3)),
            m=reference.new_zeros((atom_count, 3)),
        )

    def _constraints(
        self,
        state: ElectronicState,
        batch: Any,
        conditions: Conditions,
    ) -> LinearConstraints:
        count = int(batch.max().detach().item()) + 1 if batch.numel() else 0
        total_charge = conditions.total_charge
        if total_charge is None:
            target_charge = state.q.new_zeros(count)
        else:
            target_charge = torch.as_tensor(
                total_charge, device=state.q.device, dtype=state.q.dtype
            ).reshape(-1)
            if target_charge.shape == (1,) and count != 1:
                target_charge = target_charge.expand(count)
            if target_charge.shape != (count,):
                raise ValueError("total_charge must be scalar or have shape [B]")
        charge_weights = functional.one_hot(
            batch.to(torch.long), num_classes=count
        ).to(dtype=state.q.dtype).T
        if not self.scf_config.warm_start:
            # This branch has no numerical effect; it keeps the warm-start
            # policy visible in the immutable solver contract.
            target_charge = target_charge.clone()
        if conditions.total_magnetization is None:
            if self.constrain_total_magnetization:
                raise ValueError(
                    "constrain_total_magnetization=True requires a per-graph "
                    "total_magnetization condition"
                )
            return LinearConstraints(
                target=target_charge,
                q_weights=charge_weights,
                labels=tuple(f"total_charge[{index}]" for index in range(count)),
            )
        if not self.constrain_total_magnetization:
            raise ValueError(
                "total_magnetization was supplied but its SCF constraint is disabled"
            )
        magnetization = torch.as_tensor(
            conditions.total_magnetization,
            device=state.q.device,
            dtype=state.q.dtype,
        ).reshape(count, 3)
        graph_membership = charge_weights.T
        m_weights = state.m.new_zeros((3 * count, state.atom_count, 3))
        for graph in range(count):
            for component in range(3):
                m_weights[3 * graph + component, :, component] = graph_membership[:, graph]
        q_weights = state.q.new_zeros((4 * count, state.atom_count))
        q_weights[:count] = charge_weights
        all_m_weights = state.m.new_zeros((4 * count, state.atom_count, 3))
        all_m_weights[count:] = m_weights
        target = torch.cat((target_charge, magnetization.reshape(-1)))
        labels = tuple(f"total_charge[{index}]" for index in range(count)) + tuple(
            f"total_magnetization[{graph},{component}]"
            for graph in range(count)
            for component in range(3)
        )
        return LinearConstraints(
            target=target,
            q_weights=q_weights,
            m_weights=all_m_weights,
            labels=labels,
        )

    @staticmethod
    def _preconditioner_diagonal(parameters: ElectroSpinParameters) -> Any:
        return torch.cat(
            (
                parameters.hardness[:, None],
                parameters.inverse_polarizability[:, None].expand(-1, 3),
                parameters.inverse_quadrupole_polarizability[:, None].expand(-1, 5),
                parameters.inverse_magnetic_susceptibility[:, None].expand(-1, 3),
            ),
            dim=-1,
        ).reshape(-1)

    def forward(
        self,
        features: Any,
        *,
        positions: Any,
        batch: Any,
        edge_index: Any,
        shifts: Any | None,
        cell: Any | None,
        pbc: Any | None,
        cutoff_A: float,
        conditions: dict[str, Any] | Conditions | None,
        external_spins: Any,
        short_range_energy: Any,
        external_spin_energy: Any,
    ) -> VariationalPrediction:
        condition_value = (
            conditions
            if isinstance(conditions, Conditions)
            else Conditions.from_mapping(conditions)
        )
        parameters = self._make_parameters(
            features, positions, edge_index, shifts, cutoff_A
        )
        atom_count = int(features.shape[0])
        zero = self._zero_state(features, atom_count)
        detached_parameters = self._detached_parameters(parameters)
        solve_positions = positions.detach()
        solve_cell = None if cell is None else cell.detach()
        solve_operator = ElectrostaticQuadraticOperator(
            solve_positions,
            batch,
            solve_cell,
            pbc,
            self.electronic_config,
            self.electrostatics_config,
            differentiable=False,
        )
        detached_conditions = Conditions.from_mapping(
            {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in condition_value.as_dict().items()
            }
        )
        linear_state = self.functional.stationarity_gradient(
            zero,
            detached_parameters,
            batch=batch,
            positions=solve_positions,
            conditions=detached_conditions,
            external_spins=external_spins.detach(),
            quadratic_operator=solve_operator,
        )
        linear = linear_state.pack().reshape(-1)

        def matvec(value: Any) -> Any:
            state = ElectronicState.from_packed(value.reshape(atom_count, -1))
            response = self.functional.quadratic_matvec(
                state,
                detached_parameters,
                batch=batch,
                conditions=detached_conditions,
                quadratic_operator=solve_operator,
            )
            return response.pack().reshape(-1)

        operator = CallableLinearOperator(
            int(linear.numel()),
            matvec,
            dtype=linear.dtype,
            device=linear.device,
        )
        constraints = self._constraints(zero, batch, detached_conditions)
        constraint_matrix = constraints.packed_matrix(zero)
        warm_start = detached_conditions.extras.get("scf_warm_start")
        if warm_start is not None:
            if not self.scf_config.warm_start:
                raise ValueError("scf_warm_start was supplied while warm starts are disabled")
            warm_start = torch.as_tensor(
                warm_start, device=linear.device, dtype=linear.dtype
            ).reshape(-1)
        preconditioner = (
            DiagonalPreconditioner(
                self._preconditioner_diagonal(detached_parameters)
            )
            if self.scf_config.preconditioner == "onsite"
            else None
        )
        solver_config = SCFSolverConfig(
            atol=self.scf_config.atol,
            rtol=self.scf_config.rtol,
            energy_atol=(
                self.scf_config.energy_atol_eV_per_atom * max(1, atom_count)
            ),
            max_iter=self.scf_config.max_iter,
            constraint_atol=min(
                self.electronic_config.constraint_tolerance,
                self.scf_config.atol,
            ),
            curvature_tolerance=self.scf_config.negative_curvature_tolerance,
        )
        result = solve_quadratic_scf(
            operator,
            linear,
            constraint=constraint_matrix,
            target=constraints.target,
            preconditioner=preconditioner,
            warm_start=warm_start,
            config=solver_config,
        )
        stationary_state = ElectronicState.from_packed(
            result.solution.detach().reshape(atom_count, -1)
        )
        energy_operator = ElectrostaticQuadraticOperator(
            positions,
            batch,
            cell,
            pbc,
            self.electronic_config,
            self.electrostatics_config,
            differentiable=True,
        )
        live_gradient = self.functional.stationarity_gradient(
            stationary_state,
            parameters,
            batch=batch,
            positions=positions,
            conditions=condition_value,
            external_spins=external_spins,
            quadratic_operator=energy_operator,
        ).flatten()
        correction = _ImplicitMatrixFreeSolve.apply(
            live_gradient,
            operator,
            constraint_matrix,
            preconditioner,
            solver_config,
        )
        state = ElectronicState.from_flattened(
            result.solution.detach() - correction,
            atom_count=atom_count,
        )
        # The custom adjoint has zero forward correction at the stationary
        # point, so first derivatives reduce to the constrained envelope
        # theorem.  Evaluating the scalar with this live state also preserves
        # the implicit-state terms needed by mixed second derivatives during
        # force/stress training.
        breakdown = self.functional(
            state,
            parameters,
            batch=batch,
            positions=positions,
            conditions=condition_value,
            external_spins=external_spins,
            short_range_energy=short_range_energy,
            external_spin_energy=external_spin_energy,
            quadratic_operator=energy_operator,
        )
        response = energy_operator.matvec(
            state, batch=batch, conditions=condition_value
        )
        backend = tuple(
            (
                self.electrostatics_config.method
                if bool(torch.all(energy_operator._flags(graph)).detach())
                else "isolated_direct_matrix_free"
            )
            for graph in range(energy_operator.graph_count)
        )
        return VariationalPrediction(
            state=state,
            parameters=parameters,
            breakdown=breakdown,
            report=result.report,
            constraints=constraints,
            electrostatic_potential=response.q,
            electrostatic_backend=backend,
        )


__all__ = [
    "ElectrostaticQuadraticOperator",
    "VariationalElectroSpinModel",
    "VariationalPrediction",
]
