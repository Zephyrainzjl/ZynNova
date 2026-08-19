"""One convex variational energy for ZIVAR charge, polarisation and magnetism.

For fixed geometry and external dynamical spins ``S``, the variational state
is ``x=(q,p,Q,m)`` and this module evaluates

.. math::

   E = E_{short}(R,S) + E_{spin}(R,S)
       + l(S)^T x + \tfrac12 x^T (D + K) x + E_{external}.

``D`` contains strictly positive onsite hardness/response coefficients and
``K`` is an optional symmetric matrix-free interaction operator.  Coulomb
operators with their self term removed need not be positive-semidefinite by
themselves: production convexity requires the *total* Hessian ``D + K`` to be
positive-definite on the linear-constraint null space.  The SCF Krylov solve
must detect negative curvature and fail closed.  ``S`` remains an external
state; the induced axial moment ``m`` is minimised together with charge ``q``,
polar dipole ``p`` and Cartesian STF quadrupole ``Q``.

All public energies are assembled into a single :class:`EnergyBreakdown`.
Forces, stress, effective fields and torques must be derived from its scalar
``breakdown.scalar`` rather than from independent prediction heads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ._deps import require_torch
from .errors import (
    ConstraintError,
    NonFiniteFunctionalError,
    NonPositiveDefiniteError,
    ShapeError,
    SymmetryError,
    ZIVARValidationError,
)
from .types import (
    Conditions,
    ConstraintOperatorLike,
    ElectronicState,
    EnergyBreakdown,
    ZIVARBatch,
    assert_stf,
    project_stf,
    stf_to_components,
)

torch = require_torch()
nn = torch.nn


@dataclass(frozen=True, slots=True)
class ElectroSpinParameters:
    """Local coefficients of the convex electro-spin functional.

    Scalar response coefficients may be scalar tensors or have shape ``[N]``
    and must be strictly positive.  Linear drives are physical tensors:
    ``dipole_drive`` is polar, ``quadrupole_drive`` is polar STF rank two and
    ``magnetic_drive`` is axial and time odd.  ``spin_coupling`` multiplies
    the invariant ``-S·m`` and therefore preserves convexity in ``m``.  With
    charge in ``e``, length in angstrom and moments in ``mu_B``, the onsite
    stiffnesses have units eV/e², eV/(e angstrom)², eV/(e angstrom²)² and
    eV/mu_B² respectively.
    """

    electronegativity: Any
    hardness: Any
    inverse_polarizability: Any
    inverse_quadrupole_polarizability: Any
    inverse_magnetic_susceptibility: Any
    reference_atomic_energy: Any | None = None
    dipole_drive: Any | None = None
    quadrupole_drive: Any | None = None
    magnetic_drive: Any | None = None
    spin_coupling: Any | None = None

    @property
    def dipole_stiffness(self) -> Any:
        return self.inverse_polarizability

    @property
    def quadrupole_stiffness(self) -> Any:
        return self.inverse_quadrupole_polarizability

    @property
    def magnetic_stiffness(self) -> Any:
        return self.inverse_magnetic_susceptibility


@dataclass(frozen=True, slots=True)
class _ResolvedParameters:
    reference_atomic_energy: Any
    electronegativity: Any
    hardness: Any
    inverse_polarizability: Any
    inverse_quadrupole_polarizability: Any
    inverse_magnetic_susceptibility: Any
    dipole_drive: Any
    quadrupole_drive: Any
    magnetic_drive: Any
    spin_coupling: Any


@runtime_checkable
class QuadraticOperatorLike(Protocol):
    """Matrix-free symmetric interaction operator used by the functional.

    ``positive_semidefinite`` is auditable metadata, not a requirement: an
    indefinite interaction is valid when the total constrained Hessian is
    positive-definite.
    """

    symmetric: bool
    positive_semidefinite: bool

    def matvec(
        self,
        state: ElectronicState,
        *,
        batch: Any,
        conditions: Conditions,
    ) -> ElectronicState:
        """Return ``K @ state`` in structured form."""


def _as_tensor(value: Any, reference: Any, name: str) -> Any:
    try:
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ZIVARValidationError(f"{name} is not tensor-compatible") from exc


def _atom_scalar(value: Any, reference: Any, atom_count: int, name: str) -> Any:
    tensor = _as_tensor(value, reference, name)
    if tensor.ndim == 0:
        return tensor.expand(atom_count)
    if tuple(tensor.shape) != (atom_count,):
        raise ShapeError(f"{name} must be scalar or have shape [N]")
    return tensor


def _atom_vector(value: Any | None, reference: Any, atom_count: int, name: str) -> Any:
    if value is None:
        return reference.new_zeros((atom_count, 3))
    tensor = _as_tensor(value, reference, name)
    if tuple(tensor.shape) == (3,):
        return tensor.expand(atom_count, 3)
    if tuple(tensor.shape) != (atom_count, 3):
        raise ShapeError(f"{name} must have shape [3] or [N,3]")
    return tensor


def _atom_rank2(value: Any | None, reference: Any, atom_count: int, name: str) -> Any:
    if value is None:
        return reference.new_zeros((atom_count, 3, 3))
    tensor = _as_tensor(value, reference, name)
    if tuple(tensor.shape) == (3, 3):
        tensor = tensor.expand(atom_count, 3, 3)
    if tuple(tensor.shape) != (atom_count, 3, 3):
        raise ShapeError(f"{name} must have shape [3,3] or [N,3,3]")
    return tensor


def _ensure_finite(values: tuple[tuple[str, Any], ...]) -> None:
    for name, value in values:
        if not bool(torch.isfinite(value).all().detach()):
            raise NonFiniteFunctionalError(f"{name} contains non-finite values")


def _resolve_parameters(
    parameters: ElectroSpinParameters,
    state: ElectronicState,
    *,
    positive_floor: float,
) -> _ResolvedParameters:
    if not isinstance(parameters, ElectroSpinParameters):
        raise TypeError("parameters must be ElectroSpinParameters")
    count = state.atom_count
    reference = state.q
    zero = reference.new_zeros(count)
    resolved = _ResolvedParameters(
        reference_atomic_energy=(
            zero
            if parameters.reference_atomic_energy is None
            else _atom_scalar(
                parameters.reference_atomic_energy,
                reference,
                count,
                "reference_atomic_energy",
            )
        ),
        electronegativity=_atom_scalar(
            parameters.electronegativity, reference, count, "electronegativity"
        ),
        hardness=_atom_scalar(parameters.hardness, reference, count, "hardness"),
        inverse_polarizability=_atom_scalar(
            parameters.inverse_polarizability,
            reference,
            count,
            "inverse_polarizability",
        ),
        inverse_quadrupole_polarizability=_atom_scalar(
            parameters.inverse_quadrupole_polarizability,
            reference,
            count,
            "inverse_quadrupole_polarizability",
        ),
        inverse_magnetic_susceptibility=_atom_scalar(
            parameters.inverse_magnetic_susceptibility,
            reference,
            count,
            "inverse_magnetic_susceptibility",
        ),
        dipole_drive=_atom_vector(parameters.dipole_drive, reference, count, "dipole_drive"),
        quadrupole_drive=_atom_rank2(
            parameters.quadrupole_drive, reference, count, "quadrupole_drive"
        ),
        magnetic_drive=_atom_vector(
            parameters.magnetic_drive, reference, count, "magnetic_drive"
        ),
        spin_coupling=_atom_scalar(
            zero if parameters.spin_coupling is None else parameters.spin_coupling,
            reference,
            count,
            "spin_coupling",
        ),
    )
    try:
        assert_stf(resolved.quadrupole_drive)
    except ValueError as exc:
        raise SymmetryError("quadrupole_drive must be symmetric and traceless") from exc
    finite = tuple((name, getattr(resolved, name)) for name in resolved.__dataclass_fields__)
    _ensure_finite(finite)
    for name in (
        "hardness",
        "inverse_polarizability",
        "inverse_quadrupole_polarizability",
        "inverse_magnetic_susceptibility",
    ):
        value = getattr(resolved, name)
        if bool(torch.any(value <= float(positive_floor)).detach()):
            raise NonPositiveDefiniteError(
                f"{name} must be strictly greater than {positive_floor}"
            )
    return resolved


def _resolve_geometry(
    batch: Any,
    positions: Any | None,
    state: ElectronicState,
) -> tuple[Any, Any | None, int]:
    if isinstance(batch, ZIVARBatch):
        if positions is not None:
            raise ZIVARValidationError(
                "positions must not be repeated when batch is a ZIVARBatch"
            )
        positions = batch.positions
        batch_index = batch.batch
    elif isinstance(batch, Mapping):
        if "batch" not in batch:
            raise ShapeError("batch mapping requires a 'batch' tensor")
        batch_index = batch["batch"]
        if positions is None:
            positions = batch.get("positions")
    else:
        batch_index = batch
    if tuple(batch_index.shape) != (state.atom_count,):
        raise ShapeError("batch must have shape [N]")
    if batch_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ShapeError("batch must contain integer graph indices")
    if state.atom_count == 0:
        raise ShapeError("electronic state must contain at least one atom")
    if bool(torch.any(batch_index < 0).detach()):
        raise ShapeError("batch graph indices must be nonnegative")
    graph_count = int(batch_index.max().detach().item()) + 1
    occupied = torch.bincount(batch_index.to(torch.long), minlength=graph_count)
    if bool(torch.any(occupied == 0).detach()):
        raise ShapeError("batch graph indices must be contiguous and nonempty")
    if positions is not None:
        positions = _as_tensor(positions, state.q, "positions")
        if tuple(positions.shape) != (state.atom_count, 3):
            raise ShapeError("positions must have shape [N,3]")
    return batch_index.to(device=state.q.device, dtype=torch.long), positions, graph_count


def _scatter_sum(values: Any, batch: Any, graph_count: int) -> Any:
    result = values.new_zeros((graph_count, *values.shape[1:]))
    result.index_add_(0, batch, values)
    return result


def _graph_energy(value: Any | None, reference: Any, graph_count: int, name: str) -> Any:
    if value is None:
        return reference.new_zeros(graph_count)
    tensor = _as_tensor(value, reference, name)
    if tensor.ndim == 0 or tuple(tensor.shape) == (1,):
        return tensor.reshape(1).expand(graph_count)
    if tuple(tensor.shape) != (graph_count,):
        raise ShapeError(f"{name} must be scalar or have shape [B]")
    return tensor


def _condition_tensor(
    value: Any | None,
    reference: Any,
    batch: Any,
    graph_count: int,
    tail: tuple[int, ...],
    name: str,
) -> Any:
    atom_count = int(batch.shape[0])
    if value is None:
        return reference.new_zeros((atom_count, *tail))
    tensor = _as_tensor(value, reference, name)
    if tuple(tensor.shape) == tail:
        return tensor.expand(atom_count, *tail)
    if tuple(tensor.shape) == (1, *tail):
        return tensor.expand(atom_count, *tail)
    if tuple(tensor.shape) == (graph_count, *tail):
        return tensor[batch]
    if tuple(tensor.shape) == (atom_count, *tail):
        return tensor
    expected = f"{list(tail)}, [B,{','.join(map(str, tail))}] or [N,{','.join(map(str, tail))}]"
    raise ShapeError(f"{name} has invalid shape; expected {expected}")


def _condition_scalar(
    value: Any | None,
    reference: Any,
    batch: Any,
    graph_count: int,
    name: str,
) -> Any:
    atom_count = int(batch.shape[0])
    if value is None:
        return reference.new_zeros(atom_count)
    tensor = _as_tensor(value, reference, name)
    if tensor.ndim == 0 or tuple(tensor.shape) == (1,):
        return tensor.reshape(1).expand(atom_count)
    if tuple(tensor.shape) == (graph_count,):
        return tensor[batch]
    if tuple(tensor.shape) == (atom_count,):
        return tensor
    raise ShapeError(f"{name} must be scalar or have shape [B] or [N]")


@dataclass(frozen=True, slots=True)
class _ExternalFields:
    potential: Any
    electric: Any
    gradient: Any
    magnetic: Any


def _resolve_external_fields(
    conditions: Conditions,
    *,
    reference: Any,
    batch: Any,
    graph_count: int,
    positions: Any | None,
) -> _ExternalFields:
    electric = _condition_tensor(
        conditions.external_electric_field,
        reference,
        batch,
        graph_count,
        (3,),
        "external_electric_field",
    )
    magnetic = _condition_tensor(
        conditions.external_magnetic_field,
        reference,
        batch,
        graph_count,
        (3,),
        "external_magnetic_field",
    )
    gradient = _condition_tensor(
        conditions.electric_field_gradient,
        reference,
        batch,
        graph_count,
        (3, 3),
        "electric_field_gradient",
    )
    potential = _condition_scalar(
        conditions.electric_potential,
        reference,
        batch,
        graph_count,
        "electric_potential",
    )
    if conditions.external_electric_field is not None:
        if positions is None:
            raise ZIVARValidationError(
                "positions are required for charge coupling to an external electric field"
            )
        origin = _condition_tensor(
            conditions.electric_field_origin,
            reference,
            batch,
            graph_count,
            (3,),
            "electric_field_origin",
        )
        potential = potential - ((positions - origin) * electric).sum(-1)
    fields = _ExternalFields(potential, electric, gradient, magnetic)
    _ensure_finite(tuple((name, getattr(fields, name)) for name in fields.__dataclass_fields__))
    return fields


def _resolve_spins(
    explicit: Any | None,
    conditions: Conditions,
    state: ElectronicState,
) -> Any:
    value = explicit if explicit is not None else conditions.external_spins
    spins = _atom_vector(value, state.q, state.atom_count, "external_spins")
    _ensure_finite((("external_spins", spins),))
    return spins


def _validate_state(state: ElectronicState, *, stf_atol: float | None) -> None:
    if not isinstance(state, ElectronicState):
        raise TypeError("state must be ElectronicState")
    try:
        state.assert_stf(atol=stf_atol)
    except ValueError as exc:
        raise SymmetryError("electronic Q must be symmetric and traceless") from exc
    _ensure_finite((("q", state.q), ("p", state.p), ("Q", state.Q), ("m", state.m)))


def _operator_response(
    operator: QuadraticOperatorLike | None,
    state: ElectronicState,
    *,
    batch: Any,
    conditions: Conditions,
    stf_atol: float | None,
) -> ElectronicState | None:
    if operator is None:
        return None
    if getattr(operator, "symmetric", None) is not True:
        raise NonPositiveDefiniteError("quadratic operator must declare symmetric=True")
    response = operator.matvec(state, batch=batch, conditions=conditions)
    _validate_state(response, stf_atol=stf_atol)
    if response.atom_count != state.atom_count:
        raise ShapeError("quadratic operator changed the atom count")
    if response.q.device != state.q.device or response.q.dtype != state.q.dtype:
        raise ShapeError("quadratic operator response must preserve device and dtype")
    return response


@dataclass(frozen=True, slots=True)
class LinearConstraints(ConstraintOperatorLike):
    """Dense small-constraint interface over the structured electronic state.

    Constraint weights have leading shape ``[K,N]`` and physical trailing
    dimensions matching their state field.  The corresponding packed matrix
    has shape ``[K, 12*N]`` and can be passed directly to a constrained SCF
    solver.  This object stores only ``O(KN)`` values, never an ``N x N``
    interaction matrix.
    """

    target: Any
    q_weights: Any | None = None
    p_weights: Any | None = None
    Q_weights: Any | None = None
    m_weights: Any | None = None
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target.ndim != 1 or int(self.target.shape[0]) < 1:
            raise ConstraintError("constraint target must have shape [K] with K > 0")
        count = int(self.target.shape[0])
        if self.labels and len(self.labels) != count:
            raise ConstraintError("constraint labels must match the target length")
        if all(
            value is None
            for value in (self.q_weights, self.p_weights, self.Q_weights, self.m_weights)
        ):
            raise ConstraintError("at least one constraint weight tensor is required")

    @property
    def count(self) -> int:
        return int(self.target.shape[0])

    @classmethod
    def total_charge(cls, batch: Any, target: Any) -> LinearConstraints:
        """Construct exact per-graph constraints ``sum_i q_i = target_b``."""

        if batch.ndim != 1 or target.ndim != 1:
            raise ConstraintError("batch and total-charge target must be one-dimensional")
        graph_count = int(target.shape[0])
        if graph_count < 1 or int(batch.shape[0]) < 1:
            raise ConstraintError("total-charge constraints require atoms and graphs")
        if bool(torch.any((batch < 0) | (batch >= graph_count)).detach()):
            raise ConstraintError("batch index lies outside the target graph range")
        occupancy = torch.bincount(batch.to(torch.long), minlength=graph_count)
        if bool(torch.any(occupancy == 0).detach()):
            raise ConstraintError("every total-charge target graph must contain atoms")
        weights = torch.nn.functional.one_hot(
            batch.to(torch.long), num_classes=graph_count
        ).to(device=target.device, dtype=target.dtype)
        labels = tuple(f"total_charge[{index}]" for index in range(graph_count))
        return cls(target=target, q_weights=weights.T, labels=labels)

    def _validated_weights(self, state: ElectronicState) -> tuple[Any, Any, Any, Any]:
        expected = {
            "q_weights": (self.count, state.atom_count),
            "p_weights": (self.count, state.atom_count, 3),
            "Q_weights": (self.count, state.atom_count, 3, 3),
            "m_weights": (self.count, state.atom_count, 3),
        }
        zero_q = state.q.new_zeros(expected["q_weights"])
        zero_p = state.q.new_zeros(expected["p_weights"])
        zero_Q = state.q.new_zeros(expected["Q_weights"])
        zero_m = state.q.new_zeros(expected["m_weights"])
        output = []
        for name, zero in (
            ("q_weights", zero_q),
            ("p_weights", zero_p),
            ("Q_weights", zero_Q),
            ("m_weights", zero_m),
        ):
            value = getattr(self, name)
            if value is None:
                output.append(zero)
                continue
            tensor = _as_tensor(value, state.q, name)
            if tuple(tensor.shape) != expected[name]:
                raise ConstraintError(f"{name} must have shape {list(expected[name])}")
            output.append(tensor)
        try:
            assert_stf(output[2])
        except ValueError as exc:
            raise ConstraintError("Q constraint weights must be symmetric and traceless") from exc
        _ensure_finite(
            tuple(
                (name, value)
                for name, value in zip(
                    ("q_weights", "p_weights", "Q_weights", "m_weights"),
                    output,
                    strict=True,
                )
            )
        )
        return output[0], output[1], output[2], output[3]

    def residual(self, state: ElectronicState) -> Any:
        q_weight, p_weight, Q_weight, m_weight = self._validated_weights(state)
        target = _as_tensor(self.target, state.q, "constraint target")
        return (
            torch.einsum("kn,n->k", q_weight, state.q)
            + torch.einsum("kni,ni->k", p_weight, state.p)
            + torch.einsum("knij,nij->k", Q_weight, state.Q)
            + torch.einsum("kni,ni->k", m_weight, state.m)
            - target
        )

    def adjoint(self, multipliers: Any) -> ElectronicState:
        if multipliers.ndim != 1 or tuple(multipliers.shape) != (self.count,):
            raise ConstraintError("constraint multipliers must have shape [K]")
        atom_count = next(
            int(value.shape[1])
            for value in (self.q_weights, self.p_weights, self.Q_weights, self.m_weights)
            if value is not None
        )
        reference = multipliers
        prototype = ElectronicState(
            q=reference.new_zeros(atom_count),
            p=reference.new_zeros((atom_count, 3)),
            Q=reference.new_zeros((atom_count, 3, 3)),
            m=reference.new_zeros((atom_count, 3)),
        )
        q_weight, p_weight, Q_weight, m_weight = self._validated_weights(prototype)
        values = multipliers.to(device=prototype.q.device, dtype=prototype.q.dtype)
        return ElectronicState(
            q=torch.einsum("k,kn->n", values, q_weight),
            p=torch.einsum("k,kni->ni", values, p_weight),
            Q=torch.einsum("k,knij->nij", values, Q_weight),
            m=torch.einsum("k,kni->ni", values, m_weight),
        )

    def packed_matrix(self, state: ElectronicState) -> Any:
        """Return ``A`` with shape ``[K, 12*N]`` for a flattened SCF solve."""

        q_weight, p_weight, Q_weight, m_weight = self._validated_weights(state)
        packed_Q = stf_to_components(Q_weight)
        packed = torch.cat((q_weight[..., None], p_weight, packed_Q, m_weight), dim=-1)
        return packed.reshape(self.count, state.atom_count * ElectronicState.PACKED_WIDTH)


class ElectroSpinFunctional(nn.Module):
    """Evaluate the one quadratic ``q/p/Q/m`` energy.

    Strict convexity is established by the SCF solver's constrained-curvature
    gate for the assembled onsite plus interaction Hessian.
    """

    def __init__(
        self,
        *,
        positive_floor: float = 0.0,
        stf_atol: float | None = None,
        bohr_magneton_eV_per_T: float = 5.7883818060e-5,
    ) -> None:
        super().__init__()
        if positive_floor < 0.0 or bohr_magneton_eV_per_T <= 0.0:
            raise ValueError("functional scales and tolerances are invalid")
        self.positive_floor = float(positive_floor)
        self.stf_atol = stf_atol
        self.bohr_magneton_eV_per_T = float(bohr_magneton_eV_per_T)

    @staticmethod
    def _conditions(value: Conditions | Mapping[str, Any] | None) -> Conditions:
        if isinstance(value, Conditions):
            return value
        if value is None or isinstance(value, Mapping):
            return Conditions.from_mapping(value)
        raise TypeError("conditions must be Conditions, a mapping, or None")

    def quadratic_matvec(
        self,
        state: ElectronicState,
        parameters: ElectroSpinParameters,
        *,
        batch: Any,
        conditions: Conditions | Mapping[str, Any] | None = None,
        quadratic_operator: QuadraticOperatorLike | None = None,
    ) -> ElectronicState:
        """Return the Hessian-vector product ``(D+K) @ state``."""

        _validate_state(state, stf_atol=self.stf_atol)
        batch_index, _, _ = _resolve_geometry(batch, None, state)
        resolved = _resolve_parameters(
            parameters, state, positive_floor=self.positive_floor
        )
        condition_value = self._conditions(conditions)
        response = ElectronicState(
            q=resolved.hardness * state.q,
            p=resolved.inverse_polarizability[:, None] * state.p,
            Q=resolved.inverse_quadrupole_polarizability[:, None, None] * state.Q,
            m=resolved.inverse_magnetic_susceptibility[:, None] * state.m,
        )
        interaction = _operator_response(
            quadratic_operator,
            state,
            batch=batch_index,
            conditions=condition_value,
            stf_atol=self.stf_atol,
        )
        if interaction is None:
            return response
        return ElectronicState(
            response.q + interaction.q,
            response.p + interaction.p,
            response.Q + interaction.Q,
            response.m + interaction.m,
        )

    def stationarity_gradient(
        self,
        state: ElectronicState,
        parameters: ElectroSpinParameters,
        *,
        batch: Any,
        positions: Any | None = None,
        conditions: Conditions | Mapping[str, Any] | None = None,
        external_spins: Any | None = None,
        quadratic_operator: QuadraticOperatorLike | None = None,
    ) -> ElectronicState:
        """Return ``dE/d(q,p,Q,m)`` in the twelve-dimensional STF space."""

        _validate_state(state, stf_atol=self.stf_atol)
        batch_index, positions, graph_count = _resolve_geometry(batch, positions, state)
        resolved = _resolve_parameters(
            parameters, state, positive_floor=self.positive_floor
        )
        condition_value = self._conditions(conditions)
        fields = _resolve_external_fields(
            condition_value,
            reference=state.q,
            batch=batch_index,
            graph_count=graph_count,
            positions=positions,
        )
        spins = _resolve_spins(external_spins, condition_value, state)
        quadratic = self.quadratic_matvec(
            state,
            parameters,
            batch=batch_index,
            conditions=condition_value,
            quadratic_operator=quadratic_operator,
        )
        return ElectronicState(
            q=quadratic.q + resolved.electronegativity + fields.potential,
            p=quadratic.p - resolved.dipole_drive - fields.electric,
            Q=quadratic.Q
            - resolved.quadrupole_drive
            - 0.5 * project_stf(fields.gradient),
            m=quadratic.m
            - resolved.magnetic_drive
            - resolved.spin_coupling[:, None] * spins
            - self.bohr_magneton_eV_per_T * fields.magnetic,
        )

    def constraint_residual(
        self,
        state: ElectronicState,
        constraints: ConstraintOperatorLike,
    ) -> Any:
        """Evaluate an SCF-compatible linear constraint without solving it."""

        _validate_state(state, stf_atol=self.stf_atol)
        try:
            residual = constraints.residual(state)
        except AttributeError as exc:
            raise ConstraintError("constraints must implement residual(state)") from exc
        if residual.ndim != 1:
            raise ConstraintError("constraint residual must have shape [K]")
        _ensure_finite((("constraint residual", residual),))
        return residual

    def forward(
        self,
        state: ElectronicState,
        parameters: ElectroSpinParameters,
        *,
        batch: Any,
        positions: Any | None = None,
        conditions: Conditions | Mapping[str, Any] | None = None,
        external_spins: Any | None = None,
        short_range_energy: Any | None = None,
        external_spin_energy: Any | None = None,
        quadratic_operator: QuadraticOperatorLike | None = None,
    ) -> EnergyBreakdown:
        """Return every term and the authoritative per-graph total energy.

        ``short_range_energy`` and ``external_spin_energy`` may be scalars or
        ``[B]`` graph tensors.  They may depend on geometry and ``S`` upstream,
        but they must not depend on independently predicted forces or fields.
        """

        _validate_state(state, stf_atol=self.stf_atol)
        batch_index, positions, graph_count = _resolve_geometry(batch, positions, state)
        resolved = _resolve_parameters(
            parameters, state, positive_floor=self.positive_floor
        )
        condition_value = self._conditions(conditions)
        fields = _resolve_external_fields(
            condition_value,
            reference=state.q,
            batch=batch_index,
            graph_count=graph_count,
            positions=positions,
        )
        spins = _resolve_spins(external_spins, condition_value, state)
        interaction = _operator_response(
            quadratic_operator,
            state,
            batch=batch_index,
            conditions=condition_value,
            stf_atol=self.stf_atol,
        )

        charge_atomic = (
            resolved.electronegativity * state.q
            + 0.5 * resolved.hardness * state.q.square()
        )
        polar_atomic = (
            0.5
            * resolved.inverse_polarizability
            * state.p.square().sum(-1)
            - (resolved.dipole_drive * state.p).sum(-1)
        )
        quadrupole_atomic = (
            0.5
            * resolved.inverse_quadrupole_polarizability
            * state.Q.square().sum((-2, -1))
            - (resolved.quadrupole_drive * state.Q).sum((-2, -1))
        )
        magnetic_atomic = (
            0.5
            * resolved.inverse_magnetic_susceptibility
            * state.m.square().sum(-1)
            - (resolved.magnetic_drive * state.m).sum(-1)
        )
        if interaction is not None:
            interaction_channels = (
                0.5 * state.q * interaction.q,
                0.5 * (state.p * interaction.p).sum(-1),
                0.5 * (state.Q * interaction.Q).sum((-2, -1)),
                0.5 * (state.m * interaction.m).sum(-1),
            )
            charge_atomic = charge_atomic + interaction_channels[0]
            polar_atomic = polar_atomic + interaction_channels[1]
            quadrupole_atomic = quadrupole_atomic + interaction_channels[2]
            magnetic_atomic = magnetic_atomic + interaction_channels[3]

        coupling_atomic = -resolved.spin_coupling * (spins * state.m).sum(-1)
        external_atomic = (
            state.q * fields.potential
            - (state.p * fields.electric).sum(-1)
            - 0.5 * (state.Q * fields.gradient).sum((-2, -1))
            - self.bohr_magneton_eV_per_T
            * ((spins + state.m) * fields.magnetic).sum(-1)
        )

        short_range = _graph_energy(
            short_range_energy, state.q, graph_count, "short_range_energy"
        ) + _scatter_sum(resolved.reference_atomic_energy, batch_index, graph_count)
        external_spin = _graph_energy(
            external_spin_energy, state.q, graph_count, "external_spin_energy"
        )
        charge = _scatter_sum(charge_atomic, batch_index, graph_count)
        polarization = _scatter_sum(polar_atomic, batch_index, graph_count)
        quadrupole = _scatter_sum(quadrupole_atomic, batch_index, graph_count)
        magnetic = _scatter_sum(magnetic_atomic, batch_index, graph_count)
        coupling = _scatter_sum(coupling_atomic, batch_index, graph_count)
        external = _scatter_sum(external_atomic, batch_index, graph_count)
        total = (
            short_range
            + external_spin
            + charge
            + polarization
            + quadrupole
            + magnetic
            + coupling
            + external
        )
        _ensure_finite(
            (
                ("short_range energy", short_range),
                ("external spin energy", external_spin),
                ("charge energy", charge),
                ("polarization energy", polarization),
                ("quadrupole energy", quadrupole),
                ("magnetic energy", magnetic),
                ("coupling energy", coupling),
                ("external energy", external),
                ("total energy", total),
            )
        )
        return EnergyBreakdown(
            short_range=short_range,
            external_spin=external_spin,
            charge=charge,
            polarization=polarization,
            quadrupole=quadrupole,
            magnetic=magnetic,
            coupling=coupling,
            external=external,
            total=total,
        )

    def energy(self, *args: Any, **kwargs: Any) -> Any:
        """Return the one scalar energy used for all variational derivatives."""

        return self.forward(*args, **kwargs).scalar


def _validate_o3(rotation: Any, reference: Any) -> tuple[Any, Any]:
    matrix = _as_tensor(rotation, reference, "rotation")
    if tuple(matrix.shape) != (3, 3):
        raise ShapeError("rotation must have shape [3,3]")
    _ensure_finite((("rotation", matrix),))
    tolerance = max(1.0e-12, 128.0 * float(torch.finfo(matrix.dtype).eps))
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    if not torch.allclose(matrix @ matrix.T, identity, atol=tolerance, rtol=tolerance):
        raise SymmetryError("rotation must be an orthogonal O(3) matrix")
    determinant = torch.linalg.det(matrix)
    if not torch.allclose(
        determinant.abs(), determinant.new_ones(()), atol=tolerance, rtol=tolerance
    ):
        raise SymmetryError("rotation determinant must be +1 or -1")
    return matrix, determinant


def transform_polar_vector(vector: Any, rotation: Any) -> Any:
    """Act on a polar vector by a proper or improper O(3) transform."""

    matrix, _ = _validate_o3(rotation, vector)
    if vector.shape[-1] != 3:
        raise ShapeError("polar vectors must have shape [...,3]")
    return torch.einsum("ij,...j->...i", matrix, vector)


def transform_axial_vector(
    vector: Any,
    rotation: Any,
    *,
    time_reversal: bool = False,
) -> Any:
    """Act on an axial vector; time reversal contributes an additional minus."""

    matrix, determinant = _validate_o3(rotation, vector)
    if vector.shape[-1] != 3:
        raise ShapeError("axial vectors must have shape [...,3]")
    sign = -1.0 if time_reversal else 1.0
    return sign * determinant * torch.einsum("ij,...j->...i", matrix, vector)


def transform_rank2(tensor: Any, rotation: Any) -> Any:
    """Act on a Cartesian polar rank-two tensor as ``R Q R.T``."""

    matrix, _ = _validate_o3(rotation, tensor)
    if tensor.shape[-2:] != (3, 3):
        raise ShapeError("rank-two tensors must have shape [...,3,3]")
    return torch.einsum("ia,...ab,jb->...ij", matrix, tensor, matrix)


def transform_electronic_state(
    state: ElectronicState,
    rotation: Any,
    *,
    time_reversal: bool = False,
) -> ElectronicState:
    """Transform ``q/p/Q/m`` using their O(3) and time-reversal characters."""

    return ElectronicState(
        q=state.q,
        p=transform_polar_vector(state.p, rotation),
        Q=project_stf(transform_rank2(state.Q, rotation)),
        m=transform_axial_vector(
            state.m, rotation, time_reversal=time_reversal
        ),
    )


def transform_external_spins(
    spins: Any,
    rotation: Any,
    *,
    time_reversal: bool = False,
) -> Any:
    """Transform the external dynamical spin ``S`` as a time-odd axial vector."""

    return transform_axial_vector(spins, rotation, time_reversal=time_reversal)


def transform_parameters(
    parameters: ElectroSpinParameters,
    rotation: Any,
    *,
    time_reversal: bool = False,
) -> ElectroSpinParameters:
    """Transform every non-scalar local drive consistently with the state."""

    dipole = (
        None
        if parameters.dipole_drive is None
        else transform_polar_vector(parameters.dipole_drive, rotation)
    )
    quadrupole = (
        None
        if parameters.quadrupole_drive is None
        else project_stf(transform_rank2(parameters.quadrupole_drive, rotation))
    )
    magnetic = (
        None
        if parameters.magnetic_drive is None
        else transform_axial_vector(
            parameters.magnetic_drive,
            rotation,
            time_reversal=time_reversal,
        )
    )
    return replace(
        parameters,
        dipole_drive=dipole,
        quadrupole_drive=quadrupole,
        magnetic_drive=magnetic,
    )


def transform_conditions(
    conditions: Conditions,
    rotation: Any,
    *,
    time_reversal: bool = False,
) -> Conditions:
    """Transform the typed external fields and spins; scalar conditions remain."""

    def polar(value: Any | None) -> Any | None:
        return None if value is None else transform_polar_vector(value, rotation)

    def axial(value: Any | None) -> Any | None:
        return (
            None
            if value is None
            else transform_axial_vector(value, rotation, time_reversal=time_reversal)
        )

    return replace(
        conditions,
        external_spins=axial(conditions.external_spins),
        external_electric_field=polar(conditions.external_electric_field),
        external_magnetic_field=axial(conditions.external_magnetic_field),
        electric_field_gradient=(
            None
            if conditions.electric_field_gradient is None
            else transform_rank2(conditions.electric_field_gradient, rotation)
        ),
        electric_field_origin=polar(conditions.electric_field_origin),
        total_magnetization=axial(conditions.total_magnetization),
    )


__all__ = [
    "ElectroSpinFunctional",
    "ElectroSpinParameters",
    "LinearConstraints",
    "QuadraticOperatorLike",
    "transform_axial_vector",
    "transform_conditions",
    "transform_electronic_state",
    "transform_external_spins",
    "transform_parameters",
    "transform_polar_vector",
    "transform_rank2",
]
