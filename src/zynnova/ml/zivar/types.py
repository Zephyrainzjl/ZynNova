"""Typed public data contracts for the unified ZIVAR electro-spin core.

The electronic variables use Cartesian tensors with explicit physical
transformation rules:

``q``
    A time-even scalar charge, shape ``[N]``.
``p``
    A time-even polar dipole vector, shape ``[N, 3]``.
``Q``
    A time-even symmetric-traceless (STF) polar rank-two tensor, shape
    ``[N, 3, 3]``.  Only its five independent orthonormal components are
    packed for an SCF solve.
``m``
    A time-odd axial induced magnetic moment, shape ``[N, 3]``.

The externally supplied dynamical spin ``S`` is deliberately stored in
:class:`Conditions`, not :class:`ElectronicState`: ``S`` is an external
state variable while ``m`` is an induced variational degree of freedom.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

Tensor = Any


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError) as exc:
        raise TypeError("ZIVAR tensor fields must expose a shape") from exc


def _require_shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    actual = _shape(value)
    if actual != expected:
        raise ValueError(f"{name} must have shape {list(expected)}, got {list(actual)}")


def project_stf(tensor: Tensor) -> Tensor:
    """Project Cartesian rank-two tensors onto the symmetric-traceless space."""

    if len(_shape(tensor)) < 2 or _shape(tensor)[-2:] != (3, 3):
        raise ValueError("STF projection requires shape [...,3,3]")
    torch = __import__("torch")
    symmetric = 0.5 * (tensor + tensor.transpose(-1, -2))
    trace = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, device=tensor.device, dtype=tensor.dtype)
    return symmetric - trace[..., None, None] * identity / 3.0


def _rank2_stf_basis(reference: Tensor) -> Tensor:
    """Return an orthonormal Cartesian basis for the five rank-two STF modes."""

    root2 = 2.0**0.5
    root6 = 6.0**0.5
    return reference.new_tensor(
        (
            ((1.0 / root2, 0.0, 0.0), (0.0, -1.0 / root2, 0.0), (0.0, 0.0, 0.0)),
            ((1.0 / root6, 0.0, 0.0), (0.0, 1.0 / root6, 0.0), (0.0, 0.0, -2.0 / root6)),
            ((0.0, 1.0 / root2, 0.0), (1.0 / root2, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 1.0 / root2), (0.0, 0.0, 0.0), (1.0 / root2, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0 / root2), (0.0, 1.0 / root2, 0.0)),
        )
    ).to(device=reference.device, dtype=reference.dtype)


def stf_to_components(tensor: Tensor, *, validate: bool = True) -> Tensor:
    """Pack ``[...,3,3]`` STF tensors into five orthonormal components."""

    if len(_shape(tensor)) < 2 or _shape(tensor)[-2:] != (3, 3):
        raise ValueError("STF component packing requires shape [...,3,3]")
    if validate:
        assert_stf(tensor)
    torch = __import__("torch")
    return torch.einsum("...ij,aij->...a", tensor, _rank2_stf_basis(tensor))


def components_to_stf(components: Tensor) -> Tensor:
    """Unpack five orthonormal components into a Cartesian STF tensor."""

    if not _shape(components) or _shape(components)[-1] != 5:
        raise ValueError("STF component unpacking requires shape [...,5]")
    torch = __import__("torch")
    return torch.einsum(
        "...a,aij->...ij", components, _rank2_stf_basis(components)
    )


def assert_stf(tensor: Tensor, *, atol: float | None = None, rtol: float = 0.0) -> None:
    """Raise if a Cartesian tensor is not symmetric and traceless."""

    if len(_shape(tensor)) < 2 or _shape(tensor)[-2:] != (3, 3):
        raise ValueError("STF validation requires shape [...,3,3]")
    torch = __import__("torch")
    tolerance = (
        float(atol)
        if atol is not None
        else 256.0 * float(torch.finfo(tensor.dtype).eps)
    )
    detached = tensor.detach()
    symmetric = torch.allclose(
        detached, detached.transpose(-1, -2), atol=tolerance, rtol=float(rtol)
    )
    trace = torch.diagonal(detached, dim1=-2, dim2=-1).sum(-1)
    traceless = torch.allclose(
        trace, torch.zeros_like(trace), atol=tolerance, rtol=float(rtol)
    )
    if not symmetric or not traceless:
        raise ValueError("Q must be a symmetric-traceless tensor")


@dataclass(frozen=True, slots=True)
class ZIVARBatch:
    """Backbone-neutral atomistic batch used by every ZIVAR execution path."""

    positions: Tensor
    atomic_numbers: Tensor
    batch: Tensor
    edge_index: Tensor
    shifts: Tensor | None = None
    cell: Tensor | None = None
    pbc: Tensor | None = None
    node_attrs: Tensor | None = None
    unit_shifts: Tensor | None = None
    ptr: Tensor | None = None
    head: Tensor | None = None

    def __post_init__(self) -> None:
        positions = _shape(self.positions)
        if len(positions) != 2 or positions[1] != 3:
            raise ValueError("positions must have shape [N,3]")
        atom_count = positions[0]
        _require_shape(self.atomic_numbers, (atom_count,), "atomic_numbers")
        _require_shape(self.batch, (atom_count,), "batch")
        edges = _shape(self.edge_index)
        if len(edges) != 2 or edges[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = edges[1]
        if self.shifts is not None:
            _require_shape(self.shifts, (edge_count, 3), "shifts")
        if self.unit_shifts is not None:
            _require_shape(self.unit_shifts, (edge_count, 3), "unit_shifts")
        if self.cell is not None:
            cell = _shape(self.cell)
            if len(cell) != 3 or cell[-2:] != (3, 3):
                raise ValueError("cell must have shape [B,3,3]")
        if self.pbc is not None:
            periodic = _shape(self.pbc)
            if len(periodic) != 2 or periodic[-1] != 3:
                raise ValueError("pbc must have shape [B,3]")
            if self.cell is not None and periodic[0] != _shape(self.cell)[0]:
                raise ValueError("cell and pbc graph counts differ")
        if self.node_attrs is not None:
            attrs = _shape(self.node_attrs)
            if len(attrs) != 2 or attrs[0] != atom_count:
                raise ValueError("node_attrs must have shape [N,F]")
        if self.ptr is not None:
            pointer = _shape(self.ptr)
            if len(pointer) != 1 or pointer[0] < 2:
                raise ValueError("ptr must have shape [B+1]")
        if self.head is not None:
            _require_shape(self.head, (self.graph_count,), "head")

    @property
    def atom_count(self) -> int:
        return _shape(self.positions)[0]

    @property
    def edge_count(self) -> int:
        return _shape(self.edge_index)[1]

    @property
    def graph_count(self) -> int:
        if self.ptr is not None:
            return _shape(self.ptr)[0] - 1
        if self.cell is not None:
            return _shape(self.cell)[0]
        if self.pbc is not None:
            return _shape(self.pbc)[0]
        return int(self.batch.max().detach().item()) + 1 if self.atom_count else 0

    def as_dict(self) -> dict[str, Tensor | None]:
        """Return the legacy mapping surface without copying tensor storage."""

        return {
            "positions": self.positions,
            "atomic_numbers": self.atomic_numbers,
            "batch": self.batch,
            "edge_index": self.edge_index,
            "shifts": self.shifts,
            "cell": self.cell,
            "pbc": self.pbc,
            "node_attrs": self.node_attrs,
            "unit_shifts": self.unit_shifts,
            "ptr": self.ptr,
            "head": self.head,
        }


@dataclass(frozen=True, slots=True)
class Conditions:
    """Physical boundary conditions, including the external spin state ``S``.

    Positions are measured in angstrom, ``external_electric_field`` in
    V/angstrom, its gradient in V/angstrom squared, electric potential in V,
    and ``external_magnetic_field`` in tesla.  Spins and magnetic moments are
    measured in Bohr magnetons.
    """

    total_charge: Tensor | None = None
    external_spins: Tensor | None = None
    external_electric_field: Tensor | None = None
    external_magnetic_field: Tensor | None = None
    electric_field_gradient: Tensor | None = None
    electric_potential: Tensor | None = None
    electric_field_origin: Tensor | None = None
    total_magnetization: Tensor | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def spin_vectors(self) -> Tensor | None:
        """Compatibility alias for the existing model condition name."""

        return self.external_spins

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> Conditions:
        if values is None:
            return cls()
        known = {
            "total_charge",
            "external_spins",
            "spin_vectors",
            "external_electric_field",
            "external_magnetic_field",
            "electric_field_gradient",
            "external_electric_field_gradient",
            "electric_potential",
            "external_electric_potential",
            "electric_field_origin",
            "total_magnetization",
        }
        return cls(
            total_charge=values.get("total_charge"),
            external_spins=values.get("external_spins", values.get("spin_vectors")),
            external_electric_field=values.get("external_electric_field"),
            external_magnetic_field=values.get("external_magnetic_field"),
            electric_field_gradient=values.get(
                "electric_field_gradient", values.get("external_electric_field_gradient")
            ),
            electric_potential=values.get(
                "electric_potential", values.get("external_electric_potential")
            ),
            electric_field_origin=values.get("electric_field_origin"),
            total_magnetization=values.get("total_magnetization"),
            extras={key: value for key, value in values.items() if key not in known},
        )

    def as_dict(self) -> dict[str, Any]:
        values = dict(self.extras)
        for name, value in (
            ("total_charge", self.total_charge),
            ("spin_vectors", self.external_spins),
            ("external_electric_field", self.external_electric_field),
            ("external_magnetic_field", self.external_magnetic_field),
            ("electric_field_gradient", self.electric_field_gradient),
            ("electric_potential", self.electric_potential),
            ("electric_field_origin", self.electric_field_origin),
            ("total_magnetization", self.total_magnetization),
        ):
            if value is not None:
                values[name] = value
        return values

    def get(self, name: str, default: Any = None) -> Any:
        return self.as_dict().get(name, default)


@dataclass(frozen=True, slots=True)
class Targets:
    """Optional supervised observables with unambiguous physical semantics."""

    energy: Tensor | None = None
    forces: Tensor | None = None
    stress: Tensor | None = None
    charges: Tensor | None = None
    dipoles: Tensor | None = None
    quadrupoles: Tensor | None = None
    induced_moments: Tensor | None = None
    scalar_moments: Tensor | None = None
    external_spins: Tensor | None = None
    effective_fields: Tensor | None = None
    torques: Tensor | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.induced_moments is not None and self.scalar_moments is not None:
            raise ValueError(
                "vector induced-moment and legacy scalar-moment supervision "
                "cannot be enabled in the same target contract"
            )

    def as_dict(self) -> dict[str, Any]:
        values = dict(self.extras)
        for name, value in (
            ("energy", self.energy),
            ("forces", self.forces),
            ("stress", self.stress),
            ("charges", self.charges),
            ("dipoles", self.dipoles),
            ("quadrupoles", self.quadrupoles),
            ("magmom_vectors", self.induced_moments),
            ("magmoms", self.scalar_moments),
            ("spin_vectors", self.external_spins),
            ("effective_field_T", self.effective_fields),
            ("magnetic_torque_eV", self.torques),
        ):
            if value is not None:
                values[name] = value
        return values


@dataclass(frozen=True, slots=True)
class ElectronicState:
    """Variational electronic state ``x=(q,p,Q,m)``.

    ``pack`` uses twelve independent values per atom in the fixed order
    ``q(1), p(3), Q_STF(5), m(3)``.  The five STF modes use an orthonormal
    Cartesian basis, so their Euclidean norm equals ``Q:Q``.
    """

    q: Tensor
    p: Tensor
    Q: Tensor
    m: Tensor

    PACKED_WIDTH: ClassVar[int] = 12

    def __post_init__(self) -> None:
        q_shape = _shape(self.q)
        if len(q_shape) != 1:
            raise ValueError("q must have shape [N]")
        atom_count = q_shape[0]
        _require_shape(self.p, (atom_count, 3), "p")
        _require_shape(self.Q, (atom_count, 3, 3), "Q")
        _require_shape(self.m, (atom_count, 3), "m")
        devices = {str(value.device) for value in (self.q, self.p, self.Q, self.m)}
        dtypes = {value.dtype for value in (self.q, self.p, self.Q, self.m)}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("q, p, Q and m must share device and dtype")
        if not all(value.is_floating_point() for value in (self.q, self.p, self.Q, self.m)):
            raise TypeError("q, p, Q and m must use a floating dtype")

    @property
    def atom_count(self) -> int:
        return _shape(self.q)[0]

    @property
    def charges(self) -> Tensor:
        return self.q

    @property
    def dipoles(self) -> Tensor:
        return self.p

    @property
    def quadrupoles(self) -> Tensor:
        return self.Q

    @property
    def induced_moments(self) -> Tensor:
        return self.m

    def assert_stf(self, *, atol: float | None = None, rtol: float = 0.0) -> None:
        assert_stf(self.Q, atol=atol, rtol=rtol)

    def projected(self) -> ElectronicState:
        """Return the same state with ``Q`` projected exactly onto its STF space."""

        return ElectronicState(self.q, self.p, project_stf(self.Q), self.m)

    def pack(self, *, validate_stf: bool = True) -> Tensor:
        """Pack the independent SCF variables to shape ``[N,12]``."""

        torch = __import__("torch")
        quadrupole = stf_to_components(self.Q, validate=validate_stf)
        return torch.cat((self.q[:, None], self.p, quadrupole, self.m), dim=-1)

    def flatten(self, *, validate_stf: bool = True) -> Tensor:
        """Pack the state into the atom-major one-dimensional SCF vector."""

        return self.pack(validate_stf=validate_stf).reshape(-1)

    @classmethod
    def from_packed(cls, packed: Tensor) -> ElectronicState:
        """Construct a state from the independent ``[N,12]`` SCF vector."""

        shape = _shape(packed)
        if len(shape) != 2 or shape[1] != cls.PACKED_WIDTH:
            raise ValueError(f"packed electronic state must have shape [N,{cls.PACKED_WIDTH}]")
        return cls(
            q=packed[:, 0],
            p=packed[:, 1:4],
            Q=components_to_stf(packed[:, 4:9]),
            m=packed[:, 9:12],
        )

    @classmethod
    def from_flattened(cls, flattened: Tensor, *, atom_count: int) -> ElectronicState:
        """Invert :meth:`flatten` for a known number of atoms."""

        if atom_count < 0 or _shape(flattened) != (atom_count * cls.PACKED_WIDTH,):
            raise ValueError(
                f"flattened electronic state must have shape [{atom_count * cls.PACKED_WIDTH}]"
            )
        return cls.from_packed(flattened.reshape(atom_count, cls.PACKED_WIDTH))


@runtime_checkable
class SCFReportLike(Protocol):
    """Structural protocol shared with the concrete SCF implementation."""

    converged: bool
    iterations: int
    final_residual: float
    energy_change: float
    energy_error: float
    termination: str


@runtime_checkable
class ConstraintOperatorLike(Protocol):
    """Matrix-free linear constraint ``A x = b`` used by an SCF solver."""

    def residual(self, state: ElectronicState) -> Tensor:
        """Return ``A x - b``."""

    def adjoint(self, multipliers: Tensor) -> ElectronicState:
        """Return ``A.T @ multipliers`` in structured state form."""


@dataclass(frozen=True, slots=True)
class EnergyBreakdown:
    """Per-graph terms whose sum is the one authoritative total energy."""

    short_range: Tensor
    external_spin: Tensor
    charge: Tensor
    polarization: Tensor
    quadrupole: Tensor
    magnetic: Tensor
    coupling: Tensor
    external: Tensor
    total: Tensor

    def __post_init__(self) -> None:
        expected = _shape(self.total)
        if len(expected) != 1:
            raise ValueError("energy breakdown terms must have shape [B]")
        for name in (
            "short_range",
            "external_spin",
            "charge",
            "polarization",
            "quadrupole",
            "magnetic",
            "coupling",
            "external",
        ):
            _require_shape(getattr(self, name), expected, name)
        torch = __import__("torch")
        reconstructed = self.reconstructed_total().detach()
        total = self.total.detach()
        tolerance = 256.0 * float(torch.finfo(total.dtype).eps)
        if not torch.allclose(total, reconstructed, atol=tolerance, rtol=tolerance):
            raise ValueError("energy breakdown total does not equal the sum of its terms")

    @property
    def scalar(self) -> Tensor:
        """Scalar energy summed over independent structures in the batch."""

        return self.total.sum()

    def reconstructed_total(self) -> Tensor:
        return (
            self.short_range
            + self.external_spin
            + self.charge
            + self.polarization
            + self.quadrupole
            + self.magnetic
            + self.coupling
            + self.external
        )


@dataclass(frozen=True, slots=True)
class ZIVARPrediction:
    """Typed result shared by training, ASE, export and deployment wrappers."""

    energy: Tensor
    state: ElectronicState
    breakdown: EnergyBreakdown
    atomic_energy: Tensor | None = None
    forces: Tensor | None = None
    stress: Tensor | None = None
    virial: Tensor | None = None
    effective_field: Tensor | None = None
    torque: Tensor | None = None
    scf_report: SCFReportLike | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _shape(self.energy) != _shape(self.breakdown.total):
            raise ValueError("prediction energy and breakdown total shapes differ")
        torch = __import__("torch")
        energy = self.energy.detach()
        total = self.breakdown.total.detach()
        tolerance = 256.0 * float(torch.finfo(total.dtype).eps)
        if not torch.allclose(energy, total, atol=tolerance, rtol=tolerance):
            raise ValueError("prediction energy differs from its authoritative breakdown")


__all__ = [
    "Conditions",
    "ConstraintOperatorLike",
    "ElectronicState",
    "EnergyBreakdown",
    "SCFReportLike",
    "Targets",
    "Tensor",
    "ZIVARBatch",
    "ZIVARPrediction",
    "assert_stf",
    "components_to_stf",
    "project_stf",
    "stf_to_components",
]
