"""Stable local-backbone contract and registry for ZIVAR.

Backbones own only the short-range atomistic representation and energy.  The
    stable electronic-density model, electrostatics and all public
materials workflows remain outside this package.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import threading
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .._deps import require_torch

torch = require_torch()
nn = torch.nn

BACKBONE_CONTRACT_VERSION = "zivar-local-backbone-1"
_KIND = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@dataclass(frozen=True, slots=True)
class BackboneCapabilities:
    """Auditable physical and deployment surface exposed by one adapter."""

    e3_equivariant: bool = True
    invariant_energy: bool = True
    invariant_node_features: bool = True
    conservative_forces: bool = True
    differentiable_stress: bool = True
    periodic_boundaries: bool = True
    second_order_training: bool = True
    maximum_ell: int = 4
    global_lammps: bool = True
    local_mliap: bool = False
    torch_compile: bool = True

    def validate_required(self) -> None:
        required = (
            self.e3_equivariant,
            self.invariant_energy,
            self.invariant_node_features,
            self.conservative_forces,
            self.differentiable_stress,
            self.periodic_boundaries,
            self.second_order_training,
            self.global_lammps,
        )
        if not all(required):
            raise ValueError("a registered ZIVAR backbone lacks a required capability")
        if not 0 <= self.maximum_ell <= 4:
            raise ValueError("maximum_ell must lie in [0, 4]")


@dataclass(frozen=True, slots=True)
class BackboneManifest:
    """Checkpoint identity; architecture selection is immutable after training."""

    contract: str
    kind: str
    architecture: str
    implementation: str
    invariant_dim: int
    atomic_numbers: tuple[int, ...]
    cutoff_A: float
    capabilities: BackboneCapabilities

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BackboneAdapter(nn.Module):
    """Base class every local implementation must satisfy.

    ``forward`` must return graph energy ``[B]``, optional atomic energy
    ``[N]``, and reflection/rotation invariant node features ``[N,F]``.
    """

    def __init__(
        self,
        model: Any,
        *,
        kind: str,
        architecture: str,
        implementation: str,
        invariant_dim: int,
        atomic_numbers: tuple[int, ...],
        cutoff_A: float,
        capabilities: BackboneCapabilities,
    ) -> None:
        super().__init__()
        _validate_kind(kind)
        if invariant_dim < 1:
            raise ValueError("invariant_dim must be positive")
        capabilities.validate_required()
        self.model = model
        self._kind = kind
        self._architecture = architecture
        self._implementation = implementation
        self._invariant_dim = int(invariant_dim)
        self._atomic_numbers = tuple(int(value) for value in atomic_numbers)
        self._cutoff_A = float(cutoff_A)
        self._capabilities = capabilities

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def architecture(self) -> str:
        return self._architecture

    @property
    def invariant_dim(self) -> int:
        return self._invariant_dim

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return self._atomic_numbers

    @property
    def cutoff_A(self) -> float:
        return self._cutoff_A

    @property
    def capabilities(self) -> BackboneCapabilities:
        return self._capabilities

    @property
    def execution_backend(self) -> str:
        """Actual local execution path, separate from checkpoint architecture."""

        value = getattr(self.model, "zivar_acceleration_backend", None)
        return str(value) if value is not None else "native"

    @property
    def manifest(self) -> BackboneManifest:
        return BackboneManifest(
            contract=BACKBONE_CONTRACT_VERSION,
            kind=self.kind,
            architecture=self.architecture,
            implementation=self._implementation,
            invariant_dim=self.invariant_dim,
            atomic_numbers=self.atomic_numbers,
            cutoff_A=self.cutoff_A,
            capabilities=self.capabilities,
        )

    def compile_model(self, mode: str) -> BackboneAdapter:
        if mode != "none" and not hasattr(self.model, "_orig_mod"):
            self.model = torch.compile(self.model, mode=mode)
        return self

    @abstractmethod
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


BackboneBuilder = Callable[..., BackboneAdapter]


@dataclass(frozen=True, slots=True)
class BackboneRegistration:
    kind: str
    builder: BackboneBuilder
    description: str
    provenance: str


_REGISTRY: dict[str, BackboneRegistration] = {}
_LOCK = threading.RLock()


def _validate_kind(kind: str) -> str:
    if not isinstance(kind, str) or _KIND.fullmatch(kind) is None:
        raise ValueError(
            "backbone kind must match ^[a-z][a-z0-9_]{0,31}$"
        )
    return kind


def register_backbone(
    kind: str,
    builder: BackboneBuilder,
    *,
    description: str,
    provenance: str,
    replace: bool = False,
) -> None:
    """Register a builder without importing it into any other model family."""

    kind = _validate_kind(kind)
    if not callable(builder):
        raise TypeError("backbone builder must be callable")
    registration = BackboneRegistration(kind, builder, description, provenance)
    with _LOCK:
        if kind in _REGISTRY and not replace:
            raise KeyError(f"backbone {kind!r} is already registered")
        _REGISTRY[kind] = registration


def registered_backbones() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_REGISTRY))


def backbone_registration(kind: str) -> BackboneRegistration:
    with _LOCK:
        try:
            return _REGISTRY[kind]
        except KeyError as exc:
            choices = ", ".join(sorted(_REGISTRY)) or "<none>"
            raise KeyError(f"unknown backbone {kind!r}; available: {choices}") from exc


def load_backbone_plugins(
    *, group: str = "zynnova.zivar_backbones"
) -> tuple[str, ...]:
    """Load explicitly installed third-party registrations from entry points."""

    before = set(registered_backbones())
    discovered = importlib.metadata.entry_points()
    entries = (
        discovered.select(group=group)
        if hasattr(discovered, "select")
        else discovered.get(group, ())
    )
    for entry in entries:
        hook = entry.load()
        if not callable(hook):
            raise TypeError(f"backbone entry point {entry.name!r} is not callable")
        hook()
    return tuple(sorted(set(registered_backbones()) - before))


def build_backbone(config: Any, *, device: Any = "cpu") -> BackboneAdapter:
    registration = backbone_registration(config.kind)
    adapter = registration.builder(config, device=device)
    if not isinstance(adapter, BackboneAdapter):
        raise TypeError("registered builder did not return a BackboneAdapter")
    if adapter.kind != config.kind:
        raise ValueError("builder returned an adapter with a different kind")
    if adapter.atomic_numbers != tuple(config.atomic_numbers):
        raise ValueError("adapter atomic-number table differs from its configuration")
    if abs(adapter.cutoff_A - float(config.cutoff_A)) > 1.0e-12:
        raise ValueError("adapter cutoff differs from its configuration")
    if config.max_ell > adapter.capabilities.maximum_ell:
        raise ValueError("requested max_ell exceeds the adapter capability")
    return adapter


def validate_backbone_output(
    adapter: BackboneAdapter,
    data: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    require_finite: bool = True,
) -> None:
    """Strict contract assertion used by tests and third-party registrations."""

    if not isinstance(output, Mapping):
        raise TypeError("backbone output must be a mapping")
    atom_count = int(data["positions"].shape[0])
    batch = data["batch"]
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    energy = output.get("energy")
    features = output.get("invariant_features")
    if energy is None or energy.reshape(-1).shape != (graph_count,):
        raise ValueError("backbone energy must have shape [B]")
    if features is None or features.shape != (atom_count, adapter.invariant_dim):
        raise ValueError("backbone invariant_features must have shape [N,F]")
    node_energy = output.get("node_energy")
    if node_energy is not None and node_energy.reshape(-1).shape != (atom_count,):
        raise ValueError("backbone node_energy must have shape [N]")
    if require_finite:
        tensors = [energy, features]
        if node_energy is not None:
            tensors.append(node_energy)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("backbone output contains non-finite values")


__all__ = [
    "BACKBONE_CONTRACT_VERSION",
    "BackboneAdapter",
    "BackboneCapabilities",
    "BackboneManifest",
    "BackboneRegistration",
    "backbone_registration",
    "build_backbone",
    "load_backbone_plugins",
    "register_backbone",
    "registered_backbones",
    "validate_backbone_output",
]
