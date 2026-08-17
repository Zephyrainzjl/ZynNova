from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chemistry import MolecularGraph
from .enums import ArchitectureType, EdgeKind, UnitRole
from .process import ProcessHistory
from .spatial import SpatialState
from .statistics import EnsembleStatistics


@dataclass
class PolymerUnit:
    id: str
    role: UnitRole
    graph: MolecularGraph
    name: str | None = None
    aliases: list[str] = field(default_factory=list)
    descriptors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("polymer unit id cannot be empty")
        self.graph.validate()


@dataclass(slots=True)
class ArchitectureNode:
    id: str
    unit_id: str
    occurrence: int | None = None
    role: str | None = None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArchitectureEdge:
    source: str
    target: str
    source_port: str | None = None
    target_port: str | None = None
    kind: EdgeKind = EdgeKind.POLYMER_CONNECTION
    bond_order: float = 1.0
    probability: float = 1.0
    directed: bool = False
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolymerArchitecture:
    architecture_type: ArchitectureType
    nodes: list[ArchitectureNode] = field(default_factory=list)
    edges: list[ArchitectureEdge] = field(default_factory=list)
    sequence: list[str] | None = None
    head_node: str | None = None
    tail_node: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, units: dict[str, PolymerUnit]) -> None:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("architecture node ids must be unique")
        node_map = {node.id: node for node in self.nodes}
        for node in self.nodes:
            if node.unit_id not in units:
                raise ValueError(f"architecture node references unknown unit: {node.unit_id}")
        for edge in self.edges:
            if edge.source not in node_map or edge.target not in node_map:
                raise ValueError("architecture edge references unknown node")
            if not 0 <= edge.probability <= 1:
                raise ValueError("edge probability must lie in [0, 1]")
            source_unit = units[node_map[edge.source].unit_id]
            target_unit = units[node_map[edge.target].unit_id]
            if edge.source_port is not None:
                source_unit.graph.port(edge.source_port)
            if edge.target_port is not None:
                target_unit.graph.port(edge.target_port)
        if self.sequence is not None:
            unknown = set(self.sequence) - set(units)
            if unknown:
                raise ValueError(f"sequence references unknown units: {sorted(unknown)}")
        if self.head_node is not None and self.head_node not in node_map:
            raise ValueError("head_node is unknown")
        if self.tail_node is not None and self.tail_node not in node_map:
            raise ValueError("tail_node is unknown")


@dataclass
class PropertyValue:
    name: str
    value: float | int | str | list[float]
    unit: str | None = None
    uncertainty: float | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    source_type: str | None = None
    dataset_name: str | None = None
    record_id: str | None = None
    reference: str | None = None
    software: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolymerRecord:
    id: str
    units: dict[str, PolymerUnit]
    architecture: PolymerArchitecture
    ensemble: EnsembleStatistics = field(default_factory=EnsembleStatistics)
    spatial_states: list[SpatialState] = field(default_factory=list)
    properties: dict[str, PropertyValue] = field(default_factory=dict)
    process_history: ProcessHistory = field(default_factory=ProcessHistory)
    provenance: Provenance = field(default_factory=Provenance)
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1.0"

    def validate(self) -> None:
        if not self.id:
            raise ValueError("polymer record id cannot be empty")
        if not self.units:
            raise ValueError("polymer record must contain at least one unit")
        for key, unit in self.units.items():
            if key != unit.id:
                raise ValueError(f"unit dictionary key {key!r} differs from unit.id {unit.id!r}")
            unit.validate()
        self.architecture.validate(self.units)
        self.ensemble.validate(set(self.units))
        state_ids: set[str] = set()
        for state in self.spatial_states:
            state.validate()
            if state.id in state_ids:
                raise ValueError(f"duplicate spatial state id: {state.id}")
            state_ids.add(state.id)
        self.process_history.validate()

    def get_state(self, state_id: str) -> SpatialState:
        for state in self.spatial_states:
            if state.id == state_id:
                return state
        raise KeyError(f"unknown spatial state: {state_id}")
