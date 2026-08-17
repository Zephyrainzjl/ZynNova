from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .core.polymer import PolymerRecord


@dataclass
class RepresentationSchema:
    """Dataset-level tensor contract shared by all polymer training views."""

    unit_vocabulary: dict[str, int]
    edge_vocabulary: dict[str, int]
    max_nodes: int
    max_tokens: int = 512
    atom_vocabulary: dict[str, int] = field(
        default_factory=lambda: {"[PAD]": 0, "[UNK]": 1}
    )
    bond_vocabulary: dict[str, int] = field(
        default_factory=lambda: {
            "none": 0,
            "single": 1,
            "aromatic": 2,
            "double": 3,
            "triple": 4,
            "other": 5,
        }
    )
    max_atoms: int | None = None
    property_names: list[str] = field(default_factory=list)
    continuous_feature_names: list[str] = field(
        default_factory=lambda: [
            "log_dp",
            "log_mn",
            "log_dispersity_minus_one",
            "crosslink_density_log1p",
            "tacticity_logit",
        ]
    )
    schema_id: str = "default"

    PAD_UNIT: str = "[PAD]"
    UNK_UNIT: str = "[UNK]"
    PAD_ATOM: str = "[PAD]"
    UNK_ATOM: str = "[UNK]"
    NO_EDGE: str = "none"

    def __post_init__(self) -> None:
        if self.PAD_UNIT not in self.unit_vocabulary:
            raise ValueError("unit_vocabulary must contain [PAD]")
        if self.UNK_UNIT not in self.unit_vocabulary:
            raise ValueError("unit_vocabulary must contain [UNK]")
        if self.NO_EDGE not in self.edge_vocabulary:
            raise ValueError("edge_vocabulary must contain 'none'")
        if self.PAD_ATOM not in self.atom_vocabulary:
            raise ValueError("atom_vocabulary must contain [PAD]")
        if self.UNK_ATOM not in self.atom_vocabulary:
            raise ValueError("atom_vocabulary must contain [UNK]")
        if self.NO_EDGE not in self.bond_vocabulary:
            raise ValueError("bond_vocabulary must contain 'none'")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if self.max_atoms is None:
            self.max_atoms = self.max_nodes
        if self.max_atoms < 1:
            raise ValueError("max_atoms must be positive")

    def unit_index(self, unit_id: str) -> int:
        return self.unit_vocabulary.get(unit_id, self.unit_vocabulary[self.UNK_UNIT])

    def atom_index(self, atomic_number: int) -> int:
        return self.atom_vocabulary.get(
            f"Z{int(atomic_number)}", self.atom_vocabulary[self.UNK_ATOM]
        )

    @property
    def inverse_unit_vocabulary(self) -> dict[int, str]:
        return {value: key for key, value in self.unit_vocabulary.items()}

    @property
    def inverse_atom_vocabulary(self) -> dict[int, str]:
        return {value: key for key, value in self.atom_vocabulary.items()}

    @property
    def inverse_edge_vocabulary(self) -> dict[int, str]:
        return {value: key for key, value in self.edge_vocabulary.items()}

    @property
    def inverse_bond_vocabulary(self) -> dict[int, str]:
        return {value: key for key, value in self.bond_vocabulary.items()}

    @staticmethod
    def bond_label(order: float, aromatic: bool = False) -> str:
        if aromatic or abs(order - 1.5) < 0.2:
            return "aromatic"
        if abs(order - 1.0) < 0.2:
            return "single"
        if abs(order - 2.0) < 0.2:
            return "double"
        if abs(order - 3.0) < 0.2:
            return "triple"
        return "other"

    @classmethod
    def fit(
        cls,
        records: Iterable[PolymerRecord],
        *,
        max_nodes: int | None = None,
        max_atoms: int | None = None,
        max_tokens: int = 512,
        schema_id: str = "fitted",
    ) -> "RepresentationSchema":
        records = list(records)
        if not records:
            raise ValueError("cannot fit schema from an empty record collection")
        unit_ids: set[str] = set()
        edge_types: set[str] = {cls.NO_EDGE}
        atomic_numbers: set[int] = set()
        property_names: set[str] = set()
        observed_max_nodes = 1
        observed_max_atoms = 1
        for record in records:
            record.validate()
            unit_ids.update(record.units)
            edge_types.update(edge.kind.value for edge in record.architecture.edges)
            property_names.update(record.properties)
            observed_max_nodes = max(observed_max_nodes, len(record.architecture.nodes))
            template_atoms = 0
            for unit in record.units.values():
                atomic_numbers.update(atom.atomic_number for atom in unit.graph.atoms)
                template_atoms += unit.graph.num_atoms
            observed_max_atoms = max(observed_max_atoms, template_atoms)
            for state in record.spatial_states:
                for frame in state.frames:
                    if frame.resolution.value == "atomistic":
                        observed_max_atoms = max(observed_max_atoms, len(frame.node_ids))
        unit_vocabulary = {cls.PAD_UNIT: 0, cls.UNK_UNIT: 1}
        for unit_id in sorted(unit_ids):
            unit_vocabulary[unit_id] = len(unit_vocabulary)
        ordered_edges = [cls.NO_EDGE] + [
            item for item in sorted(edge_types) if item != cls.NO_EDGE
        ]
        edge_vocabulary = {item: index for index, item in enumerate(ordered_edges)}
        atom_vocabulary = {cls.PAD_ATOM: 0, cls.UNK_ATOM: 1}
        for atomic_number in sorted(atomic_numbers):
            atom_vocabulary[f"Z{atomic_number}"] = len(atom_vocabulary)
        return cls(
            unit_vocabulary=unit_vocabulary,
            edge_vocabulary=edge_vocabulary,
            atom_vocabulary=atom_vocabulary,
            max_nodes=max_nodes or observed_max_nodes,
            max_atoms=max_atoms or observed_max_atoms,
            max_tokens=max_tokens,
            property_names=sorted(property_names),
            schema_id=schema_id,
        )
