from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from .exceptions import SchemaError
from .record import MaterialSample, MaterialType


class FieldRole(StrEnum):
    FEATURE = "feature"
    LABEL = "label"
    CONDITION = "condition"
    METADATA = "metadata"


class FieldLevel(StrEnum):
    GRAPH = "graph"
    NODE = "node"
    EDGE = "edge"
    ATOM = "atom"
    STRUCTURE = "structure"
    SEQUENCE = "sequence"
    TOKEN = "token"


class TaskKind(StrEnum):
    PREDICTION = "prediction"
    GENERATION = "generation"
    POTENTIAL = "potential"
    REPRESENTATION = "representation"


class MissingPolicy(StrEnum):
    ERROR = "error"
    DROP = "drop"
    MASK = "mask"
    FILL = "fill"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Description of one freely selectable model field."""

    name: str
    source: str
    role: FieldRole | str = FieldRole.FEATURE
    level: FieldLevel | str = FieldLevel.GRAPH
    dtype: str | None = None
    unit: str | None = None
    shape: tuple[int | None, ...] | None = None
    required: bool = True
    missing: MissingPolicy | str = MissingPolicy.ERROR
    fill_value: Any = 0.0
    normalize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", FieldRole(self.role))
        object.__setattr__(self, "level", FieldLevel(self.level))
        object.__setattr__(self, "missing", MissingPolicy(self.missing))
        if not self.name:
            raise ValueError("field name cannot be empty")
        if not self.source:
            raise ValueError("field source cannot be empty")

    def read(self, sample: MaterialSample) -> tuple[Any, bool]:
        sentinel = object()
        value = sample.get(self.source, sentinel)
        if value is not sentinel and value is not None:
            return value, True
        if self.missing is MissingPolicy.FILL:
            return self.fill_value, False
        if self.missing in {MissingPolicy.MASK, MissingPolicy.DROP} or not self.required:
            return None, False
        raise SchemaError(
            f"required field {self.name!r} ({self.source}) is missing in sample {sample.id!r}"
        )


@dataclass(frozen=True, slots=True)
class StructureEncodingSpec:
    """How the structure portion of a sample is represented for a model."""

    representation: str = "pyg"
    view: str = "graph"
    options: dict[str, Any] = field(default_factory=dict)
    include_reconstruction: bool = False


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Compile the same physical records for prediction, generation or potentials."""

    name: str
    kind: TaskKind | str
    inputs: tuple[FieldSpec, ...] = ()
    targets: tuple[FieldSpec, ...] = ()
    conditions: tuple[FieldSpec, ...] = ()
    structure: StructureEncodingSpec | None = field(default_factory=StructureEncodingSpec)
    material_types: tuple[MaterialType | str, ...] = ()
    include_id: bool = True
    include_metadata: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TaskKind(self.kind))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(
            self,
            "material_types",
            tuple(MaterialType(value) for value in self.material_types),
        )
        if not self.name:
            raise ValueError("task name cannot be empty")

    def accepts(self, sample: MaterialSample) -> bool:
        return not self.material_types or sample.material_type in self.material_types

    @classmethod
    def property_prediction(
        cls,
        name: str,
        *,
        targets: Iterable[str | FieldSpec],
        inputs: Iterable[str | FieldSpec] = (),
        conditions: Iterable[str | FieldSpec] = (),
        representation: str = "pyg",
        view: str = "graph",
        structure_options: dict[str, Any] | None = None,
        material_types: Iterable[MaterialType | str] = (),
    ) -> "TaskSpec":
        return cls(
            name=name,
            kind=TaskKind.PREDICTION,
            inputs=tuple(_field(value, FieldRole.FEATURE) for value in inputs),
            targets=tuple(_field(value, FieldRole.LABEL) for value in targets),
            conditions=tuple(_field(value, FieldRole.CONDITION) for value in conditions),
            structure=StructureEncodingSpec(
                representation=representation,
                view=view,
                options=structure_options or {},
            ),
            material_types=tuple(material_types),
        )

    @classmethod
    def generation(
        cls,
        name: str,
        *,
        conditions: Iterable[str | FieldSpec] = (),
        representation: str = "dense_graph",
        view: str = "generative",
        structure_options: dict[str, Any] | None = None,
        material_types: Iterable[MaterialType | str] = (),
    ) -> "TaskSpec":
        return cls(
            name=name,
            kind=TaskKind.GENERATION,
            conditions=tuple(_field(value, FieldRole.CONDITION) for value in conditions),
            structure=StructureEncodingSpec(
                representation=representation,
                view=view,
                options=structure_options or {},
            ),
            material_types=tuple(material_types),
        )

    @classmethod
    def neural_potential(
        cls,
        name: str = "neural_potential",
        *,
        energy: str | FieldSpec | None = "labels.energy",
        forces: str | FieldSpec | None = "labels.forces",
        stress: str | FieldSpec | None = None,
        inputs: Iterable[str | FieldSpec] = (),
        conditions: Iterable[str | FieldSpec] = (),
        extra_targets: Iterable[FieldSpec] = (),
        structure_options: dict[str, Any] | None = None,
        material_types: Iterable[MaterialType | str] = (),
    ) -> "TaskSpec":
        targets: list[FieldSpec] = []
        if energy is not None:
            targets.append(
                energy
                if isinstance(energy, FieldSpec)
                else _target_field("energy", energy, FieldLevel.GRAPH)
            )
        if forces is not None:
            targets.append(
                forces
                if isinstance(forces, FieldSpec)
                else _target_field("forces", forces, FieldLevel.ATOM)
            )
        if stress is not None:
            targets.append(
                stress
                if isinstance(stress, FieldSpec)
                else _target_field("stress", stress, FieldLevel.GRAPH)
            )
        targets.extend(extra_targets)
        if not targets:
            raise ValueError("neural potential task requires at least one target")
        return cls(
            name=name,
            kind=TaskKind.POTENTIAL,
            inputs=tuple(_field(value, FieldRole.FEATURE) for value in inputs),
            targets=tuple(targets),
            conditions=tuple(_field(value, FieldRole.CONDITION) for value in conditions),
            structure=StructureEncodingSpec(
                representation="potential",
                view="atomistic",
                options=structure_options or {},
            ),
            material_types=tuple(material_types),
        )



def _target_field(name: str, source: str, level: FieldLevel) -> FieldSpec:
    if "." not in source:
        source = f"labels.{source}"
    return FieldSpec(name, source, FieldRole.LABEL, level)


def _field(value: str | FieldSpec, role: FieldRole) -> FieldSpec:
    if isinstance(value, FieldSpec):
        return value
    name = value.rsplit(".", 1)[-1]
    if "." not in value:
        prefix = {
            FieldRole.FEATURE: "features",
            FieldRole.LABEL: "labels",
            FieldRole.CONDITION: "conditions",
            FieldRole.METADATA: "metadata",
        }[role]
        source = f"{prefix}.{value}"
    else:
        source = value
    return FieldSpec(name=name, source=source, role=role)
