from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import SchemaError
from ..record import MaterialSample
from ..schema import FieldSpec, MissingPolicy, TaskKind, TaskSpec
from .structure import encode_structure


@dataclass(slots=True)
class CompiledSample:
    id: str
    kind: TaskKind | str
    structure: Any | None
    inputs: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    masks: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = TaskKind(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "structure": self.structure,
            "inputs": self.inputs,
            "targets": self.targets,
            "conditions": self.conditions,
            "masks": self.masks,
            "metadata": self.metadata,
        }


class TaskCompiler:
    """Compile normalized records into directly trainable model samples."""

    def __init__(
        self,
        task: TaskSpec,
        *,
        representation_schema: Any | None = None,
        tensorize: bool = True,
        device: str | None = None,
    ) -> None:
        self.task = task
        self.representation_schema = representation_schema
        self.tensorize = tensorize
        self.device = device

    def __call__(self, sample: MaterialSample) -> CompiledSample | None:
        if not self.task.accepts(sample):
            return None
        compiled = CompiledSample(
            id=sample.id,
            kind=self.task.kind,
            structure=(
                encode_structure(
                    sample,
                    self.task.structure,
                    representation_schema=self.representation_schema,
                )
                if self.task.structure is not None
                else None
            ),
            metadata=dict(sample.metadata) if self.task.include_metadata else {},
        )
        if not self._read_fields(sample, self.task.inputs, compiled.inputs, compiled.masks):
            return None
        if not self._read_fields(sample, self.task.targets, compiled.targets, compiled.masks):
            return None
        if not self._read_fields(
            sample,
            self.task.conditions,
            compiled.conditions,
            compiled.masks,
        ):
            return None
        if self.tensorize:
            compiled.inputs = _tensorize_mapping(compiled.inputs, self.device)
            compiled.targets = _tensorize_mapping(compiled.targets, self.device)
            compiled.conditions = _tensorize_mapping(compiled.conditions, self.device)
            if self.task.kind is TaskKind.POTENTIAL and isinstance(compiled.structure, dict):
                compiled.structure = _tensorize_mapping(compiled.structure, self.device)
        return compiled

    @staticmethod
    def _read_fields(
        sample: MaterialSample,
        fields: tuple[FieldSpec, ...],
        destination: dict[str, Any],
        masks: dict[str, bool],
    ) -> bool:
        for spec in fields:
            value, present = spec.read(sample)
            if not present and spec.missing is MissingPolicy.DROP:
                return False
            if value is not None:
                destination[spec.name] = _cast_value(value, spec.dtype)
            masks[spec.name] = present
        return True


def _cast_value(value: Any, dtype: str | None) -> Any:
    if dtype is None:
        return value
    if dtype in {"str", "string"}:
        return str(value)
    if dtype in {"bool", "boolean"}:
        return np.asarray(value, dtype=bool)
    return np.asarray(value, dtype=np.dtype(dtype))


def _tensorize_mapping(values: dict[str, Any], device: str | None) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required; install zynnova[data]") from exc
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device) if device else value
        elif isinstance(value, np.ndarray):
            output[key] = torch.as_tensor(value, device=device)
        elif isinstance(value, (bool, int, float, np.number)):
            output[key] = torch.as_tensor(value, device=device)
        else:
            output[key] = value
    return output


def compile_sample(
    sample: MaterialSample,
    task: TaskSpec,
    *,
    representation_schema: Any | None = None,
    tensorize: bool = True,
) -> CompiledSample:
    compiled = TaskCompiler(
        task,
        representation_schema=representation_schema,
        tensorize=tensorize,
    )(sample)
    if compiled is None:
        raise SchemaError(f"sample {sample.id!r} was rejected by task {task.name!r}")
    return compiled
