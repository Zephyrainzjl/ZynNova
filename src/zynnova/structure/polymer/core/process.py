from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProcessStep:
    id: str
    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    input_state_ids: list[str] = field(default_factory=list)
    output_state_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessHistory:
    steps: list[ProcessStep] = field(default_factory=list)
    dependencies: list[tuple[str, str]] = field(default_factory=list)

    def validate(self) -> None:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("process step ids must be unique")
        known = set(ids)
        for source, target in self.dependencies:
            if source not in known or target not in known:
                raise ValueError("process dependency references unknown step")
