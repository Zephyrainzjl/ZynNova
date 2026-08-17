"""Field schemas, simulation states, trajectories, and diagnostic records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .config import GridSpec


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Definition of a phase-field variable."""

    name: str
    conserved: bool
    components: int = 1
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field names cannot be empty")
        if self.components <= 0:
            raise ValueError("components must be positive")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound cannot exceed upper_bound")


@dataclass(slots=True)
class PhaseFieldState:
    """Named fields evaluated on a common Cartesian grid."""

    grid: GridSpec
    fields: dict[str, Any]
    time: float = 0.0
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a phase-field state must contain at least one field")
        for name, values in self.fields.items():
            shape = tuple(int(value) for value in getattr(values, "shape", ()))
            if shape[-self.grid.dimensions :] != self.grid.shape:
                raise ValueError(
                    f"field {name!r} has spatial shape {shape}; expected trailing shape "
                    f"{self.grid.shape}"
                )

    def copy(self, *, deep: bool = True) -> "PhaseFieldState":
        if not deep:
            fields = dict(self.fields)
        else:
            fields = {}
            for name, values in self.fields.items():
                if hasattr(values, "clone"):
                    fields[name] = values.clone()
                elif hasattr(values, "copy"):
                    fields[name] = values.copy()
                else:
                    fields[name] = np.asarray(values).copy()
        return PhaseFieldState(
            self.grid,
            fields,
            float(self.time),
            int(self.step),
            dict(self.metadata),
        )

    def numpy(self) -> "PhaseFieldState":
        converted: dict[str, np.ndarray] = {}
        for name, values in self.fields.items():
            if hasattr(values, "detach"):
                values = values.detach()
            if hasattr(values, "cpu"):
                values = values.cpu()
            converted[name] = np.asarray(values)
        return PhaseFieldState(
            self.grid,
            converted,
            float(self.time),
            int(self.step),
            dict(self.metadata),
        )

    def field(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f"unknown phase-field variable {name!r}") from exc


@dataclass(frozen=True, slots=True)
class PhaseFieldDiagnostics:
    """Scalar diagnostics produced during time integration."""

    step: int
    time: float
    dt: float
    free_energy: float
    energy_change: float
    accepted: bool
    mass: Mapping[str, float]
    extrema: Mapping[str, tuple[float, float]]
    residual_norm: float = 0.0
    rejected_steps: int = 0


@dataclass(slots=True)
class PhaseFieldTrajectory:
    """Saved phase-field frames and corresponding diagnostics."""

    frames: list[PhaseFieldState] = field(default_factory=list)
    diagnostics: list[PhaseFieldDiagnostics] = field(default_factory=list)

    def append(self, state: PhaseFieldState, *, deep: bool = True) -> None:
        self.frames.append(state.copy(deep=deep).numpy())

    @property
    def times(self) -> np.ndarray:
        return np.asarray([frame.time for frame in self.frames], dtype=float)

    @property
    def steps(self) -> np.ndarray:
        return np.asarray([frame.step for frame in self.frames], dtype=int)

    def stack(self, field_name: str) -> np.ndarray:
        if not self.frames:
            raise ValueError("trajectory is empty")
        return np.stack([np.asarray(frame.fields[field_name]) for frame in self.frames])

    @property
    def final_state(self) -> PhaseFieldState:
        if not self.frames:
            raise ValueError("trajectory is empty")
        return self.frames[-1]


@dataclass(frozen=True, slots=True)
class PhaseFieldResult:
    """Completed phase-field simulation and runtime metadata."""

    trajectory: PhaseFieldTrajectory
    backend: str
    scheme: str
    wall_time_s: float
    accepted_steps: int
    rejected_steps: int
    converged: bool
    message: str = ""


__all__ = [
    "FieldSpec",
    "PhaseFieldDiagnostics",
    "PhaseFieldResult",
    "PhaseFieldState",
    "PhaseFieldTrajectory",
]
