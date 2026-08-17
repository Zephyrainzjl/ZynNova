"""Checkpoint and trajectory I/O for phase-field simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import BoundaryCondition, GridSpec
from .fields import PhaseFieldState, PhaseFieldTrajectory


def save_state(state: PhaseFieldState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "shape": state.grid.shape,
        "spacing": state.grid.spacing,
        "origin": state.grid.origin,
        "boundary": [item.value for item in state.grid.boundary],
        "time": state.time,
        "step": state.step,
        "metadata": state.metadata,
        "field_names": list(state.fields),
    }
    payload = {name: np.asarray(values) for name, values in state.fields.items()}
    payload["__metadata__"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(target, **payload)
    return target


def load_state(path: str | Path) -> PhaseFieldState:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["__metadata__"]))
        fields = {name: np.asarray(data[name]) for name in metadata["field_names"]}
    grid = GridSpec(
        tuple(metadata["shape"]),
        spacing=tuple(metadata["spacing"]),
        origin=tuple(metadata["origin"]),
        boundary=tuple(BoundaryCondition(item) for item in metadata["boundary"]),
    )
    return PhaseFieldState(
        grid,
        fields,
        time=float(metadata["time"]),
        step=int(metadata["step"]),
        metadata=dict(metadata.get("metadata", {})),
    )


def save_trajectory(trajectory: PhaseFieldTrajectory, path: str | Path) -> Path:
    if not trajectory.frames:
        raise ValueError("cannot save an empty trajectory")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(trajectory.frames[0].fields)
    payload: dict[str, Any] = {
        "times": trajectory.times,
        "steps": trajectory.steps,
    }
    for name in field_names:
        payload[name] = trajectory.stack(name)
    first = trajectory.frames[0]
    payload["__metadata__"] = np.asarray(
        json.dumps(
            {
                "shape": first.grid.shape,
                "spacing": first.grid.spacing,
                "origin": first.grid.origin,
                "boundary": [item.value for item in first.grid.boundary],
                "field_names": field_names,
            }
        )
    )
    np.savez_compressed(target, **payload)
    return target


def load_trajectory(path: str | Path) -> PhaseFieldTrajectory:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["__metadata__"]))
        times = np.asarray(data["times"], dtype=float)
        steps = np.asarray(data["steps"], dtype=int)
        stacks = {name: np.asarray(data[name]) for name in metadata["field_names"]}
    grid = GridSpec(
        tuple(metadata["shape"]),
        spacing=tuple(metadata["spacing"]),
        origin=tuple(metadata["origin"]),
        boundary=tuple(BoundaryCondition(item) for item in metadata["boundary"]),
    )
    trajectory = PhaseFieldTrajectory()
    for index, (time, step) in enumerate(zip(times, steps, strict=True)):
        trajectory.frames.append(
            PhaseFieldState(
                grid,
                {name: values[index] for name, values in stacks.items()},
                time=float(time),
                step=int(step),
            )
        )
    return trajectory


__all__ = ["load_state", "load_trajectory", "save_state", "save_trajectory"]
