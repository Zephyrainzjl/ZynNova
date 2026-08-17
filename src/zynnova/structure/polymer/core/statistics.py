from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .enums import DistributionKind


@dataclass
class Distribution:
    kind: DistributionKind
    parameters: dict[str, float] = field(default_factory=dict)
    samples: np.ndarray | None = None
    bin_edges: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    unit: str | None = None
    uncertainty: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.samples is not None:
            self.samples = np.asarray(self.samples, dtype=float)
            if self.samples.ndim != 1:
                raise ValueError("distribution samples must be one-dimensional")
        if self.bin_edges is not None or self.probabilities is not None:
            if self.bin_edges is None or self.probabilities is None:
                raise ValueError("histogram requires both bin_edges and probabilities")
            self.bin_edges = np.asarray(self.bin_edges, dtype=float)
            self.probabilities = np.asarray(self.probabilities, dtype=float)
            if len(self.bin_edges) != len(self.probabilities) + 1:
                raise ValueError("len(bin_edges) must equal len(probabilities) + 1")
            if np.any(self.probabilities < 0):
                raise ValueError("histogram probabilities cannot be negative")
            total = float(self.probabilities.sum())
            if total <= 0:
                raise ValueError("histogram probability sum must be positive")
            self.probabilities = self.probabilities / total

    def representative_value(self) -> float | None:
        for key in ("mean", "value", "number_average", "Mn", "mode"):
            if key in self.parameters:
                return float(self.parameters[key])
        if self.samples is not None and len(self.samples):
            return float(np.mean(self.samples))
        if self.bin_edges is not None and self.probabilities is not None:
            centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
            return float(np.sum(centers * self.probabilities))
        return None


@dataclass
class EnsembleStatistics:
    composition: dict[str, float] = field(default_factory=dict)
    transition_matrix: np.ndarray | None = None
    transition_unit_order: list[str] = field(default_factory=list)
    degree_of_polymerization: Distribution | None = None
    molecular_weight: Distribution | None = None
    branch_length: Distribution | None = None
    block_length: Distribution | None = None
    tacticity: dict[str, float] = field(default_factory=dict)
    end_group_fraction: dict[str, float] = field(default_factory=dict)
    crosslink_density: float | None = None
    number_of_chains: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, known_unit_ids: set[str]) -> None:
        if self.composition:
            unknown = set(self.composition) - known_unit_ids
            if unknown:
                raise ValueError(f"composition references unknown units: {sorted(unknown)}")
            values = np.asarray(list(self.composition.values()), dtype=float)
            if np.any(values < 0):
                raise ValueError("composition fractions cannot be negative")
            total = float(values.sum())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"composition must sum to 1, got {total}")
        if self.transition_matrix is not None:
            matrix = np.asarray(self.transition_matrix, dtype=float)
            n = len(self.transition_unit_order)
            if matrix.shape != (n, n):
                raise ValueError("transition_matrix shape must match transition_unit_order")
            if set(self.transition_unit_order) - known_unit_ids:
                raise ValueError("transition matrix references unknown units")
            if np.any(matrix < 0):
                raise ValueError("transition probabilities cannot be negative")
            if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("each transition matrix row must sum to 1")
            self.transition_matrix = matrix
        for distribution in (
            self.degree_of_polymerization,
            self.molecular_weight,
            self.branch_length,
            self.block_length,
        ):
            if distribution is not None:
                distribution.validate()
        if self.crosslink_density is not None and self.crosslink_density < 0:
            raise ValueError("crosslink_density cannot be negative")
        if self.number_of_chains is not None and self.number_of_chains < 1:
            raise ValueError("number_of_chains must be positive")
