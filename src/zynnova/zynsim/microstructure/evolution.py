"""Topology-aware SEI/CEI growth and crack evolution on voxel microstructures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import BatteryPhase, validate_phase_labels


@dataclass(slots=True)
class MicrostructureEvolutionConfig:
    crack_energy_threshold_J_m3: float = 2.0e5
    crack_growth_rate: float = 0.05
    sei_growth_rate_voxels_sqrt_s: float = 1.0e-4
    cei_growth_rate_voxels_sqrt_s: float = 5.0e-5
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.crack_energy_threshold_J_m3 <= 0.0:
            raise ValueError("crack threshold must be positive")
        if min(self.crack_growth_rate, self.sei_growth_rate_voxels_sqrt_s, self.cei_growth_rate_voxels_sqrt_s) < 0.0:
            raise ValueError("microstructure evolution rates cannot be negative")


class MicrostructureEvolutionModel:
    def __init__(self, config: MicrostructureEvolutionConfig | None = None) -> None:
        self.config = config or MicrostructureEvolutionConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def step(
        self,
        labels: np.ndarray,
        *,
        mechanical_energy_density_J_m3: np.ndarray,
        elapsed_time_s: float,
        dt_s: float,
    ) -> np.ndarray:
        result = validate_phase_labels(labels).copy()
        energy = np.asarray(mechanical_energy_density_J_m3, dtype=float)
        if energy.shape != result.shape or dt_s <= 0.0 or elapsed_time_s < 0.0:
            raise ValueError("microstructure evolution fields or time are invalid")
        active = (result == int(BatteryPhase.POSITIVE_ACTIVE)) | (
            result == int(BatteryPhase.NEGATIVE_ACTIVE)
        )
        excess = np.maximum(
            energy / self.config.crack_energy_threshold_J_m3 - 1.0, 0.0
        )
        probability = 1.0 - np.exp(-self.config.crack_growth_rate * excess * dt_s)
        crack = active & (self._rng.random(result.shape) < probability)
        result[crack] = int(BatteryPhase.CRACK)
        result = self._time_grow_interphase(
            result,
            elapsed_time_s,
            dt_s,
            active=int(BatteryPhase.NEGATIVE_ACTIVE),
            electrolyte=int(BatteryPhase.NEGATIVE_ELECTROLYTE),
            interphase=int(BatteryPhase.NEGATIVE_SEI),
            rate=self.config.sei_growth_rate_voxels_sqrt_s,
        )
        result = self._time_grow_interphase(
            result,
            elapsed_time_s,
            dt_s,
            active=int(BatteryPhase.POSITIVE_ACTIVE),
            electrolyte=int(BatteryPhase.POSITIVE_ELECTROLYTE),
            interphase=int(BatteryPhase.POSITIVE_CEI),
            rate=self.config.cei_growth_rate_voxels_sqrt_s,
        )
        return result

    def _time_grow_interphase(
        self,
        labels: np.ndarray,
        elapsed_time_s: float,
        dt_s: float,
        *,
        active: int,
        electrolyte: int,
        interphase: int,
        rate: float,
    ) -> np.ndarray:
        result = labels.copy()
        increment = rate * (np.sqrt(elapsed_time_s + dt_s) - np.sqrt(elapsed_time_s))
        if increment <= 0.0:
            return result
        active_or_interphase = (result == active) | (result == interphase)
        adjacent = np.zeros_like(active_or_interphase)
        for axis in range(3):
            adjacent |= np.roll(active_or_interphase, 1, axis=axis)
            adjacent |= np.roll(active_or_interphase, -1, axis=axis)
        candidates = adjacent & (result == electrolyte)
        probability = min(increment, 1.0)
        convert = candidates & (self._rng.random(result.shape) < probability)
        result[convert] = interphase
        return result


__all__ = ["MicrostructureEvolutionConfig", "MicrostructureEvolutionModel"]
