"""Local phase-field-inspired damage and fracture update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DamageConfig:
    critical_energy_density_J_m3: float = 2.0e5
    relaxation_time_s: float = 10.0
    irreversibility: bool = True
    residual_stiffness: float = 1.0e-6
    gradient_regularization: float = 0.0

    def __post_init__(self) -> None:
        if self.critical_energy_density_J_m3 <= 0.0 or self.relaxation_time_s <= 0.0:
            raise ValueError("damage energy and time scales must be positive")
        if not 0.0 <= self.residual_stiffness < 1.0:
            raise ValueError("residual stiffness must lie in [0,1)")
        if self.gradient_regularization < 0.0:
            raise ValueError("gradient regularization cannot be negative")


class DamageModel:
    def __init__(self, config: DamageConfig | None = None) -> None:
        self.config = config or DamageConfig()

    def step(
        self,
        damage: np.ndarray,
        stress_Pa: np.ndarray,
        dt_s: float,
        *,
        laplacian: np.ndarray | None = None,
    ) -> np.ndarray:
        old = np.asarray(damage, dtype=float)
        stress = np.asarray(stress_Pa, dtype=float)
        if stress.shape[:-1] != old.shape or stress.shape[-1] != 6 or dt_s <= 0.0:
            raise ValueError("damage inputs are inconsistent")
        normal_energy = np.sum(stress[..., :3] ** 2, axis=-1)
        shear_energy = 2.0 * np.sum(stress[..., 3:] ** 2, axis=-1)
        elastic_energy = (normal_energy + shear_energy) / (2.0 * 5.0e9)
        driving = np.maximum(
            elastic_energy / self.config.critical_energy_density_J_m3 - 1.0,
            0.0,
        )
        rate = (1.0 - old) * driving / self.config.relaxation_time_s
        if laplacian is not None and self.config.gradient_regularization > 0.0:
            matrix = np.asarray(laplacian, dtype=float)
            flat = old.reshape(-1)
            if matrix.shape != (flat.size, flat.size):
                raise ValueError("damage Laplacian shape is inconsistent")
            rate += self.config.gradient_regularization * (matrix @ flat).reshape(old.shape)
        updated = np.clip(old + dt_s * rate, 0.0, 1.0)
        if self.config.irreversibility:
            updated = np.maximum(updated, old)
        return updated


__all__ = ["DamageConfig", "DamageModel"]
