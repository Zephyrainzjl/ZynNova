from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...common import require_torch

torch = require_torch()
nn = torch.nn


@dataclass(slots=True)
class MigrationBarrierConfig:
    max_atomic_number: int = 118
    hidden_dim: int = 128
    num_radial: int = 24
    cutoff_A: float = 6.0
    minimum_uncertainty_eV: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.max_atomic_number < 1:
            raise ValueError("max_atomic_number must be positive")
        if self.hidden_dim < 16:
            raise ValueError("hidden_dim must be at least 16")
        if self.num_radial < 4 or self.cutoff_A <= 0:
            raise ValueError("invalid radial environment configuration")
        if self.minimum_uncertainty_eV <= 0:
            raise ValueError("minimum_uncertainty_eV must be positive")


@dataclass(slots=True)
class MigrationBarrierPrediction:
    barrier_eV: float
    uncertainty_eV: float
    migrating_index: int
    destination_A: np.ndarray


class MigrationBarrierModel(nn.Module):
    """Predict a Ni migration barrier from a dual-endpoint local path tube.

    Every neighboring atom is encoded relative to both the initial site and the
    candidate destination. Path-parallel and path-normal coordinates distinguish
    a channel from a compact radial shell. Optional JouleWeave charge, magnetic
    moment, and oxidation-state predictions enter as local electronic features.
    The second output is a heteroscedastic uncertainty trained with Gaussian NLL.
    """

    def __init__(self, config: MigrationBarrierConfig | None = None) -> None:
        super().__init__()
        self.config = config or MigrationBarrierConfig()
        channels = self.config.hidden_dim
        self.element_embedding = nn.Embedding(
            self.config.max_atomic_number + 1,
            channels,
            padding_idx=0,
        )
        self.register_buffer(
            "radial_centers_A",
            torch.linspace(0.0, self.config.cutoff_A, self.config.num_radial),
        )
        spacing = self.config.cutoff_A / max(self.config.num_radial - 1, 1)
        self.radial_gamma = 1.0 / max(spacing * spacing, 1.0e-8)
        self.electronic_projection = nn.Sequential(
            nn.Linear(3, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.message = nn.Sequential(
            nn.Linear(
                2 * channels + 2 * self.config.num_radial + 2,
                2 * channels,
            ),
            nn.SiLU(),
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * channels + 4, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 2),
        )

    @staticmethod
    def _minimum_image(displacement: Any, cell: Any, pbc: Any) -> Any:
        if not bool(torch.any(pbc).item()):
            return displacement
        if bool((torch.abs(torch.linalg.det(cell)) < 1.0e-12).item()):
            raise ValueError("periodic migration encoding requires a non-singular cell")
        fractional = displacement @ torch.linalg.inv(cell)
        fractional = fractional - torch.round(fractional) * pbc.to(fractional)
        return fractional @ cell

    def _radial(self, distance: Any) -> Any:
        centers = self.radial_centers_A.to(distance)
        gaussian = torch.exp(-self.radial_gamma * (distance[:, None] - centers[None, :]).square())
        envelope = torch.where(
            distance < self.config.cutoff_A,
            0.5 * (torch.cos(torch.pi * distance / self.config.cutoff_A) + 1.0),
            torch.zeros_like(distance),
        )
        return gaussian * envelope[:, None]

    @staticmethod
    def _as_event_vector(value: Any, *, device: Any, dtype: Any) -> Any:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError("destination_positions must have shape [events, 3]")
        return value

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        positions = inputs.get("pos", inputs.get("positions"))
        atomic_numbers = inputs.get("z", inputs.get("atomic_numbers"))
        destination = inputs.get(
            "destination_positions",
            inputs.get("final_positions"),
        )
        migrating = inputs.get("migrating_index", inputs.get("migrating_indices"))
        if positions is None or atomic_numbers is None:
            raise KeyError("MigrationBarrierModel requires atomic numbers and positions")
        if migrating is None:
            raise KeyError("migrating_index is required")
        positions = positions.to(dtype=self.element_embedding.weight.dtype)
        atomic_numbers = atomic_numbers.to(device=positions.device, dtype=torch.long)
        if not torch.is_tensor(migrating):
            migrating = torch.as_tensor(
                migrating,
                device=positions.device,
                dtype=torch.long,
            )
        else:
            migrating = migrating.to(device=positions.device, dtype=torch.long)
        migrating = migrating.reshape(-1)
        if destination is None:
            destination_index = inputs.get("destination_index")
            if destination_index is None:
                raise KeyError(
                    "destination_positions/final_positions or destination_index is required"
                )
            destination_index = torch.as_tensor(
                destination_index,
                device=positions.device,
                dtype=torch.long,
            ).reshape(-1)
            destination = positions[destination_index]
        destination = self._as_event_vector(
            destination,
            device=positions.device,
            dtype=positions.dtype,
        )
        if destination.shape[0] == 1 and migrating.numel() > 1:
            destination = destination.expand(migrating.numel(), -1)
        if destination.shape[0] != migrating.numel():
            raise ValueError("one destination is required per migration event")
        if bool(torch.any((migrating < 0) | (migrating >= len(atomic_numbers))).item()):
            raise IndexError("migrating atom index is out of range")

        batch = inputs.get("batch")
        if batch is None:
            batch = torch.zeros(
                len(atomic_numbers),
                device=positions.device,
                dtype=torch.long,
            )
        else:
            batch = batch.to(device=positions.device, dtype=torch.long)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        cell = inputs.get("cell")
        if cell is None:
            cell = positions.new_zeros((graph_count, 3, 3))
        else:
            cell = cell.to(device=positions.device, dtype=positions.dtype)
            if cell.ndim == 2:
                cell = cell[None, :, :]
            if cell.shape[0] == 1 and graph_count > 1:
                cell = cell.expand(graph_count, -1, -1)
        pbc = inputs.get("pbc")
        if pbc is None:
            pbc = torch.zeros((graph_count, 3), device=positions.device, dtype=torch.bool)
        else:
            pbc = pbc.to(device=positions.device, dtype=torch.bool)
            if pbc.ndim == 1:
                pbc = pbc[None, :]
            if pbc.shape[0] == 1 and graph_count > 1:
                pbc = pbc.expand(graph_count, -1)

        charge = inputs.get("charges")
        magmom = inputs.get("magmoms")
        oxidation = inputs.get("oxidation_states")
        electronic = positions.new_zeros((len(positions), 3))
        for column, (value, scale) in enumerate(((charge, 4.0), (magmom, 5.0), (oxidation, 8.0))):
            if value is not None:
                electronic[:, column] = (
                    value.to(device=positions.device, dtype=positions.dtype).reshape(-1) / scale
                )
        element = self.element_embedding(atomic_numbers)
        electronic_embedding = self.electronic_projection(electronic)
        event_features: list[Any] = []
        for event_index, atom_index in enumerate(migrating.tolist()):
            graph_index = int(batch[atom_index].item())
            atom_mask = batch == graph_index
            local_positions = positions[atom_mask]
            local_element = element[atom_mask]
            local_electronic = electronic_embedding[atom_mask]
            start = positions[atom_index]
            end = destination[event_index]
            path = self._minimum_image(
                (end - start)[None, :],
                cell[graph_index],
                pbc[graph_index],
            )[0]
            path_length = torch.linalg.vector_norm(path).clamp_min(1.0e-8)
            path_unit = path / path_length
            from_start = self._minimum_image(
                local_positions - start,
                cell[graph_index],
                pbc[graph_index],
            )
            from_end = self._minimum_image(
                local_positions - end,
                cell[graph_index],
                pbc[graph_index],
            )
            distance_start = torch.linalg.vector_norm(from_start, dim=-1)
            distance_end = torch.linalg.vector_norm(from_end, dim=-1)
            projection = torch.sum(from_start * path_unit[None, :], dim=-1)
            perpendicular = torch.linalg.vector_norm(
                from_start - projection[:, None] * path_unit[None, :],
                dim=-1,
            )
            geometry = torch.stack(
                (
                    projection / self.config.cutoff_A,
                    perpendicular / self.config.cutoff_A,
                ),
                dim=-1,
            )
            messages = self.message(
                torch.cat(
                    (
                        local_element,
                        local_electronic,
                        self._radial(distance_start),
                        self._radial(distance_end),
                        geometry,
                    ),
                    dim=-1,
                )
            )
            nearest_distance = torch.minimum(distance_start, distance_end)
            weight = torch.where(
                nearest_distance < self.config.cutoff_A,
                0.5 * (torch.cos(torch.pi * nearest_distance / self.config.cutoff_A) + 1.0),
                torch.zeros_like(nearest_distance),
            )
            pooled = torch.sum(messages * weight[:, None], dim=0) / torch.sqrt(
                weight.sum().clamp_min(1.0)
            )
            event_features.append(
                torch.cat(
                    (
                        pooled,
                        element[atom_index],
                        path_length[None] / self.config.cutoff_A,
                        electronic[atom_index],
                    )
                )
            )
        raw = self.readout(torch.stack(event_features, dim=0))
        barrier = torch.nn.functional.softplus(raw[:, 0])
        minimum_log_variance = 2.0 * math.log(self.config.minimum_uncertainty_eV)
        log_variance = raw[:, 1].clamp(min=minimum_log_variance, max=6.0)
        return {
            "barrier_eV": barrier,
            "log_variance": log_variance,
            "uncertainty_eV": torch.exp(0.5 * log_variance),
        }

    def predict(
        self,
        structure: Any,
        *,
        migrating_index: int,
        destination_A: Sequence[float],
        charges: Sequence[float] | np.ndarray | None = None,
        magmoms: Sequence[float] | np.ndarray | None = None,
        oxidation_states: Sequence[float] | np.ndarray | None = None,
    ) -> MigrationBarrierPrediction:
        from ....dynamics.adapters import to_ase_atoms

        atoms = to_ase_atoms(structure)
        parameter = next(self.parameters())
        inputs: dict[str, Any] = {
            "z": torch.as_tensor(
                atoms.get_atomic_numbers(),
                device=parameter.device,
                dtype=torch.long,
            ),
            "pos": torch.as_tensor(
                atoms.get_positions(),
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            "cell": torch.as_tensor(
                atoms.cell.array,
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            "pbc": torch.as_tensor(
                atoms.pbc,
                device=parameter.device,
                dtype=torch.bool,
            ),
            "migrating_index": int(migrating_index),
            "destination_positions": destination_A,
        }
        for name, value in (
            ("charges", charges),
            ("magmoms", magmoms),
            ("oxidation_states", oxidation_states),
        ):
            if value is not None:
                inputs[name] = torch.as_tensor(
                    value,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
        self.eval()
        with torch.no_grad():
            output = self(inputs)
        return MigrationBarrierPrediction(
            barrier_eV=float(output["barrier_eV"][0].cpu()),
            uncertainty_eV=float(output["uncertainty_eV"][0].cpu()),
            migrating_index=int(migrating_index),
            destination_A=np.asarray(destination_A, dtype=float).reshape(3),
        )


def migration_barrier_loss(
    prediction: Mapping[str, Any],
    target_barrier_eV: Any,
) -> Any:
    """Heteroscedastic Gaussian negative log-likelihood for NEB barriers."""

    barrier = prediction["barrier_eV"].reshape(-1)
    log_variance = prediction["log_variance"].reshape(-1)
    target = torch.as_tensor(
        target_barrier_eV,
        device=barrier.device,
        dtype=barrier.dtype,
    ).reshape(-1)
    if target.shape != barrier.shape:
        raise ValueError("target_barrier_eV must contain one value per event")
    residual = barrier - target
    return 0.5 * torch.mean(torch.exp(-log_variance) * residual.square() + log_variance)


__all__ = [
    "MigrationBarrierConfig",
    "MigrationBarrierModel",
    "MigrationBarrierPrediction",
    "migration_barrier_loss",
]
