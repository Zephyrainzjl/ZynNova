"""Grand-canonical, constant-potential extension for JouleWeave.

The extension deliberately wraps an existing :class:`JouleWeave` backbone so
legacy checkpoints remain loadable.  It adds graph-level electron-number,
Fermi-level, differential-capacitance, uncertainty, and atom-level reaction
heads.  The thermodynamic scalar used for dynamics is the grand potential
rather than the canonical energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...common import require_torch


torch = require_torch()
nn = torch.nn


def _graph_vector(
    value: Any | None,
    *,
    graph_count: int,
    reference: Any,
    name: str,
    default: float = 0.0,
) -> Any:
    if value is None:
        return torch.full(
            (graph_count,),
            float(default),
            device=reference.device,
            dtype=reference.dtype,
        )
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    result = result.reshape(-1)
    if result.numel() == 1 and graph_count > 1:
        result = result.expand(graph_count)
    if result.shape != (graph_count,):
        raise ValueError(f"{name} must be scalar or have shape [graphs]")
    return result


def _scatter_mean(values: Any, batch: Any, graph_count: int) -> Any:
    pooled = values.new_zeros((graph_count, values.shape[-1]))
    pooled.index_add_(0, batch, values)
    counts = values.new_zeros((graph_count,))
    counts.index_add_(0, batch, values.new_ones((values.shape[0],)))
    return pooled / counts.clamp_min(1.0).unsqueeze(-1)


@dataclass(slots=True)
class GrandCanonicalConfig:
    """Configuration for the electrochemical JouleWeave wrapper.

    ``reference_fermi_eV`` fixes the conversion from electrode potential to an
    electronic chemical potential using ``mu_e = reference_fermi_eV - V``.
    This sign convention is explicit and can be changed for a chosen reference
    electrode by changing the reference value, without retraining the backbone.
    """

    hidden_dim: int | None = None
    reference_fermi_eV: float = 0.0
    reference_electron_count: float = 0.0
    minimum_capacitance_e_per_V: float = 1.0e-4
    maximum_electron_deviation: float = 64.0
    reaction_channels: int = 8
    uncertainty_floor_eV: float = 1.0e-4
    self_consistency_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim is not None and self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least eight")
        if self.minimum_capacitance_e_per_V <= 0.0:
            raise ValueError("minimum capacitance must be positive")
        if self.maximum_electron_deviation <= 0.0:
            raise ValueError("maximum electron deviation must be positive")
        if self.reaction_channels < 1:
            raise ValueError("reaction_channels must be positive")
        if self.uncertainty_floor_eV <= 0.0:
            raise ValueError("uncertainty_floor_eV must be positive")
        if self.self_consistency_weight < 0.0:
            raise ValueError("self_consistency_weight cannot be negative")


class ConstantPotentialJouleWeave(nn.Module):
    """Constant-potential reactive potential built on a JouleWeave backbone.

    Inputs use the normal JouleWeave structure mapping plus:

    - ``electrode_potential_V``: scalar or one value per graph;
    - ``electron_count``: optional supervised/externally imposed electron count;
    - ``reference_electron_count``: optional graph-specific neutral count.

    When ``electron_count`` is omitted, a bounded response head predicts it.
    For inference that requires an explicitly self-consistent electronic state,
    use :meth:`solve_electron_count` before calling :meth:`forward`.
    """

    def __init__(self, backbone: Any, config: GrandCanonicalConfig | None = None) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        if not hasattr(backbone, "config") or not hasattr(backbone.config, "hidden_dim"):
            raise TypeError("backbone must expose config.hidden_dim")
        self.backbone = backbone
        self.config = config or GrandCanonicalConfig()
        channels = int(self.config.hidden_dim or backbone.config.hidden_dim)
        backbone_channels = int(backbone.config.hidden_dim)
        self.pool_projection = (
            nn.Identity()
            if channels == backbone_channels
            else nn.Linear(backbone_channels, channels)
        )
        self.potential_embedding = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.electron_embedding = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.electron_response = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.fermi_head = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.capacitance_head = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        self.reaction_head = nn.Sequential(
            nn.Linear(backbone_channels + channels, channels),
            nn.SiLU(),
            nn.Linear(channels, self.config.reaction_channels),
        )

    def electronic_chemical_potential(self, electrode_potential_V: Any) -> Any:
        return float(self.config.reference_fermi_eV) - electrode_potential_V

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        base = self.backbone(inputs)
        node_scalar = base.get("node_scalar")
        if node_scalar is None:
            raise KeyError("JouleWeave backbone output must contain node_scalar")
        batch = inputs.get("batch")
        if batch is None:
            batch = torch.zeros(
                node_scalar.shape[0], device=node_scalar.device, dtype=torch.long
            )
        else:
            batch = torch.as_tensor(batch, device=node_scalar.device, dtype=torch.long)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        pooled = self.pool_projection(_scatter_mean(node_scalar, batch, graph_count))
        potential = _graph_vector(
            inputs.get("electrode_potential_V"),
            graph_count=graph_count,
            reference=pooled,
            name="electrode_potential_V",
        )
        target_mu = self.electronic_chemical_potential(potential)
        potential_latent = self.potential_embedding(potential.unsqueeze(-1))
        response_input = torch.cat((pooled, potential_latent), dim=-1)

        reference_electrons = _graph_vector(
            inputs.get("reference_electron_count"),
            graph_count=graph_count,
            reference=pooled,
            name="reference_electron_count",
            default=self.config.reference_electron_count,
        )
        supplied_electrons = inputs.get("electron_count")
        if supplied_electrons is None:
            raw_delta = self.electron_response(response_input).squeeze(-1)
            delta_electrons = self.config.maximum_electron_deviation * torch.tanh(
                raw_delta / self.config.maximum_electron_deviation
            )
            electron_count = reference_electrons + delta_electrons
            electron_count_source = "response_head"
        else:
            electron_count = _graph_vector(
                supplied_electrons,
                graph_count=graph_count,
                reference=pooled,
                name="electron_count",
            )
            delta_electrons = electron_count - reference_electrons
            electron_count_source = "input"

        electron_latent = self.electron_embedding(delta_electrons.unsqueeze(-1))
        electronic_latent = torch.cat((pooled + potential_latent, electron_latent), dim=-1)
        fermi_level = self.fermi_head(electronic_latent).squeeze(-1)
        capacitance = torch.nn.functional.softplus(
            self.capacitance_head(electronic_latent).squeeze(-1)
        ) + self.config.minimum_capacitance_e_per_V
        log_variance = self.log_variance_head(electronic_latent).squeeze(-1)
        standard_uncertainty = torch.sqrt(
            torch.exp(log_variance).clamp_min(self.config.uncertainty_floor_eV**2)
        )

        canonical_energy = base["energy"].reshape(-1)
        if canonical_energy.shape != (graph_count,):
            raise ValueError("backbone energy must contain one scalar per graph")
        self_consistency_residual = fermi_level - target_mu
        grand_potential = (
            canonical_energy
            - target_mu * delta_electrons
            + 0.5
            * self.config.self_consistency_weight
            * capacitance
            * self_consistency_residual.square()
        )

        atom_condition = potential_latent[batch]
        reaction_logits = self.reaction_head(torch.cat((node_scalar, atom_condition), dim=-1))
        reaction_propensity = torch.sigmoid(reaction_logits)
        return {
            **base,
            "canonical_energy": canonical_energy,
            "grand_potential": grand_potential,
            "electrode_potential_V": potential,
            "target_electronic_chemical_potential_eV": target_mu,
            "electron_count": electron_count,
            "reference_electron_count": reference_electrons,
            "delta_electrons": delta_electrons,
            "electron_count_source": electron_count_source,
            "fermi_level_eV": fermi_level,
            "differential_capacitance_e_per_V": capacitance,
            "self_consistency_residual_eV": self_consistency_residual,
            "energy_standard_uncertainty_eV": standard_uncertainty,
            "reaction_logits": reaction_logits,
            "reaction_propensity": reaction_propensity,
        }

    def grand_potential_and_forces(
        self,
        inputs: Mapping[str, Any],
        *,
        create_graph: bool = False,
    ) -> dict[str, Any]:
        positions = inputs.get("pos", inputs.get("positions"))
        if positions is None:
            raise KeyError("positions are required")
        if not positions.requires_grad:
            positions = positions.clone().requires_grad_(True)
            inputs = dict(inputs)
            inputs["pos"] = positions
            inputs["positions"] = positions
        output = self(inputs)
        forces = -torch.autograd.grad(
            output["grand_potential"].sum(),
            positions,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return {**output, "forces": forces}

    @torch.no_grad()
    def solve_electron_count(
        self,
        inputs: Mapping[str, Any],
        *,
        tolerance_eV: float = 1.0e-4,
        maximum_iterations: int = 24,
        damping: float = 0.7,
    ) -> tuple[Any, dict[str, Any]]:
        """Solve the learned Fermi-level matching condition by damped Newton steps."""

        if tolerance_eV <= 0.0 or maximum_iterations < 1 or not 0.0 < damping <= 1.0:
            raise ValueError("invalid self-consistent electron solver controls")
        trial = dict(inputs)
        trial.pop("electron_count", None)
        initial = self(trial)
        electrons = initial["electron_count"].detach().clone()
        residual = initial["self_consistency_residual_eV"].detach()
        for _ in range(maximum_iterations):
            if float(torch.max(torch.abs(residual)).item()) <= tolerance_eV:
                break
            step_size = torch.maximum(
                electrons.new_full(electrons.shape, 1.0e-3),
                torch.abs(electrons) * 1.0e-4,
            )
            plus_inputs = dict(inputs)
            minus_inputs = dict(inputs)
            plus_inputs["electron_count"] = electrons + step_size
            minus_inputs["electron_count"] = electrons - step_size
            plus = self(plus_inputs)["fermi_level_eV"]
            minus = self(minus_inputs)["fermi_level_eV"]
            derivative = (plus - minus) / (2.0 * step_size)
            safe_derivative = torch.where(
                torch.abs(derivative) < 1.0e-6,
                torch.sign(derivative).where(derivative != 0.0, torch.ones_like(derivative))
                * 1.0e-6,
                derivative,
            )
            electrons = electrons - damping * residual / safe_derivative
            reference = initial["reference_electron_count"]
            electrons = torch.clamp(
                electrons,
                reference - self.config.maximum_electron_deviation,
                reference + self.config.maximum_electron_deviation,
            )
            resolved_inputs = dict(inputs)
            resolved_inputs["electron_count"] = electrons
            resolved = self(resolved_inputs)
            residual = resolved["self_consistency_residual_eV"]
        final_inputs = dict(inputs)
        final_inputs["electron_count"] = electrons
        final = self(final_inputs)
        final["electron_solver_converged"] = bool(
            float(torch.max(torch.abs(final["self_consistency_residual_eV"])).item())
            <= tolerance_eV
        )
        return electrons, final


__all__ = ["ConstantPotentialJouleWeave", "GrandCanonicalConfig"]
