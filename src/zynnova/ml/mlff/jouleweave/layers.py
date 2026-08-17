from __future__ import annotations

from typing import Any

from ...common import require_torch
from .graph import scatter_sum, smooth_cutoff

torch = require_torch()
nn = torch.nn


def vector_norm(vector: Any, *, eps: float = 1.0e-12) -> Any:
    return torch.sqrt(vector.square().sum(dim=-1) + eps)


def tensor_norm(tensor: Any, *, eps: float = 1.0e-12) -> Any:
    return torch.sqrt(tensor.square().sum(dim=(-1, -2)) + eps)


def symmetric_traceless(matrix: Any) -> Any:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    trace = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    return symmetric - trace[..., None, None] * identity / 3.0


def direction_quadrupole(unit_vector: Any) -> Any:
    outer = unit_vector[..., :, None] * unit_vector[..., None, :]
    identity = torch.eye(3, device=unit_vector.device, dtype=unit_vector.dtype)
    return outer - identity / 3.0


def vector_direction_tensor(vector: Any, unit_vector: Any) -> Any:
    first = vector[..., :, None] * unit_vector[:, None, None, :]
    second = unit_vector[:, None, :, None] * vector[..., None, :]
    return symmetric_traceless(first + second)


class ChannelLinear(nn.Module):
    """Bias-free channel mixing for vectors or Cartesian rank-2 tensors."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.orthogonal_(self.weight)

    def forward(self, values: Any) -> Any:
        if values.ndim == 3:
            return torch.einsum("nci,oc->noi", values, self.weight)
        if values.ndim == 4:
            return torch.einsum("ncij,oc->noij", values, self.weight)
        raise ValueError("ChannelLinear expects [N,C,3] or [N,C,3,3]")


class CompactSplineBasis(nn.Module):
    """Uniform cubic B-splines multiplied by a C2 compact envelope."""

    def __init__(self, size: int, cutoff: float, *, trainable: bool = True) -> None:
        super().__init__()
        if size < 4:
            raise ValueError("spline basis needs at least four functions")
        self.size = int(size)
        self.cutoff = float(cutoff)
        self.register_buffer("knots", torch.arange(size, dtype=torch.get_default_dtype()))
        if trainable:
            self.log_scale = nn.Parameter(torch.zeros(size))
        else:
            self.register_buffer("log_scale", torch.zeros(size))

    def forward(self, distance: Any) -> Any:
        coordinate = distance[:, None] * (self.size - 1) / self.cutoff
        delta = torch.abs(coordinate - self.knots[None, :].to(distance))
        inner = (4.0 - 6.0 * delta.square() + 3.0 * delta**3) / 6.0
        outer = (2.0 - delta).clamp_min(0.0) ** 3 / 6.0
        basis = torch.where(delta < 1.0, inner, torch.where(delta < 2.0, outer, 0.0))
        scale = torch.exp(self.log_scale).to(distance)
        return basis * scale[None, :] * smooth_cutoff(distance, self.cutoff)[:, None]


class EquivariantRMSNorm(nn.Module):
    """One smooth norm shared by scalar, vector and rank-2 field channels."""

    def __init__(self, channels: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.scalar_weight = nn.Parameter(torch.ones(channels))
        self.scalar_bias = nn.Parameter(torch.zeros(channels))
        self.vector_weight = nn.Parameter(torch.ones(channels))
        self.tensor_weight = nn.Parameter(torch.ones(channels))

    def forward(self, scalar: Any, vector: Any, tensor: Any) -> tuple[Any, Any, Any]:
        power = (
            scalar.square().mean(dim=-1)
            + vector.square().mean(dim=(-1, -2))
            + tensor.square().mean(dim=(-1, -2, -3))
        ) / 3.0
        inverse_rms = torch.rsqrt(power + self.eps)
        scalar = scalar * inverse_rms[:, None]
        vector = vector * inverse_rms[:, None, None]
        tensor = tensor * inverse_rms[:, None, None, None]
        return (
            scalar * self.scalar_weight + self.scalar_bias,
            vector * self.vector_weight[None, :, None],
            tensor * self.tensor_weight[None, :, None, None],
        )


class SmoothFieldAttentionBlock(nn.Module):
    """Tensor-product-free O(3)-equivariant graph-field interaction.

    Attention is deliberately *not* normalized over a changing neighbor set.
    Every edge message is instead multiplied by a C2 envelope and a smooth
    learned gate, while a smooth density statistic controls coordination scaling.
    This keeps energy, forces, and higher derivatives continuous at the cutoff.
    """

    _GATE_GROUPS = 9

    def __init__(
        self,
        channels: int,
        radial_size: int,
        *,
        attention_heads: int,
        cutoff: float,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if channels % attention_heads:
            raise ValueError("channels must be divisible by attention_heads")
        self.channels = int(channels)
        self.attention_heads = int(attention_heads)
        self.channels_per_head = channels // attention_heads
        self.cutoff = float(cutoff)
        self.residual_scale = float(residual_scale)

        self.norm = EquivariantRMSNorm(channels)
        self.radial_projection = nn.Sequential(
            nn.Linear(radial_size, channels, bias=False),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=False),
        )
        self.receiver_projection = nn.Linear(channels, channels, bias=False)
        self.sender_projection = nn.Linear(channels, channels, bias=False)
        self.invariant_projection = nn.Linear(4 * channels, channels, bias=False)
        self.edge_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                channels,
                self._GATE_GROUPS * channels + attention_heads,
                bias=False,
            ),
        )

        self.scalar_sender = nn.Linear(channels, channels, bias=False)
        self.vector_to_scalar = nn.Linear(channels, channels, bias=False)
        self.tensor_to_scalar = nn.Linear(channels, channels, bias=False)
        self.scalar_to_vector = nn.Linear(channels, channels, bias=False)
        self.scalar_to_tensor = nn.Linear(channels, channels, bias=False)
        self.vector_sender = ChannelLinear(channels)
        self.vector_cross = ChannelLinear(channels)
        self.tensor_sender = ChannelLinear(channels)
        self.tensor_to_vector = ChannelLinear(channels)

        self.update = nn.Sequential(
            nn.Linear(4 * channels, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, 3 * channels),
        )
        self.vector_update = ChannelLinear(channels)
        self.tensor_update = ChannelLinear(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        scalar: Any,
        vector: Any,
        tensor: Any,
        edge_index: Any,
        edge_vector: Any,
        radial: Any,
    ) -> tuple[Any, Any, Any]:
        if not edge_vector.numel():
            return scalar, vector, tensor

        normalized_scalar, normalized_vector, normalized_tensor = self.norm(
            scalar,
            vector,
            tensor,
        )
        receiver, sender = edge_index
        distance = torch.linalg.vector_norm(edge_vector, dim=-1)
        unit = edge_vector / distance.clamp_min(1.0e-12)[:, None]
        envelope = smooth_cutoff(distance, self.cutoff)

        sender_vector = normalized_vector[sender]
        sender_tensor = normalized_tensor[sender]
        vector_projection = torch.einsum("eci,ei->ec", sender_vector, unit)
        tensor_projection = torch.einsum(
            "ecij,ei,ej->ec",
            sender_tensor,
            unit,
            unit,
        )
        invariant = torch.cat(
            (
                vector_projection,
                tensor_projection,
                vector_norm(sender_vector),
                tensor_norm(sender_tensor),
            ),
            dim=-1,
        )
        edge_state = (
            self.radial_projection(radial)
            + self.receiver_projection(normalized_scalar[receiver])
            + self.sender_projection(normalized_scalar[sender])
            + self.invariant_projection(invariant)
        )
        raw_gate = self.edge_gate(edge_state)
        channel_gates = torch.tanh(raw_gate[:, : self._GATE_GROUPS * self.channels]).reshape(
            -1, self._GATE_GROUPS, self.channels
        )
        attention = 2.0 * torch.sigmoid(raw_gate[:, self._GATE_GROUPS * self.channels :])
        attention = attention.repeat_interleave(self.channels_per_head, dim=-1)
        edge_weight = attention * envelope[:, None]

        scalar_message = (
            channel_gates[:, 0] * self.scalar_sender(normalized_scalar[sender])
            + channel_gates[:, 1] * self.vector_to_scalar(vector_projection)
            + channel_gates[:, 2] * self.tensor_to_scalar(tensor_projection)
        )

        tensor_vector = torch.einsum("ecij,ej->eci", sender_tensor, unit)
        vector_message = (
            channel_gates[:, 3, :, None] * self.vector_sender(sender_vector)
            + channel_gates[:, 4, :, None]
            * self.scalar_to_vector(normalized_scalar[sender])[:, :, None]
            * unit[:, None, :]
            + channel_gates[:, 5, :, None] * self.tensor_to_vector(tensor_vector)
        )

        quadrupole = direction_quadrupole(unit)
        mixed_tensor = vector_direction_tensor(self.vector_cross(sender_vector), unit)
        tensor_message = (
            channel_gates[:, 6, :, None, None] * self.tensor_sender(sender_tensor)
            + channel_gates[:, 7, :, None, None]
            * self.scalar_to_tensor(normalized_scalar[sender])[:, :, None, None]
            * quadrupole[:, None, :, :]
            + channel_gates[:, 8, :, None, None] * mixed_tensor
        )

        scalar_message = scalar_message * edge_weight
        vector_message = vector_message * edge_weight[:, :, None]
        tensor_message = tensor_message * edge_weight[:, :, None, None]

        node_count = scalar.shape[0]
        density = scatter_sum(envelope.square(), receiver, node_count)
        density_scale = torch.rsqrt(1.0 + density)
        aggregate_scalar = scatter_sum(scalar_message, receiver, node_count)
        aggregate_vector = scatter_sum(vector_message, receiver, node_count)
        aggregate_tensor = scatter_sum(tensor_message, receiver, node_count)
        aggregate_scalar = aggregate_scalar * density_scale[:, None]
        aggregate_vector = aggregate_vector * density_scale[:, None, None]
        aggregate_tensor = aggregate_tensor * density_scale[:, None, None, None]

        update_input = torch.cat(
            (
                normalized_scalar,
                aggregate_scalar,
                vector_norm(aggregate_vector),
                tensor_norm(aggregate_tensor),
            ),
            dim=-1,
        )
        scalar_delta, vector_gate, tensor_gate = self.update(update_input).chunk(3, dim=-1)
        scalar_delta = self.dropout(scalar_delta)
        vector_delta = torch.sigmoid(vector_gate)[:, :, None] * self.vector_update(aggregate_vector)
        tensor_delta = torch.sigmoid(tensor_gate)[:, :, None, None] * self.tensor_update(
            aggregate_tensor
        )
        return (
            scalar + self.residual_scale * scalar_delta,
            vector + self.residual_scale * vector_delta,
            tensor + self.residual_scale * tensor_delta,
        )


class ExpertAtomicReadout(nn.Module):
    """Environment-routed atomic-energy experts with invariant inputs."""

    def __init__(
        self,
        channels: int,
        num_experts: int,
        max_atomic_number: int,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.feature_mixer = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
        )
        self.experts = nn.Linear(channels, num_experts)
        self.router = nn.Linear(channels, num_experts, bias=False)
        self.element_router = nn.Embedding(max_atomic_number + 1, num_experts)
        nn.init.zeros_(self.element_router.weight)

    def forward(
        self,
        scalar: Any,
        vector: Any,
        tensor: Any,
        atomic_numbers: Any,
    ) -> tuple[Any, Any]:
        invariant = torch.cat(
            (scalar, vector_norm(vector), tensor_norm(tensor)),
            dim=-1,
        )
        hidden = self.feature_mixer(invariant)
        expert_energy = self.experts(hidden)
        if self.num_experts == 1:
            weights = torch.ones_like(expert_energy)
        else:
            logits = self.router(hidden) + self.element_router(atomic_numbers.long())
            weights = torch.softmax(logits, dim=-1)
        return torch.sum(weights * expert_energy, dim=-1), weights


class MagneticMomentConstraint(nn.Module):
    """Predict site magnetic moments and condition the final scalar field.

    The input is the O(3)-invariant projection of the penultimate scalar,
    vector, and rank-2 fields. Consequently, ``magmoms`` are invariant under
    rotations and reflections. The supervised magnetic latent is injected only
    into the scalar field, preserving equivariance of the final interaction.
    The final projection is zero-initialized so an old non-magnetic checkpoint
    can be fine-tuned with ``strict=False`` without an immediate energy shift.
    """

    def __init__(
        self,
        channels: int,
        *,
        nonnegative: bool = True,
        condition_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.nonnegative = bool(nonnegative)
        self.condition_scale = float(condition_scale)
        self.feature_mixer = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
        )
        self.magmom_head = nn.Linear(channels, 1)
        self.condition_gate = nn.Sequential(
            nn.Linear(2 * channels + 1, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        nn.init.zeros_(self.condition_gate[-1].weight)
        nn.init.zeros_(self.condition_gate[-1].bias)

    def forward(
        self,
        scalar: Any,
        vector: Any,
        tensor: Any,
    ) -> tuple[Any, Any, Any]:
        invariant = torch.cat(
            (scalar, vector_norm(vector), tensor_norm(tensor)),
            dim=-1,
        )
        magnetic_latent = self.feature_mixer(invariant)
        raw_magmom = self.magmom_head(magnetic_latent).squeeze(-1)
        magmoms = (
            torch.nn.functional.softplus(raw_magmom)
            if self.nonnegative
            else raw_magmom
        )
        gate, candidate = self.condition_gate(
            torch.cat((scalar, magnetic_latent, magmoms[:, None]), dim=-1)
        ).chunk(2, dim=-1)
        conditioned_scalar = scalar + self.condition_scale * torch.sigmoid(gate) * candidate
        return conditioned_scalar, magmoms, magnetic_latent


class RedoxChargeConstraint(nn.Module):
    """Couple supervised partition charge and oxidation state to the last block.

    Both heads consume O(3)-invariant features from the penultimate interaction
    state. The raw site charge and the expected oxidation state gate only the
    scalar field, so the final interaction remains O(3)-equivariant.

    Reported partition charges are shifted by one graph-wise Lagrange multiplier
    to satisfy the supplied total charge exactly. The unshifted, strictly local
    charge is used for conditioning; consequently the energy remains compatible
    with domain-decomposed LAMMPS inference.
    """

    def __init__(
        self,
        channels: int,
        *,
        use_charges: bool,
        use_oxidation_states: bool,
        oxidation_state_min: int,
        oxidation_state_max: int,
        condition_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.use_charges = bool(use_charges)
        self.use_oxidation_states = bool(use_oxidation_states)
        self.condition_scale = float(condition_scale)
        oxidation_values = torch.arange(
            int(oxidation_state_min),
            int(oxidation_state_max) + 1,
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("oxidation_values", oxidation_values)
        self.feature_mixer = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.SiLU(),
        )
        self.charge_head = nn.Linear(channels, 1) if self.use_charges else None
        self.oxidation_head = (
            nn.Linear(channels, len(oxidation_values))
            if self.use_oxidation_states
            else None
        )
        self.condition_gate = nn.Sequential(
            nn.Linear(2 * channels + 2, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        if self.charge_head is not None:
            nn.init.zeros_(self.charge_head.weight)
            nn.init.zeros_(self.charge_head.bias)
        if self.oxidation_head is not None:
            nn.init.zeros_(self.oxidation_head.weight)
            nn.init.zeros_(self.oxidation_head.bias)
        nn.init.zeros_(self.condition_gate[-1].weight)
        nn.init.zeros_(self.condition_gate[-1].bias)

    @staticmethod
    def conserve_total_charge(
        raw_charges: Any,
        batch: Any,
        total_charge: Any,
        graph_count: int,
    ) -> Any:
        counts = scatter_sum(
            torch.ones_like(raw_charges),
            batch,
            graph_count,
        ).clamp_min(1.0)
        predicted_total = scatter_sum(raw_charges, batch, graph_count)
        correction = (total_charge.to(raw_charges) - predicted_total) / counts
        return raw_charges + correction[batch]

    def forward(
        self,
        scalar: Any,
        vector: Any,
        tensor: Any,
        batch: Any,
        total_charge: Any,
        graph_count: int,
    ) -> tuple[Any, dict[str, Any]]:
        invariant = torch.cat(
            (scalar, vector_norm(vector), tensor_norm(tensor)),
            dim=-1,
        )
        redox_latent = self.feature_mixer(invariant)

        raw_charges = scalar.new_zeros((scalar.shape[0],))
        partition_charges = None
        if self.charge_head is not None:
            raw_charges = self.charge_head(redox_latent).squeeze(-1)
            partition_charges = self.conserve_total_charge(
                raw_charges,
                batch,
                total_charge,
                graph_count,
            )

        oxidation_logits = None
        oxidation_probabilities = None
        oxidation_states = scalar.new_zeros((scalar.shape[0],))
        if self.oxidation_head is not None:
            oxidation_logits = self.oxidation_head(redox_latent)
            oxidation_probabilities = torch.softmax(oxidation_logits, dim=-1)
            oxidation_states = torch.sum(
                oxidation_probabilities
                * self.oxidation_values.to(oxidation_probabilities)[None, :],
                dim=-1,
            )

        gate, candidate = self.condition_gate(
            torch.cat(
                (
                    scalar,
                    redox_latent,
                    raw_charges[:, None],
                    oxidation_states[:, None],
                ),
                dim=-1,
            )
        ).chunk(2, dim=-1)
        conditioned_scalar = (
            scalar
            + self.condition_scale * torch.sigmoid(gate) * candidate
        )
        return conditioned_scalar, {
            "partition_charges": partition_charges,
            "raw_partition_charges": raw_charges if self.charge_head is not None else None,
            "oxidation_state_logits": oxidation_logits,
            "oxidation_state_probabilities": oxidation_probabilities,
            "oxidation_states": (
                oxidation_states if self.oxidation_head is not None else None
            ),
            "redox_latent": redox_latent,
        }


__all__ = [
    "ChannelLinear",
    "CompactSplineBasis",
    "EquivariantRMSNorm",
    "ExpertAtomicReadout",
    "MagneticMomentConstraint",
    "RedoxChargeConstraint",
    "SmoothFieldAttentionBlock",
    "direction_quadrupole",
    "symmetric_traceless",
    "tensor_norm",
    "vector_direction_tensor",
    "vector_norm",
]
