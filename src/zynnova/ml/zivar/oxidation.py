"""Charge-balanced formal oxidation-state prediction.

Formal oxidation states are discrete chemical labels, not rounded partial
charges. The neural head therefore predicts categorical logits. A small exact
dynamic program enforces the integer total-charge constraint at inference,
while training uses the differentiable categorical expectation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._deps import require_torch
from .config import OxidationConfig

torch = require_torch()
nn = torch.nn


# Curated common states for battery-relevant and frequently occurring inorganic
# elements. Elements not listed retain the configured full interval so the
# model never silently declares a chemically possible state impossible.
COMMON_OXIDATION_STATES: dict[int, tuple[int, ...]] = {
    1: (-1, 0, 1),
    3: (0, 1),
    4: (0, 2),
    5: (-3, 0, 3),
    6: (-4, -2, 0, 2, 4),
    7: (-3, 0, 1, 2, 3, 4, 5),
    8: (-2, -1, 0, 1, 2),
    9: (-1, 0),
    11: (0, 1),
    12: (0, 2),
    13: (0, 3),
    14: (-4, 0, 2, 4),
    15: (-3, 0, 3, 5),
    16: (-2, 0, 2, 4, 6),
    17: (-1, 0, 1, 3, 5, 7),
    19: (0, 1),
    20: (0, 2),
    21: (0, 3),
    22: (0, 2, 3, 4),
    23: (0, 2, 3, 4, 5),
    24: (0, 2, 3, 6),
    25: (0, 2, 3, 4, 6, 7),
    26: (0, 2, 3, 4, 6),
    27: (0, 2, 3, 4),
    28: (0, 2, 3, 4),
    29: (0, 1, 2, 3),
    30: (0, 2),
    31: (0, 1, 3),
    32: (-4, 0, 2, 4),
    33: (-3, 0, 3, 5),
    34: (-2, 0, 2, 4, 6),
    35: (-1, 0, 1, 3, 5, 7),
    37: (0, 1),
    38: (0, 2),
    39: (0, 3),
    40: (0, 2, 3, 4),
    41: (0, 2, 3, 4, 5),
    42: (0, 2, 3, 4, 5, 6),
    43: (0, 4, 7),
    44: (0, 2, 3, 4, 6, 8),
    45: (0, 1, 2, 3, 4),
    46: (0, 2, 4),
    47: (0, 1, 2, 3),
    48: (0, 2),
    49: (0, 1, 3),
    50: (-4, 0, 2, 4),
    51: (-3, 0, 3, 5),
    52: (-2, 0, 2, 4, 6),
    53: (-1, 0, 1, 3, 5, 7),
    55: (0, 1),
    56: (0, 2),
    57: (0, 3),
    58: (0, 3, 4),
    59: (0, 3, 4),
    60: (0, 2, 3, 4),
    62: (0, 2, 3),
    64: (0, 2, 3),
    72: (0, 4),
    73: (0, 5),
    74: (0, 4, 5, 6),
    78: (0, 2, 4),
    79: (0, 1, 3),
    80: (0, 1, 2),
    82: (0, 2, 4),
}


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> Any:
    layers: list[Any] = []
    current = input_dim
    for width in hidden:
        layers.extend((nn.Linear(current, width), nn.SiLU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


@dataclass(slots=True)
class OxidationPrediction:
    logits: Any
    expectation: Any
    states: Any
    allowed_mask: Any
    probabilities: Any
    confidence: Any
    entropy: Any


class OxidationStateHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        atomic_numbers: tuple[int, ...],
        config: OxidationConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.network = _mlp(feature_dim, config.hidden, self.state_count)
        states = torch.arange(config.minimum_state, config.maximum_state + 1)
        mask = torch.ones((119, self.state_count), dtype=torch.bool)
        for number, allowed in COMMON_OXIDATION_STATES.items():
            if number > 118:
                continue
            row = torch.zeros(self.state_count, dtype=torch.bool)
            for value in allowed:
                if config.minimum_state <= value <= config.maximum_state:
                    row[value - config.minimum_state] = True
            if bool(row.any()):
                mask[number] = row
        self.register_buffer("state_values", states, persistent=True)
        self.register_buffer("allowed_by_Z", mask, persistent=True)
        self.register_buffer(
            "element_table", torch.as_tensor(atomic_numbers, dtype=torch.long),
            persistent=True,
        )

    @property
    def state_count(self) -> int:
        return self.config.maximum_state - self.config.minimum_state + 1

    def forward(
        self,
        features: Any,
        atomic_numbers: Any,
        batch: Any,
        total_charge: Any,
        *,
        exact: bool,
    ) -> OxidationPrediction:
        raw = self.network(features)
        allowed = self.allowed_by_Z[atomic_numbers]
        floor = torch.finfo(raw.dtype).min / 4.0
        logits = torch.where(allowed, raw, raw.new_full((), floor))
        probabilities = torch.softmax(logits / self.config.temperature, dim=-1)
        values = self.state_values.to(device=raw.device, dtype=raw.dtype)
        expectation = probabilities @ values
        states = values[probabilities.argmax(-1)]
        if exact:
            states = exact_charge_balanced_states(
                logits.detach(), allowed, batch, total_charge, self.state_values
            ).to(device=raw.device, dtype=raw.dtype)
        confidence = probabilities.max(-1).values
        entropy = -(
            probabilities * probabilities.clamp_min(torch.finfo(raw.dtype).tiny).log()
        ).sum(-1)
        return OxidationPrediction(
            logits, expectation, states, allowed, probabilities, confidence, entropy
        )


def exact_charge_balanced_states(
    logits: Any,
    allowed_mask: Any,
    batch: Any,
    total_charge: Any,
    state_values: Any,
) -> Any:
    """Maximum-logit assignment with an exact per-graph integer sum."""

    if logits.ndim != 2 or allowed_mask.shape != logits.shape:
        raise ValueError("oxidation logits and mask must have shape [N,S]")
    if batch.shape != (logits.shape[0],):
        raise ValueError("oxidation batch must have shape [N]")
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    if total_charge.shape != (graph_count,):
        raise ValueError("formal total charge must have shape [B]")
    values = [int(item) for item in state_values.detach().cpu().tolist()]
    output = torch.empty(logits.shape[0], dtype=torch.long)
    score_cpu = logits.detach().cpu()
    mask_cpu = allowed_mask.detach().cpu()
    batch_cpu = batch.detach().cpu()
    target_cpu = total_charge.detach().cpu()
    for graph in range(graph_count):
        indices = torch.nonzero(batch_cpu == graph, as_tuple=False).flatten().tolist()
        target_float = float(target_cpu[graph])
        target = int(round(target_float))
        if abs(target_float - target) > 1.0e-6:
            raise ValueError("formal oxidation-state balance requires integer charge")
        # On exactly degenerate logit assignments, prefer the smaller total
        # absolute oxidation magnitude.  This makes the chemical tie-break
        # explicit and independent of dictionary/state iteration order.
        layers: list[dict[int, tuple[float, int, int, int]]] = []
        current: dict[int, tuple[float, int, int, int]] = {0: (0.0, 0, -1, -1)}
        for atom in indices:
            next_layer: dict[int, tuple[float, int, int, int]] = {}
            for previous_sum, (previous_score, previous_magnitude, _, _) in current.items():
                for state_index, state in enumerate(values):
                    if not bool(mask_cpu[atom, state_index]):
                        continue
                    candidate_sum = previous_sum + state
                    candidate_score = previous_score + float(score_cpu[atom, state_index])
                    candidate_magnitude = previous_magnitude + abs(state)
                    incumbent = next_layer.get(candidate_sum)
                    if incumbent is None or candidate_score > incumbent[0] or (
                        candidate_score == incumbent[0]
                        and candidate_magnitude < incumbent[1]
                    ):
                        next_layer[candidate_sum] = (
                            candidate_score,
                            candidate_magnitude,
                            previous_sum,
                            state_index,
                        )
            if not next_layer:
                raise ValueError("no allowed oxidation state for an atom")
            layers.append(next_layer)
            current = next_layer
        if target not in current:
            raise ValueError(
                f"no charge-balanced oxidation-state assignment for graph {graph}"
            )
        running = target
        chosen = [0] * len(indices)
        for layer_index in range(len(indices) - 1, -1, -1):
            _, _, previous, state_index = layers[layer_index][running]
            chosen[layer_index] = values[state_index]
            running = previous
        for atom, state in zip(indices, chosen, strict=True):
            output[atom] = state
    return output


__all__ = [
    "COMMON_OXIDATION_STATES",
    "OxidationPrediction",
    "OxidationStateHead",
    "exact_charge_balanced_states",
]
