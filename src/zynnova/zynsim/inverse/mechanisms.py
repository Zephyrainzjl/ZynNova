"""Sparse differentiable identification of competing degradation mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mechanism identification requires PyTorch") from exc
    return torch


@dataclass(slots=True)
class MechanismSelectionConfig:
    temperature: float = 0.5
    sparsity_weight: float = 1.0e-3
    minimum_probability: float = 1.0e-4


class MechanismSelector:
    def __new__(cls, names: Sequence[str], config: MechanismSelectionConfig | None = None):
        torch = _torch()
        nn = torch.nn
        resolved = config or MechanismSelectionConfig()
        if not names or len(set(names)) != len(names):
            raise ValueError("mechanism names must be non-empty and unique")

        class _Selector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.names = tuple(names)
                self.config = resolved
                self.logits = nn.Parameter(torch.zeros(len(names)))

            def probabilities(self) -> Any:
                return torch.softmax(self.logits / self.config.temperature, dim=0).clamp_min(
                    self.config.minimum_probability
                )

            def combine(self, contributions: Any) -> Any:
                if contributions.shape[-1] != len(self.names):
                    raise ValueError("one contribution is required per mechanism")
                return torch.sum(contributions * self.probabilities(), dim=-1)

            def sparsity_penalty(self) -> Any:
                probabilities = self.probabilities()
                entropy = -torch.sum(probabilities * torch.log(probabilities))
                return self.config.sparsity_weight * entropy

            def ranked(self) -> list[tuple[str, float]]:
                probabilities = self.probabilities().detach().cpu().tolist()
                return sorted(zip(self.names, probabilities, strict=True), key=lambda item: item[1], reverse=True)

        return _Selector()


__all__ = ["MechanismSelectionConfig", "MechanismSelector"]
