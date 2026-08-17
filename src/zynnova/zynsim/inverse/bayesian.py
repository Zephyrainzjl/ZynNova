"""Diagonal-Laplace Bayesian uncertainty propagation for differentiable twins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Bayesian inversion requires PyTorch") from exc
    return torch


@dataclass(frozen=True, slots=True)
class DiagonalLaplacePosterior:
    mean: Mapping[str, Any]
    precision: Mapping[str, Any]
    prior_precision: float

    def sample(self, count: int, *, seed: int = 42) -> list[dict[str, Any]]:
        if count < 1:
            raise ValueError("posterior sample count must be positive")
        torch = _torch()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        samples: list[dict[str, Any]] = []
        for _ in range(count):
            sample: dict[str, Any] = {}
            for name, mean in self.mean.items():
                precision = self.precision[name]
                standard = torch.rsqrt(precision.clamp_min(1.0e-12))
                noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
                sample[name] = mean + noise.to(mean.device) * standard
            samples.append(sample)
        return samples


class DiagonalLaplace:
    def __init__(self, model: Any, *, prior_precision: float = 1.0) -> None:
        if prior_precision <= 0.0:
            raise ValueError("prior precision must be positive")
        self.model = model
        self.prior_precision = float(prior_precision)

    def fit(self, loss_closure: Callable[[], Any]) -> DiagonalLaplacePosterior:
        torch = _torch()
        self.model.zero_grad(set_to_none=True)
        loss = loss_closure()
        gradients = torch.autograd.grad(
            loss,
            tuple(self.model.parameters()),
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        mean: dict[str, Any] = {}
        precision: dict[str, Any] = {}
        for (name, parameter), gradient in zip(self.model.named_parameters(), gradients, strict=True):
            mean[name] = parameter.detach().clone()
            if gradient is None:
                precision[name] = torch.full_like(parameter, self.prior_precision)
            else:
                precision[name] = gradient.detach().square() + self.prior_precision
        return DiagonalLaplacePosterior(mean, precision, self.prior_precision)


def propagate_posterior(
    model: Any,
    posterior: DiagonalLaplacePosterior,
    evaluator: Callable[[], Mapping[str, Any]],
    *,
    samples: int = 100,
) -> dict[str, Any]:
    torch = _torch()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    outputs: dict[str, list[Any]] = {}
    try:
        for draw in posterior.sample(samples):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(draw[name].to(parameter))
            result = evaluator()
            for name, value in result.items():
                outputs.setdefault(name, []).append(torch.as_tensor(value).detach())
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(original[name].to(parameter))
    return {
        name: {
            "mean": torch.mean(torch.stack(values), dim=0),
            "standard_deviation": torch.std(torch.stack(values), dim=0, unbiased=True),
        }
        for name, values in outputs.items()
    }


__all__ = [
    "DiagonalLaplace",
    "DiagonalLaplacePosterior",
    "propagate_posterior",
]
