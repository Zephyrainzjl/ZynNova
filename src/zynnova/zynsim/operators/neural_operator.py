"""Parameter-embedded Fourier neural operators for battery acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class NeuralOperatorConfig:
    input_channels: int
    output_channels: int
    width: int = 64
    modes: int = 16
    layers: int = 4
    parameter_dim: int = 0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.input_channels, self.output_channels, self.width, self.modes, self.layers) < 1:
            raise ValueError("neural-operator dimensions must be positive")
        if self.parameter_dim < 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid neural-operator parameter dimension/dropout")


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "neural operators require PyTorch; install zynnova[zynsim-gpu]"
        ) from exc
    return torch, nn


class SpectralConv1d:
    """Factory-backed spectral layer to keep module import optional."""

    def __new__(cls, in_channels: int, out_channels: int, modes: int):
        torch, nn = _torch_modules()

        class _SpectralConv1d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.modes = modes
                scale = 1.0 / max(in_channels * out_channels, 1)
                self.weight_real = nn.Parameter(
                    scale * torch.randn(in_channels, out_channels, modes)
                )
                self.weight_imag = nn.Parameter(
                    scale * torch.randn(in_channels, out_channels, modes)
                )

            def forward(self, values: Any) -> Any:
                spectrum = torch.fft.rfft(values, dim=-1)
                retained = min(self.modes, spectrum.shape[-1])
                output = torch.zeros(
                    values.shape[0],
                    self.out_channels,
                    spectrum.shape[-1],
                    device=values.device,
                    dtype=spectrum.dtype,
                )
                weight = torch.complex(
                    self.weight_real[..., :retained],
                    self.weight_imag[..., :retained],
                )
                output[..., :retained] = torch.einsum(
                    "bix,iox->box", spectrum[..., :retained], weight
                )
                return torch.fft.irfft(output, n=values.shape[-1], dim=-1)

        return _SpectralConv1d()


class ParameterEmbeddedFNO1d:
    """Create a trainable FNO conditioned on physical parameters at every layer."""

    def __new__(cls, config: NeuralOperatorConfig):
        torch, nn = _torch_modules()

        class _ParameterEmbeddedFNO1d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.input_projection = nn.Conv1d(
                    config.input_channels, config.width, kernel_size=1
                )
                self.spectral = nn.ModuleList(
                    [SpectralConv1d(config.width, config.width, config.modes) for _ in range(config.layers)]
                )
                self.local = nn.ModuleList(
                    [nn.Conv1d(config.width, config.width, kernel_size=1) for _ in range(config.layers)]
                )
                self.parameter_projection = (
                    nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(config.parameter_dim, config.width),
                                nn.SiLU(),
                                nn.Linear(config.width, 2 * config.width),
                            )
                            for _ in range(config.layers)
                        ]
                    )
                    if config.parameter_dim
                    else None
                )
                self.dropout = nn.Dropout(config.dropout)
                self.output_projection = nn.Sequential(
                    nn.Conv1d(config.width, 2 * config.width, 1),
                    nn.GELU(),
                    nn.Conv1d(2 * config.width, config.output_channels, 1),
                )

            def forward(self, fields: Any, parameters: Any | None = None) -> Any:
                x = self.input_projection(fields)
                if self.parameter_projection is not None:
                    if parameters is None:
                        raise ValueError("parameter-conditioned FNO requires parameters")
                    if parameters.ndim != 2 or parameters.shape[1] != config.parameter_dim:
                        raise ValueError("parameters must have shape [batch, parameter_dim]")
                for index, (spectral, local) in enumerate(zip(self.spectral, self.local, strict=True)):
                    update = spectral(x) + local(x)
                    if self.parameter_projection is not None:
                        affine = self.parameter_projection[index](parameters)
                        scale, shift = affine.chunk(2, dim=-1)
                        update = update * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
                    x = self.dropout(torch.nn.functional.gelu(update))
                return self.output_projection(x)

        return _ParameterEmbeddedFNO1d()


def physics_informed_operator_loss(
    prediction: Any,
    target: Any,
    *,
    residuals: Mapping[str, Any] | None = None,
    data_weight: float = 1.0,
    residual_weight: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    torch, _ = _torch_modules()
    if data_weight < 0.0 or residual_weight < 0.0:
        raise ValueError("operator-loss weights cannot be negative")
    data_loss = torch.mean((prediction - target).square())
    residual_loss = prediction.sum() * 0.0
    if residuals:
        residual_loss = torch.stack(
            [torch.mean(value.square()) for value in residuals.values()]
        ).mean()
    total = data_weight * data_loss + residual_weight * residual_loss
    return total, {
        "data": float(data_loss.detach().cpu().item()),
        "physics": float(residual_loss.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }


@dataclass(slots=True)
class SurrogateGate:
    """Choose a neural operator only inside its validated uncertainty envelope."""

    maximum_relative_uncertainty: float = 0.05
    maximum_extrapolation_score: float = 0.10

    def use_surrogate(
        self,
        *,
        relative_uncertainty: float,
        extrapolation_score: float,
    ) -> bool:
        return (
            relative_uncertainty <= self.maximum_relative_uncertainty
            and extrapolation_score <= self.maximum_extrapolation_score
        )


__all__ = [
    "NeuralOperatorConfig",
    "ParameterEmbeddedFNO1d",
    "SpectralConv1d",
    "SurrogateGate",
    "physics_informed_operator_loss",
]
