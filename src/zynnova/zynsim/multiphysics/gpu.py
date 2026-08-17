"""Torch/CUDA field backend for batched transport and thermal updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GPUBackendConfig:
    device: str = "auto"
    dtype: str = "float32"
    compile_kernels: bool = False


class GPUFieldBackend:
    """Vectorized field kernels usable on CUDA, MPS, or CPU."""

    def __init__(self, config: GPUBackendConfig | None = None) -> None:
        self.config = config or GPUBackendConfig()
        try:
            import torch
        except ImportError as exc:
            raise ImportError("GPUFieldBackend requires PyTorch; install zynnova[zynsim-gpu]") from exc
        self.torch = torch
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
        try:
            self.dtype = getattr(torch, self.config.dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch dtype {self.config.dtype!r}") from exc

    def tensor(self, values: Any, *, requires_grad: bool = False):
        return self.torch.as_tensor(
            values, device=self.device, dtype=self.dtype
        ).requires_grad_(requires_grad)

    def explicit_diffusion_step(
        self,
        field: Any,
        diffusivity: Any,
        dt_s: float,
        spacing_m: tuple[float, ...],
    ) -> Any:
        torch = self.torch
        values = self.tensor(field) if not torch.is_tensor(field) else field.to(self.device, self.dtype)
        diffusion = self.tensor(diffusivity) if not torch.is_tensor(diffusivity) else diffusivity.to(self.device, self.dtype)
        if values.ndim < 2 or len(spacing_m) != values.ndim - 1:
            raise ValueError("field must have leading batch and one spacing per spatial dimension")
        laplacian = torch.zeros_like(values)
        for spatial_axis, spacing in enumerate(spacing_m, start=1):
            if spacing <= 0.0:
                raise ValueError("grid spacing must be positive")
            laplacian = laplacian + (
                torch.roll(values, 1, dims=spatial_axis)
                - 2.0 * values
                + torch.roll(values, -1, dims=spatial_axis)
            ) / spacing**2
        return values + float(dt_s) * diffusion * laplacian

    def thermal_network_step(
        self,
        temperature_K: Any,
        heat_W: Any,
        conductance_W_K: Any,
        heat_capacity_J_K: Any,
        ambient_temperature_K: float,
        cooling_W_K: Any,
        dt_s: float,
    ) -> Any:
        torch = self.torch
        temperature = self.tensor(temperature_K) if not torch.is_tensor(temperature_K) else temperature_K.to(self.device, self.dtype)
        heat = self.tensor(heat_W) if not torch.is_tensor(heat_W) else heat_W.to(self.device, self.dtype)
        conductance = self.tensor(conductance_W_K) if not torch.is_tensor(conductance_W_K) else conductance_W_K.to(self.device, self.dtype)
        capacity = self.tensor(heat_capacity_J_K) if not torch.is_tensor(heat_capacity_J_K) else heat_capacity_J_K.to(self.device, self.dtype)
        cooling = self.tensor(cooling_W_K) if not torch.is_tensor(cooling_W_K) else cooling_W_K.to(self.device, self.dtype)
        exchange = torch.matmul(conductance, temperature.unsqueeze(-1)).squeeze(-1) - torch.sum(conductance, dim=-1) * temperature
        return temperature + dt_s * (
            heat + exchange + cooling * (ambient_temperature_K - temperature)
        ) / capacity

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


__all__ = ["GPUBackendConfig", "GPUFieldBackend"]
