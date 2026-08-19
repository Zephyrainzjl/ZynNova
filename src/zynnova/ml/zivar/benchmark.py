"""Reproducible latency/memory benchmark and regression gates."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from ._deps import require_torch

torch = require_torch()


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    device: str
    dtype: str
    atoms: int
    warmup: int
    repeats: int
    median_ms: float
    p95_ms: float
    peak_memory_MiB: float | None
    electronic_method: str
    electronic_residual: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackboneComparison:
    local_energy_force_median_ms: float
    zivar_energy_force_median_ms: float
    total_to_local_ratio: float
    repeats: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegisteredBackboneResult:
    kind: str
    parameters: int
    local_energy_force_median_ms: float
    peak_memory_MiB: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_inference(
    model: Any,
    data: dict[str, Any],
    *,
    conditions: dict[str, Any] | None = None,
    warmup: int = 5,
    repeats: int = 20,
) -> BenchmarkResult:
    if warmup < 1 or repeats < 3:
        raise ValueError("benchmark requires warmup>=1 and repeats>=3")
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = None
    with torch.enable_grad():
        for _ in range(warmup):
            output = model.energy_forces_stress(
                data, conditions=conditions, compute_stress=False
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = []
    with torch.enable_grad():
        for _ in range(repeats):
            started = time.perf_counter()
            output = model.energy_forces_stress(
                data, conditions=conditions, compute_stress=False
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - started) * 1000.0)
    if output is None:
        raise RuntimeError("benchmark produced no output")
    ordered = sorted(elapsed)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    parameter = next(model.parameters())
    return BenchmarkResult(
        str(device), str(parameter.dtype).removeprefix("torch."),
        int(data["positions"].shape[0]), warmup, repeats,
        statistics.median(elapsed), p95, peak,
        str(output["electronic_method"]),
        float(output["electronic_residual"].max().detach()),
    )


def assert_latency_regression(
    result: BenchmarkResult,
    baseline_ms: float,
    *,
    allowed_ratio: float = 1.15,
) -> None:
    if baseline_ms <= 0 or allowed_ratio < 1:
        raise ValueError("invalid latency gate")
    if result.median_ms > baseline_ms * allowed_ratio:
        raise AssertionError(
            f"median {result.median_ms:.3f} ms exceeds gate "
            f"{baseline_ms * allowed_ratio:.3f} ms"
        )


def benchmark_against_local_backbone(
    model: Any,
    data: dict[str, Any],
    *,
    conditions: dict[str, Any] | None = None,
    warmup: int = 3,
    repeats: int = 10,
) -> BackboneComparison:
    """Measure stable electronic overhead against the same local force path."""

    if warmup < 1 or repeats < 3:
        raise ValueError("comparison requires warmup>=1 and repeats>=3")
    device = next(model.parameters()).device

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def local_call() -> None:
        positions = data["positions"].detach().requires_grad_(True)
        payload = dict(data)
        payload["positions"] = positions
        output = model.backbone(payload)
        torch.autograd.grad(output["energy"].sum(), positions)

    def full_call() -> None:
        model.energy_forces_stress(
            data, conditions=conditions, compute_stress=False
        )

    with torch.enable_grad():
        for _ in range(warmup):
            local_call()
            full_call()
    synchronize()
    local_elapsed, full_elapsed = [], []
    with torch.enable_grad():
        for _ in range(repeats):
            started = time.perf_counter()
            local_call()
            synchronize()
            local_elapsed.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            full_call()
            synchronize()
            full_elapsed.append((time.perf_counter() - started) * 1000.0)
    local_median = statistics.median(local_elapsed)
    full_median = statistics.median(full_elapsed)
    return BackboneComparison(
        local_median,
        full_median,
        full_median / max(local_median, 1.0e-12),
        repeats,
    )


def benchmark_registered_backbones(
    config: Any,
    atoms: Any,
    *,
    kinds: tuple[str, ...] = ("mace", "zephyr", "zodiac", "convolution"),
    device: Any = "cpu",
    dtype: str = "float32",
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[RegisteredBackboneResult, ...]:
    """Measure identical local energy/force calls across registered adapters."""

    if warmup < 1 or repeats < 3:
        raise ValueError("suite requires warmup>=1 and repeats>=3")
    from .data import atoms_to_batch
    from .model import build_zivar

    results = []
    resolved_dtype = getattr(torch, dtype)
    for kind in kinds:
        backbone = replace(config.backbone, kind=kind)
        if kind == "convolution":
            backbone = replace(backbone, max_ell=0, correlation=1, backend="auto")
        current = replace(config, backbone=backbone)
        torch.manual_seed(0)
        model = build_zivar(current, device=device).to(dtype=resolved_dtype).eval()
        data, _ = atoms_to_batch(atoms, model)
        target_device = next(model.parameters()).device
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)

        def run(local_data: Any = data, local_model: Any = model) -> None:
            positions = local_data["positions"].detach().requires_grad_(True)
            payload = dict(local_data)
            payload["positions"] = positions
            energy = local_model.backbone(payload)["energy"].sum()
            torch.autograd.grad(energy, positions)

        with torch.enable_grad():
            for _ in range(warmup):
                run()
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        elapsed = []
        with torch.enable_grad():
            for _ in range(repeats):
                started = time.perf_counter()
                run()
                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                elapsed.append((time.perf_counter() - started) * 1000.0)
        peak = (
            torch.cuda.max_memory_allocated(target_device) / 2**20
            if target_device.type == "cuda"
            else None
        )
        results.append(
            RegisteredBackboneResult(
                kind=kind,
                parameters=sum(value.numel() for value in model.backbone.parameters()),
                local_energy_force_median_ms=statistics.median(elapsed),
                peak_memory_MiB=peak,
            )
        )
    return tuple(results)


__all__ = [
    "BackboneComparison",
    "BenchmarkResult",
    "RegisteredBackboneResult",
    "assert_latency_regression",
    "benchmark_against_local_backbone",
    "benchmark_inference",
    "benchmark_registered_backbones",
]
