from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._core import PlotResult


def synthetic_gallery(*, seed: int = 7, theme: Any = "nature") -> dict[str, PlotResult]:
    """Generate a compact synthetic gallery for API discovery and smoke tests."""
    from .atomistic import free_energy_surface, mean_squared_displacement_plot, radial_distribution_plot
    from .biology import volcano_plot
    from .embeddings import embedding_scatter
    from .model_evaluation import parity_plot
    from .optimization import pareto_front_plot
    from .uncertainty import calibration_plot

    rng = np.random.default_rng(seed)
    reference = rng.normal(size=1000)
    prediction = reference + rng.normal(scale=0.16, size=reference.size)
    radius = np.linspace(0.01, 8, 400)
    rdf = 1 + 2 * np.exp(-((radius - 2.2) / 0.25) ** 2) + 0.6 * np.exp(-((radius - 4.2) / 0.4) ** 2)
    time = np.linspace(0, 20, 200)
    msd = 0.08 * time + 0.01 * rng.normal(size=time.size)
    cv = rng.normal(size=(4000, 2))
    cv[:2000] += [-1.2, 0.5]
    embedding = rng.normal(size=(1200, 2))
    labels = np.repeat(["A", "B", "C"], 400)
    embedding[labels == "B"] += [2.5, 0]
    embedding[labels == "C"] += [1.2, 2.0]
    objectives = rng.random((300, 2))
    probabilities = np.clip(rng.beta(2, 2, 1200), 0, 1)
    outcomes = rng.binomial(1, probabilities)
    fold = rng.normal(scale=1.2, size=1500)
    p = np.clip(rng.random(1500) ** 5, 1e-12, 1)
    return {
        "parity": parity_plot(reference, prediction, theme=theme),
        "rdf": radial_distribution_plot(radius, rdf, theme=theme),
        "msd": mean_squared_displacement_plot(time, msd, fit_range=(8, 20), theme=theme),
        "free_energy": free_energy_surface(cv[:, 0], cv[:, 1], theme=theme),
        "embedding": embedding_scatter(embedding, labels=labels, theme=theme),
        "pareto": pareto_front_plot(objectives, theme=theme),
        "calibration": calibration_plot(outcomes, probabilities, theme=theme),
        "volcano": volcano_plot(fold, p, theme=theme),
    }


def save_gallery(directory: str | Path, *, seed: int = 7, theme: Any = "nature", dpi: int = 220) -> dict[str, Path]:
    """Generate and save the synthetic gallery outside the project tree."""
    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, result in synthetic_gallery(seed=seed, theme=theme).items():
        paths[name] = result.save(output / f"{name}.png", dpi=dpi)
        result.close()
    return paths
