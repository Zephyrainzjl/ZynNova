"""Small executable gallery for the extended ZynNova result API."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from zynnova.visualization import results as zv


def main(output: str | Path = "zynnova_results_gallery") -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    position = np.linspace(0.0, 1.0, 160)
    temperature = np.linspace(273.0, 333.0, 90)
    xx, yy = np.meshgrid(position, temperature)
    risk = 1.4 * xx**2 + ((yy - 298.0) / 35.0) ** 2

    plating = zv.lithium_plating_risk_map(
        position,
        temperature,
        risk,
        threshold=1.0,
        config=zv.PlotConfig(title="Lithium-plating operating boundary"),
    )
    plating.save(output / "plating_risk.pdf")
    plating.close()

    phase = 1.0 / (1.0 + np.exp(-18.0 * (xx - 0.45 - 0.08 * np.sin(yy / 8))))
    morphology = zv.dendrite_morphology_plot(
        phase,
        concentration=1.0 - 0.35 * phase + 0.02 * rng.normal(size=phase.shape),
        potential=-0.1 * phase,
        config=zv.PlotConfig(title="Coupled phase/concentration/potential"),
    )
    morphology.save(output / "phase_field_morphology.pdf")
    morphology.close()

    panels = [
        zv.PanelSpec(
            "electrode-profile",
            args=(position, np.vstack([1 - position, position])),
            kwargs={"labels": ["electrolyte", "solid"]},
            title="Through-plane profiles",
        ),
        zv.PanelSpec(
            "uncertainty-fan",
            args=(position, rng.normal(np.sin(np.pi * position), 0.08, (100, position.size))),
            kwargs={"reference": np.sin(np.pi * position)},
            title="Ensemble uncertainty",
        ),
    ]
    summary = zv.compose_plots(
        panels,
        nrows=1,
        ncols=2,
        figsize=(7.2, 3.1),
        shared_legend_enabled=True,
        panel_label_kwargs={"x": -0.11, "y": 1.04},
        theme="battery",
    )
    summary.save(output / "multiscale_summary.pdf")
    summary.close()


if __name__ == "__main__":
    main()
