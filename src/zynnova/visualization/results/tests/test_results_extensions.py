from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from zynnova.visualization import results as zv


def test_catalog_and_themes() -> None:
    assert len(zv.plot_catalog()) >= 162
    assert {"battery", "phase-field", "multiscale"}.issubset(
        {item.category for item in zv.plot_catalog()}
    )
    assert {"battery", "phase-field", "thermal", "monochrome"}.issubset(
        set(zv.available_themes())
    )


def test_user_example_and_partial_restyle(tmp_path) -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=250)
    prediction = reference + rng.normal(scale=0.12, size=250)
    groups = np.where(reference > 0, "positive", "negative")
    result = zv.parity_plot(reference, prediction, groups=groups, mode="hexbin")
    result.save(tmp_path / "parity.pdf")
    assert result.metrics["n"] == 250

    uq = zv.error_vs_uncertainty(np.abs(reference) + 0.01, np.abs(prediction) + 0.01)
    scale = uq.ax.get_xscale()
    uq.restyle(line_width=2.0, legend_loc="upper left")
    assert uq.ax.get_xscale() == scale
    result.close()
    uq.close()


def test_battery_phase_field_and_multiscale_smoke() -> None:
    x = np.linspace(0.0, 1.0, 40)
    y = np.linspace(280.0, 330.0, 25)
    xx, yy = np.meshgrid(x, y)
    field = np.sin(np.pi * xx) * np.cos((yy - 280.0) / 50.0 * np.pi)

    plots = [
        zv.lithium_plating_risk_map(x, y, field, threshold=0.2),
        zv.dendrite_morphology_plot((field > 0).astype(float), concentration=field),
        zv.spatial_error_map(field, field + 0.01, mask=np.abs(field) < 0.05),
        zv.sobol_indices_plot(["a", "b"], [0.2, 0.1], [0.4, 0.3], confidence=np.ones((2, 2)) * 0.02),
    ]
    for result in plots:
        assert result.figure is not None
        result.close()


def test_registered_multi_panel_composition() -> None:
    x = np.linspace(0.0, 1.0, 30)
    panels = [
        zv.PanelSpec("line-with-uncertainty", args=(x, np.sin(x))),
        zv.PanelSpec("stress-strain", args=(x, x * (1 - x))),
    ]
    result = zv.compose_plots(
        panels,
        nrows=1,
        ncols=2,
        shared_legend_enabled=True,
        theme="battery",
    )
    assert np.asarray(result.axes).size == 2
    result.close()
