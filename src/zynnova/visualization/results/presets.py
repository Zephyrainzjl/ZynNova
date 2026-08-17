from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context


def mlip_validation_figure(
    *,
    energy_reference: Any,
    energy_prediction: Any,
    force_reference: Any,
    force_prediction: Any,
    uncertainty: Any | None = None,
    force_error: Any | None = None,
    training_history: Mapping[str, Any] | None = None,
    config: PlotConfig | None = None,
    theme: Any = "nature",
) -> PlotResult:
    """Four-panel preset for a machine-learning interatomic-potential paper."""
    from .model_evaluation import parity_plot
    from .training import training_history_plot
    from .uncertainty import error_vs_uncertainty

    cfg = coerce_config(config, figsize=(8.0, 7.0))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=2)
        energy = parity_plot(energy_reference, energy_prediction, mode="density", ax=axes[0, 0], config=PlotConfig(title="Energy", equal_aspect=True, legend=False), theme=theme)
        force = parity_plot(np.asarray(force_reference).reshape(-1), np.asarray(force_prediction).reshape(-1), mode="hexbin", ax=axes[0, 1], config=PlotConfig(title="Force components", equal_aspect=True, legend=False), theme=theme)
        uncertainty_result = None
        if uncertainty is not None:
            error = np.abs(np.asarray(force_prediction).reshape(-1) - np.asarray(force_reference).reshape(-1)) if force_error is None else force_error
            uncertainty_result = error_vs_uncertainty(error, uncertainty, ax=axes[1, 0], config=PlotConfig(title="Uncertainty calibration", legend=False), theme=theme)
        else:
            axes[1, 0].axis("off")
        history_result = None
        if training_history:
            history_result = training_history_plot(training_history, ax=None, config=PlotConfig(title="Training", legend=True), theme=theme)
            # Copy lines into the designated panel to preserve a single figure.
            for line in history_result.ax.lines:
                axes[1, 1].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), linestyle=line.get_linestyle())
            axes[1, 1].set_title("Training")
            axes[1, 1].set_xlabel("Epoch / step")
            axes[1, 1].set_ylabel("Metric")
            history_result.close()
        else:
            axes[1, 1].axis("off")
        from .panels import add_panel_labels

        add_panel_labels(axes.flat)
        metrics = {f"energy_{key}": value for key, value in energy.metrics.items()}
        metrics.update({f"force_{key}": value for key, value in force.metrics.items()})
        if uncertainty_result:
            metrics.update({f"uncertainty_{key}": value for key, value in uncertainty_result.metrics.items()})
        return finalize(fig, axes, config=cfg, metrics=metrics)


def atomistic_dynamics_figure(
    *,
    time: Any,
    msd: Any,
    radius: Any,
    rdf: Any,
    cv_x: Any,
    cv_y: Any,
    total_energy: Any,
    config: PlotConfig | None = None,
    theme: Any = "nature",
) -> PlotResult:
    """Four-panel RDF–MSD–free-energy–conservation MD summary."""
    from .atomistic import energy_conservation_plot, free_energy_surface, mean_squared_displacement_plot, radial_distribution_plot
    from .panels import add_panel_labels

    cfg = coerce_config(config, figsize=(8.0, 7.0))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=2)
        rdf_result = radial_distribution_plot(radius, rdf, ax=axes[0, 0], config=PlotConfig(title="Radial distribution"), theme=theme)
        msd_result = mean_squared_displacement_plot(time, msd, ax=axes[0, 1], config=PlotConfig(title="Mean squared displacement"), theme=theme)
        fes_result = free_energy_surface(cv_x, cv_y, ax=axes[1, 0], config=PlotConfig(title="Free-energy surface", legend=False), theme=theme)
        energy_result = energy_conservation_plot(time, total_energy, ax=axes[1, 1], config=PlotConfig(title="Energy conservation"), theme=theme)
        add_panel_labels(axes.flat)
        return finalize(fig, axes, config=cfg, artists={"rdf": rdf_result.artists, "msd": msd_result.artists, "free_energy": fes_result.artists, "energy": energy_result.artists}, metrics={**msd_result.metrics, **energy_result.metrics})


def materials_discovery_figure(
    *,
    objectives: Any,
    embedding: Any,
    property_values: Any,
    composition: Any,
    formation_energy: Any,
    active_iteration: Any,
    best_value: Any,
    config: PlotConfig | None = None,
    theme: Any = "nature",
) -> PlotResult:
    """Pareto–latent–convex-hull–active-learning discovery figure."""
    from .embeddings import embedding_scatter
    from .materials import convex_hull_plot
    from .optimization import active_learning_progress, pareto_front_plot
    from .panels import add_panel_labels

    cfg = coerce_config(config, figsize=(8.0, 7.0))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=2)
        pareto = pareto_front_plot(objectives, color=property_values, ax=axes[0, 0], config=PlotConfig(title="Pareto front"), theme=theme)
        latent = embedding_scatter(embedding, color=property_values, ax=axes[0, 1], config=PlotConfig(title="Latent design space", legend=False), theme=theme)
        hull = convex_hull_plot(composition, formation_energy, ax=axes[1, 0], config=PlotConfig(title="Thermodynamic hull"), theme=theme)
        progress = active_learning_progress(active_iteration, best_value, ax=axes[1, 1], config=PlotConfig(title="Discovery progress"), theme=theme)
        add_panel_labels(axes.flat)
        return finalize(fig, axes, config=cfg, artists={"pareto": pareto.artists, "latent": latent.artists, "hull": hull.artists, "progress": progress.artists})
