from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, density_values, finite_columns, finite_xy, format_metric


@register_plot(category="uncertainty", aliases=("reliability", "calibration"))
def calibration_plot(
    y_true: Any,
    probability: Any,
    *,
    bins: int = 12,
    strategy: str = "quantile",
    show_histogram: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Reliability diagram with expected calibration error and confidence histogram."""
    truth = as_array(y_true, dtype=int)
    prob = np.clip(as_array(probability), 0.0, 1.0)
    mask = np.isfinite(prob)
    truth, prob = truth[mask], prob[mask]
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1)) if strategy == "quantile" else np.linspace(0, 1, bins + 1)
    edges = np.unique(edges)
    indices = np.clip(np.digitize(prob, edges[1:-1]), 0, max(len(edges) - 2, 0))
    mean_prob, fraction_positive, counts = [], [], []
    for index in range(len(edges) - 1):
        selected = indices == index
        if not np.any(selected):
            continue
        mean_prob.append(np.mean(prob[selected]))
        fraction_positive.append(np.mean(truth[selected]))
        counts.append(np.sum(selected))
    mean_prob = np.asarray(mean_prob)
    fraction_positive = np.asarray(fraction_positive)
    counts = np.asarray(counts)
    ece = float(np.sum(counts / max(counts.sum(), 1) * np.abs(mean_prob - fraction_positive)))
    brier = float(np.mean((prob - truth) ** 2))
    cfg = coerce_config(config, xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1), figsize=(5.0, 5.0 if not show_histogram else 6.0))
    with theme_context(theme):
        if ax is None and show_histogram:
            fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            main, hist_ax = axes
        else:
            fig, main, _ = create_axes(ax=ax, config=cfg, theme=theme)
            axes, hist_ax = main, None
        ideal = main.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8, label="Ideal")[0]
        line = main.plot(mean_prob, fraction_positive, marker="o", label=f"ECE={ece:.3f}; Brier={brier:.3f}")[0]
        bars = None
        if hist_ax is not None:
            bars = hist_ax.hist(prob, bins=np.linspace(0, 1, bins + 1), color=line.get_color(), alpha=0.65)
            hist_ax.set_ylabel("Count")
            hist_ax.set_xlabel(cfg.xlabel)
            main.set_xlabel("")
        return finalize(fig, axes, config=cfg, artists={"ideal": ideal, "calibration": line, "histogram": bars}, data={"mean_probability": mean_prob, "observed_frequency": fraction_positive, "counts": counts}, metrics={"ece": ece, "brier_score": brier})


@register_plot(category="uncertainty", aliases=("error-uncertainty", "uq-hexbin"))
def error_vs_uncertainty(
    error: Any,
    uncertainty: Any,
    *,
    log_scale: bool = True,
    mode: str = "hexbin",
    bins: int = 65,
    ideal: bool = True,
    groups: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Relation between predicted uncertainty and realized error."""
    err, unc, mask = finite_xy(error, uncertainty)
    err, unc = np.abs(err), np.abs(unc)
    group_arr = None if groups is None else np.asarray(groups).reshape(-1)[mask]
    if log_scale:
        positive = (err > 0) & (unc > 0)
        err, unc = err[positive], unc[positive]
        if group_arr is not None:
            group_arr = group_arr[positive]
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(unc, err).statistic)
    except Exception:
        rho = float(np.corrcoef(np.argsort(np.argsort(unc)), np.argsort(np.argsort(err)))[0, 1])
    cfg = coerce_config(config, xlabel="Predicted uncertainty", ylabel="Absolute error", xscale="log" if log_scale else "linear", yscale="log" if log_scale else "linear")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        colorbar = None
        if group_arr is not None:
            points = []
            for group in dict.fromkeys(group_arr.tolist()):
                selected = group_arr == group
                points.append(axis.scatter(unc[selected], err[selected], s=12, alpha=0.6, label=str(group), rasterized=cfg.rasterized))
        elif mode == "hexbin":
            points = axis.hexbin(unc, err, gridsize=bins, bins="log", mincnt=1, cmap="viridis", xscale="log" if log_scale else "linear", yscale="log" if log_scale else "linear")
            colorbar = fig.colorbar(points, ax=axis, label="log count")
        else:
            density = density_values(unc, err)
            points = axis.scatter(unc, err, c=density, s=10, cmap="viridis", alpha=0.65, rasterized=cfg.rasterized)
            colorbar = fig.colorbar(points, ax=axis, label="Point density")
        identity_artist = None
        if ideal and err.size:
            low = max(min(np.min(unc), np.min(err)), np.finfo(float).tiny) if log_scale else min(np.min(unc), np.min(err))
            high = max(np.max(unc), np.max(err))
            identity_artist = axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9, label="Uncertainty = error")[0]
        axis.text(0.04, 0.96, f"Spearman $\\rho$ = {rho:.3f}", transform=axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        return finalize(fig, axis, config=cfg, artists={"points": points, "colorbar": colorbar, "identity": identity_artist}, data={"error": err, "uncertainty": unc, "groups": group_arr}, metrics={"spearman_rho": rho}, theme=theme)


@register_plot(category="uncertainty", aliases=("prediction-band", "interval-coverage"))
def prediction_interval_plot(
    x: Any,
    prediction: Any,
    lower: Any,
    upper: Any,
    *,
    observed: Any | None = None,
    sort: bool = True,
    label: str = "Prediction",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Ordered prediction intervals with observed values and empirical coverage."""
    arrays, _ = finite_columns(x, prediction, lower, upper)
    x_arr, pred, lo, hi = arrays
    obs = None if observed is None else as_array(observed)[: x_arr.size]
    order = np.argsort(x_arr) if sort else np.arange(x_arr.size)
    x_arr, pred, lo, hi = x_arr[order], pred[order], lo[order], hi[order]
    if obs is not None:
        obs = obs[order]
    coverage = float(np.mean((obs >= lo) & (obs <= hi))) if obs is not None else float("nan")
    width = float(np.mean(hi - lo))
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(x_arr, pred, label=label)[0]
        band = axis.fill_between(x_arr, lo, hi, color=line.get_color(), alpha=0.22, label="Prediction interval")
        observed_artist = axis.scatter(x_arr, obs, s=12, color="black", alpha=0.6, label="Observed", rasterized=cfg.rasterized) if obs is not None else None
        if obs is not None:
            axis.text(0.04, 0.96, f"Coverage={coverage:.3f}\nMean width={width:.3g}", transform=axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        return finalize(fig, axis, config=cfg, artists={"prediction": line, "interval": band, "observed": observed_artist}, data={"x": x_arr, "prediction": pred, "lower": lo, "upper": hi, "observed": obs}, metrics={"coverage": coverage, "mean_width": width})


@register_plot(category="uncertainty", aliases=("conformal-coverage",))
def conformal_coverage_plot(
    nominal: Any,
    empirical: Any,
    *,
    group_labels: Sequence[str] | None = None,
    uncertainty: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Nominal versus empirical coverage for conformal or Bayesian intervals."""
    nominal_arr, empirical_arr, _ = finite_xy(nominal, empirical)
    error = None if uncertainty is None else as_array(uncertainty)[: nominal_arr.size]
    miscalibration = float(np.mean(np.abs(nominal_arr - empirical_arr)))
    cfg = coerce_config(config, xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1), equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        if error is None:
            points = axis.scatter(nominal_arr, empirical_arr, s=28)
        else:
            points = axis.errorbar(nominal_arr, empirical_arr, yerr=error, fmt="o", capsize=2)
        ideal = axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)[0]
        if group_labels is not None:
            for x_value, y_value, label in zip(nominal_arr, empirical_arr, group_labels):
                axis.annotate(str(label), (x_value, y_value), xytext=(3, 3), textcoords="offset points")
        axis.text(0.04, 0.96, f"Mean |gap| = {miscalibration:.3f}", transform=axis.transAxes, va="top")
        return finalize(fig, axis, config=cfg, artists={"points": points, "ideal": ideal}, data={"nominal": nominal_arr, "empirical": empirical_arr}, metrics={"mean_absolute_coverage_gap": miscalibration})


@register_plot(category="uncertainty", aliases=("variance-decomposition",))
def uncertainty_decomposition(
    components: Any,
    *,
    labels: Sequence[str],
    x: Any | None = None,
    kind: str = "stacked-area",
    normalize: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Visualize aleatoric, epistemic and misspecification contributions."""
    matrix = np.asarray(components, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] != len(labels):
        raise ValueError("labels must match component count")
    if normalize:
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-15)
    x_arr = np.arange(matrix.shape[0]) if x is None else as_array(x)
    cfg = coerce_config(config, ylabel="Fraction" if normalize else "Uncertainty")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        if kind == "stacked-bar":
            bottom = np.zeros(matrix.shape[0])
            artists = []
            for index, label in enumerate(labels):
                bars = axis.bar(x_arr, matrix[:, index], bottom=bottom, label=label)
                artists.append(bars)
                bottom += matrix[:, index]
        else:
            artists = axis.stackplot(x_arr, matrix.T, labels=list(labels), alpha=0.75)
        return finalize(fig, axis, config=cfg, artists={"components": artists}, data={"x": x_arr, "components": matrix, "labels": list(labels)})
