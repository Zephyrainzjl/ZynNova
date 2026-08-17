from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import (
    as_array,
    bootstrap_interval,
    density_values,
    finite_columns,
    finite_xy,
    make_labels,
    robust_limits,
)


@register_plot(category="statistics", aliases=("trend-band", "uncertainty-band"))
def line_with_uncertainty(
    x: Any,
    y: Any,
    *,
    lower: Any | None = None,
    upper: Any | None = None,
    error: Any | None = None,
    label: str | None = None,
    color: Any = None,
    alpha: float = 0.22,
    line_kwargs: Mapping[str, Any] | None = None,
    band_kwargs: Mapping[str, Any] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Plot a mean trajectory with asymmetric uncertainty or symmetric error."""
    x_arr, y_arr, mask = finite_xy(x, y)
    if error is not None:
        err = as_array(error)[mask]
        lower_arr, upper_arr = y_arr - err, y_arr + err
    else:
        lower_arr = y_arr if lower is None else as_array(lower)[mask]
        upper_arr = y_arr if upper is None else as_array(upper)[mask]
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(x_arr, y_arr, label=label, color=color, **dict(line_kwargs or {}))[0]
        band = axis.fill_between(
            x_arr,
            lower_arr,
            upper_arr,
            color=line.get_color() if color is None else color,
            alpha=alpha,
            linewidth=0,
            **dict(band_kwargs or {}),
        )
        return finalize(
            fig,
            axis,
            config=cfg,
            artists={"line": line, "band": band},
            data={"x": x_arr, "y": y_arr, "lower": lower_arr, "upper": upper_arr},
        )


@register_plot(category="statistics", aliases=("binned-relation", "conditional-trend"))
def binned_trend(
    x: Any,
    y: Any,
    *,
    bins: int | Sequence[float] = 20,
    statistic: str = "mean",
    interval: str = "quantile",
    quantiles: tuple[float, float] = (0.1, 0.9),
    bootstrap_samples: int = 1000,
    show_points: bool = True,
    point_alpha: float = 0.12,
    min_count: int = 3,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Show raw observations and a robust binned conditional trend."""
    x_arr, y_arr, _ = finite_xy(x, y)
    edges = np.histogram_bin_edges(x_arr, bins=bins)
    indices = np.digitize(x_arr, edges[1:-1], right=False)
    centers, values, lower, upper, counts = [], [], [], [], []
    rng = np.random.default_rng(0)
    for index in range(len(edges) - 1):
        bucket = y_arr[indices == index]
        if bucket.size < min_count:
            continue
        centers.append(0.5 * (edges[index] + edges[index + 1]))
        counts.append(bucket.size)
        value = np.median(bucket) if statistic == "median" else np.mean(bucket)
        values.append(value)
        if interval == "bootstrap":
            estimates = np.empty(bootstrap_samples)
            for j in range(bootstrap_samples):
                sample = rng.choice(bucket, size=bucket.size, replace=True)
                estimates[j] = np.median(sample) if statistic == "median" else np.mean(sample)
            lower.append(np.quantile(estimates, quantiles[0]))
            upper.append(np.quantile(estimates, quantiles[1]))
        else:
            lower.append(np.quantile(bucket, quantiles[0]))
            upper.append(np.quantile(bucket, quantiles[1]))
    centers_arr = np.asarray(centers)
    values_arr = np.asarray(values)
    lower_arr = np.asarray(lower)
    upper_arr = np.asarray(upper)
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        scatter = axis.scatter(x_arr, y_arr, s=10, alpha=point_alpha, rasterized=cfg.rasterized) if show_points else None
        line = axis.plot(centers_arr, values_arr, marker="o", label=f"Binned {statistic}")[0]
        band = axis.fill_between(centers_arr, lower_arr, upper_arr, color=line.get_color(), alpha=0.22, linewidth=0)
        return finalize(
            fig,
            axis,
            config=cfg,
            artists={"points": scatter, "line": line, "band": band},
            data={"centers": centers_arr, "values": values_arr, "lower": lower_arr, "upper": upper_arr, "counts": np.asarray(counts)},
        )


@register_plot(category="statistics", aliases=("bland-altman",))
def agreement_plot(
    method_a: Any,
    method_b: Any,
    *,
    confidence: float = 0.95,
    proportional_bias: bool = True,
    label_outliers: Sequence[str] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Bland–Altman agreement plot with limits of agreement and bias trend."""
    a, b, mask = finite_xy(method_a, method_b)
    mean = 0.5 * (a + b)
    difference = b - a
    bias = float(np.mean(difference))
    std = float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0
    z = 1.96 if np.isclose(confidence, 0.95) else 1.96
    lower, upper = bias - z * std, bias + z * std
    cfg = coerce_config(config, xlabel="Mean of methods", ylabel="Difference (B − A)")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        density = density_values(mean, difference)
        points = axis.scatter(mean, difference, c=density, s=18, alpha=0.8, rasterized=cfg.rasterized)
        bias_line = axis.axhline(bias, color="black", linewidth=1.2, label=f"Bias = {bias:.3g}")
        low_line = axis.axhline(lower, color="black", linestyle="--", linewidth=0.9, label=f"LoA = [{lower:.3g}, {upper:.3g}]")
        high_line = axis.axhline(upper, color="black", linestyle="--", linewidth=0.9)
        trend = None
        slope = intercept = float("nan")
        if proportional_bias and mean.size > 1:
            slope, intercept = np.polyfit(mean, difference, 1)
            xx = np.linspace(mean.min(), mean.max(), 100)
            trend = axis.plot(xx, slope * xx + intercept, linestyle=":", label=f"Trend slope = {slope:.3g}")[0]
        if label_outliers is not None:
            labels = np.asarray(label_outliers)[mask]
            outlier_mask = (difference < lower) | (difference > upper)
            for x_value, y_value, text in zip(mean[outlier_mask], difference[outlier_mask], labels[outlier_mask]):
                axis.annotate(str(text), (x_value, y_value), xytext=(3, 3), textcoords="offset points")
        return finalize(
            fig,
            axis,
            config=cfg,
            artists={"points": points, "bias": bias_line, "lower": low_line, "upper": high_line, "trend": trend},
            data={"mean": mean, "difference": difference},
            metrics={"bias": bias, "std_difference": std, "lower_loa": lower, "upper_loa": upper, "proportional_bias_slope": float(slope)},
        )


@register_plot(category="statistics", aliases=("forest", "effect-forest"))
def effect_size_forest(
    effects: Any,
    lower: Any,
    upper: Any,
    labels: Sequence[str],
    *,
    reference: float = 0.0,
    sizes: Any | None = None,
    sort: bool = False,
    annotate: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Forest plot for effects, confidence intervals and optional sample weights."""
    effect, lo, hi = finite_columns(effects, lower, upper)[0]
    label_arr = np.asarray(labels, dtype=object)
    if label_arr.size != effect.size:
        raise ValueError("labels must match effect count")
    size_arr = np.full(effect.size, 35.0) if sizes is None else 20.0 + 80.0 * as_array(sizes) / max(np.nanmax(as_array(sizes)), 1e-12)
    order = np.argsort(effect) if sort else np.arange(effect.size)
    effect, lo, hi, label_arr, size_arr = effect[order], lo[order], hi[order], label_arr[order], size_arr[order]
    y = np.arange(effect.size)
    cfg = coerce_config(config, xlabel="Effect size")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        errors = axis.errorbar(effect, y, xerr=np.vstack([effect - lo, hi - effect]), fmt="none", capsize=2, linewidth=1)
        points = axis.scatter(effect, y, s=size_arr, zorder=3)
        axis.axvline(reference, color="black", linestyle="--", linewidth=0.8)
        axis.set_yticks(y, label_arr)
        axis.invert_yaxis()
        if annotate:
            for value, y_value, left, right in zip(effect, y, lo, hi):
                axis.annotate(f"{value:.3g} [{left:.3g}, {right:.3g}]", (right, y_value), xytext=(5, 0), textcoords="offset points", va="center", fontsize="small")
        return finalize(fig, axis, config=cfg, artists={"errors": errors, "points": points}, data={"effect": effect, "lower": lo, "upper": hi, "labels": label_arr})


@register_plot(category="statistics", aliases=("correlation-heatmap",))
def correlation_matrix(
    data: Any,
    *,
    labels: Sequence[str] | None = None,
    method: str = "pearson",
    cluster: bool = False,
    annotate: bool = True,
    triangle: str | None = None,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Correlation heatmap with optional hierarchical reordering and masking."""
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("data must have shape (n_samples, n_features)")
    if method == "spearman":
        try:
            from scipy.stats import rankdata

            matrix = np.apply_along_axis(rankdata, 0, matrix)
        except Exception:
            matrix = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, matrix)
    corr = np.corrcoef(matrix, rowvar=False)
    names = make_labels(labels, corr.shape[0], "Feature")
    order = np.arange(corr.shape[0])
    if cluster and corr.shape[0] > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform

            distance = np.clip(1.0 - np.abs(corr), 0.0, 2.0)
            order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
            corr = corr[np.ix_(order, order)]
            names = [names[index] for index in order]
        except Exception:
            pass
    masked = corr.copy()
    if triangle == "upper":
        masked[np.triu_indices_from(masked, k=1)] = np.nan
    elif triangle == "lower":
        masked[np.tril_indices_from(masked, k=-1)] = np.nan
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(names)), names)
        if annotate and len(names) <= 30:
            for i in range(masked.shape[0]):
                for j in range(masked.shape[1]):
                    if np.isfinite(masked[i, j]):
                        axis.text(j, i, f"{masked[i, j]:.2f}", ha="center", va="center", fontsize="x-small")
        colorbar = fig.colorbar(image, ax=axis, label=f"{method.title()} correlation")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"correlation": corr, "order": order, "labels": names})


@register_plot(category="statistics", aliases=("quantile-dot",))
def quantile_dotplot(
    values: Any,
    *,
    groups: Any | None = None,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    orientation: str = "horizontal",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Compact quantile glyphs for comparing many distributions."""
    from ._utils import group_values

    grouped = group_values(values, groups)
    names = list(grouped)
    q_values = np.asarray([np.quantile(grouped[name], quantiles) for name in names])
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = []
        for index, row in enumerate(q_values):
            if orientation == "horizontal":
                artists.extend(axis.plot(row, np.full_like(row, index), marker="o"))
                axis.plot([row[0], row[-1]], [index, index], color="0.6", zorder=0)
                axis.plot([row[1], row[-2]], [index, index], linewidth=4, solid_capstyle="round")
            else:
                artists.extend(axis.plot(np.full_like(row, index), row, marker="o"))
                axis.plot([index, index], [row[0], row[-1]], color="0.6", zorder=0)
                axis.plot([index, index], [row[1], row[-2]], linewidth=4, solid_capstyle="round")
        if orientation == "horizontal":
            axis.set_yticks(np.arange(len(names)), names)
        else:
            axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        return finalize(fig, axis, config=cfg, artists={"quantiles": artists}, data={"quantiles": q_values, "levels": np.asarray(quantiles), "groups": names})
