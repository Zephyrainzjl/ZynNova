from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, density_values, group_values, make_labels


def _kde(values: np.ndarray, grid: np.ndarray, bandwidth: str | float = "scott") -> np.ndarray:
    try:
        from scipy.stats import gaussian_kde

        return gaussian_kde(values, bw_method=bandwidth)(grid)
    except Exception:
        hist, edges = np.histogram(values, bins=min(40, max(8, int(np.sqrt(values.size)))), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, hist, left=0.0, right=0.0)


@register_plot(category="distributions", aliases=("raincloud",))
def raincloud_plot(
    values: Any,
    *,
    groups: Any | None = None,
    labels: Sequence[str] | None = None,
    orientation: str = "horizontal",
    bandwidth: str | float = "scott",
    cloud_scale: float = 0.36,
    jitter: float = 0.08,
    point_size: float = 10.0,
    show_box: bool = True,
    show_mean: bool = True,
    seed: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Raincloud plot combining half-violin, box summary and raw observations."""
    grouped = group_values(values, groups)
    names = list(grouped) if labels is None else make_labels(labels, len(grouped), "Group")
    cfg = coerce_config(config)
    rng = np.random.default_rng(seed)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        clouds, points, boxes = [], [], []
        for index, (key, data) in enumerate(grouped.items()):
            if data.size == 0:
                continue
            grid = np.linspace(data.min(), data.max(), 256)
            density = _kde(data, grid, bandwidth)
            density = density / max(density.max(), 1e-15) * cloud_scale
            color = resolved.colors[index % len(resolved.colors)]
            if orientation == "horizontal":
                cloud = axis.fill_between(grid, index, index + density, color=color, alpha=0.55, linewidth=0.6)
                jitter_values = index - 0.11 - rng.random(data.size) * jitter
                point = axis.scatter(data, jitter_values, s=point_size, alpha=0.45, color=color, rasterized=cfg.rasterized)
                if show_box:
                    box = axis.boxplot(data, positions=[index], orientation="horizontal", widths=0.12, patch_artist=True, showfliers=False)
                if show_mean:
                    axis.scatter([np.mean(data)], [index], marker="D", s=18, color="black", zorder=5)
            else:
                cloud = axis.fill_betweenx(grid, index, index + density, color=color, alpha=0.55, linewidth=0.6)
                jitter_values = index - 0.11 - rng.random(data.size) * jitter
                point = axis.scatter(jitter_values, data, s=point_size, alpha=0.45, color=color, rasterized=cfg.rasterized)
                if show_box:
                    box = axis.boxplot(data, positions=[index], orientation="vertical", widths=0.12, patch_artist=True, showfliers=False)
                if show_mean:
                    axis.scatter([index], [np.mean(data)], marker="D", s=18, color="black", zorder=5)
            clouds.append(cloud)
            points.append(point)
            if show_box:
                boxes.append(box)
        if orientation == "horizontal":
            axis.set_yticks(np.arange(len(names)), names)
        else:
            axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        return finalize(fig, axis, config=cfg, artists={"clouds": clouds, "points": points, "boxes": boxes}, data={"groups": grouped})


@register_plot(category="distributions", aliases=("ridge", "joyplot"))
def ridge_plot(
    values: Any,
    *,
    groups: Any,
    order: Sequence[str] | None = None,
    bandwidth: str | float = "scott",
    overlap: float = 0.75,
    fill_alpha: float = 0.65,
    normalize: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Ridgeline density plot for temperatures, compositions or cohorts."""
    grouped = group_values(values, groups)
    names = list(grouped) if order is None else [str(item) for item in order]
    all_values = np.concatenate([grouped[name] for name in names])
    grid = np.linspace(np.nanmin(all_values), np.nanmax(all_values), 400)
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        artists = []
        for index, name in enumerate(names):
            density = _kde(grouped[name], grid, bandwidth)
            if normalize:
                density /= max(density.max(), 1e-15)
            baseline = index * (1.0 - overlap)
            color = resolved.colors[index % len(resolved.colors)]
            artist = axis.fill_between(grid, baseline, baseline + density, color=color, alpha=fill_alpha, linewidth=0.8)
            axis.plot(grid, baseline + density, color=color)
            axis.text(grid[0], baseline + 0.05, name, va="bottom", ha="left")
            artists.append(artist)
        axis.set_yticks([])
        return finalize(fig, axis, config=cfg, artists={"ridges": artists}, data={"grid": grid, "groups": grouped})


@register_plot(category="distributions", aliases=("violin-box", "distribution-summary"))
def violin_box_scatter(
    values: Any,
    *,
    groups: Any | None = None,
    labels: Sequence[str] | None = None,
    show_points: bool = True,
    point_alpha: float = 0.35,
    seed: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Layer violin, quartile box and jittered data without seaborn."""
    grouped = group_values(values, groups)
    arrays = list(grouped.values())
    names = list(grouped) if labels is None else make_labels(labels, len(arrays), "Group")
    cfg = coerce_config(config)
    rng = np.random.default_rng(seed)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        violin = axis.violinplot(arrays, showmeans=False, showmedians=False, showextrema=False)
        for index, body in enumerate(violin["bodies"]):
            body.set_facecolor(resolved.colors[index % len(resolved.colors)])
            body.set_edgecolor("black")
            body.set_alpha(0.5)
        box = axis.boxplot(arrays, widths=0.15, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.8)
        scatters = []
        if show_points:
            for index, data in enumerate(arrays, start=1):
                x = index + rng.uniform(-0.08, 0.08, size=data.size)
                scatters.append(axis.scatter(x, data, s=9, alpha=point_alpha, rasterized=cfg.rasterized))
        axis.set_xticks(np.arange(1, len(names) + 1), names, rotation=45 if len(names) > 5 else 0, ha="right" if len(names) > 5 else "center")
        return finalize(fig, axis, config=cfg, artists={"violin": violin, "box": box, "points": scatters}, data={"groups": grouped})


@register_plot(category="distributions", aliases=("ecdf",))
def empirical_cdf(
    values: Any,
    *,
    groups: Any | None = None,
    complementary: bool = False,
    log_x: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Empirical cumulative or survival distribution for robust comparisons."""
    grouped = group_values(values, groups)
    cfg = coerce_config(config, xscale="log" if log_x else None, ylabel="1 − ECDF" if complementary else "ECDF")
    curves = {}
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = []
        for name, data in grouped.items():
            ordered = np.sort(data)
            probability = np.arange(1, ordered.size + 1) / ordered.size
            if complementary:
                probability = 1.0 - probability + 1.0 / ordered.size
            artists.extend(axis.step(ordered, probability, where="post", label=name))
            curves[name] = (ordered, probability)
        return finalize(fig, axis, config=cfg, artists={"curves": artists}, data={"curves": curves})


@register_plot(category="distributions", aliases=("hist-kde",))
def histogram_density(
    values: Any,
    *,
    groups: Any | None = None,
    bins: int | Sequence[float] | str = "auto",
    density: bool = True,
    kde: bool = True,
    stacked: bool = False,
    alpha: float = 0.45,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Histogram and optional KDE overlay with consistent normalization."""
    grouped = group_values(values, groups)
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        arrays = list(grouped.values())
        hist = axis.hist(arrays, bins=bins, density=density, stacked=stacked, alpha=alpha, label=list(grouped))
        kde_artists = []
        if kde:
            all_values = np.concatenate(arrays)
            grid = np.linspace(all_values.min(), all_values.max(), 400)
            for index, (name, data) in enumerate(grouped.items()):
                kde_artists.extend(axis.plot(grid, _kde(data, grid), label=f"{name} KDE", color=resolved.colors[index % len(resolved.colors)]))
        return finalize(fig, axis, config=cfg, artists={"histogram": hist, "kde": kde_artists}, data={"groups": grouped})


@register_plot(category="distributions", aliases=("mirror-histogram",))
def mirrored_histogram(
    upper_values: Any,
    lower_values: Any,
    *,
    bins: int | Sequence[float] = 30,
    upper_label: str = "Upper",
    lower_label: str = "Lower",
    log_x: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Mirrored histograms for comparing uncertainty and true error distributions."""
    top = as_array(upper_values)
    bottom = as_array(lower_values)
    combined = np.concatenate([top[np.isfinite(top)], bottom[np.isfinite(bottom)]])
    edges = np.histogram_bin_edges(combined, bins=bins)
    top_hist, _ = np.histogram(top, bins=edges, density=True)
    bottom_hist, _ = np.histogram(bottom, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    cfg = coerce_config(config, xscale="log" if log_x else None)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        top_artist = axis.bar(centers, top_hist, width=widths, alpha=0.6, label=upper_label, color=resolved.colors[0])
        bottom_artist = axis.bar(centers, -bottom_hist, width=widths, alpha=0.6, label=lower_label, color=resolved.colors[1])
        axis.axhline(0.0, color="black", linewidth=0.8)
        ticks = axis.get_yticks()
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{abs(value):g}" for value in ticks])
        return finalize(fig, axis, config=cfg, artists={"upper": top_artist, "lower": bottom_artist}, data={"edges": edges, "upper_density": top_hist, "lower_density": bottom_hist})
