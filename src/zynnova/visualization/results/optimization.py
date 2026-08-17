from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, make_labels, pareto_mask


@register_plot(category="optimization", aliases=("pareto", "pareto-scatter"))
def pareto_front_plot(
    objectives: Any,
    *,
    objective_names: Sequence[str] | None = None,
    maximize: Sequence[bool] | bool = True,
    color: Any | None = None,
    size: Any | float = 18.0,
    labels: Sequence[str] | None = None,
    annotate_front: bool = False,
    connect_front: bool = True,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Two- or three-objective Pareto front with highlighted non-dominated set."""
    matrix = np.asarray(objectives, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] not in (2, 3):
        raise ValueError("objectives must have shape (n_samples, 2 or 3)")
    names = make_labels(objective_names, matrix.shape[1], "Objective")
    mask = pareto_mask(matrix, maximize=maximize)
    cfg = coerce_config(config, xlabel=names[0], ylabel=names[1])
    projection = "3d" if matrix.shape[1] == 3 else None
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme, projection=projection)
        if matrix.shape[1] == 2:
            points = axis.scatter(matrix[:, 0], matrix[:, 1], c=color if color is not None else "0.65", s=size, cmap=cmap if color is not None else None, alpha=0.42, edgecolors="none", rasterized=cfg.rasterized)
            front = axis.scatter(matrix[mask, 0], matrix[mask, 1], s=np.asarray(size)[mask] * 1.5 if np.ndim(size) else float(size) * 1.5, facecolors="none", edgecolors=resolved.colors[1], linewidths=1.3, label="Pareto front")
            front_line = None
            if connect_front and np.sum(mask) > 1:
                order = np.argsort(matrix[mask, 0])
                front_line = axis.plot(matrix[mask, 0][order], matrix[mask, 1][order], color=resolved.colors[1], linewidth=1.0)[0]
        else:
            points = axis.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], c=color if color is not None else "0.65", s=size, cmap=cmap if color is not None else None, alpha=0.42)
            front = axis.scatter(matrix[mask, 0], matrix[mask, 1], matrix[mask, 2], s=np.asarray(size)[mask] * 1.5 if np.ndim(size) else float(size) * 1.5, facecolors="none", edgecolors=resolved.colors[1], linewidths=1.3, label="Pareto front")
            axis.set_zlabel(names[2])
            front_line = None
        if color is not None and np.issubdtype(np.asarray(color).dtype, np.number):
            colorbar = fig.colorbar(points, ax=axis, label="Color value")
        else:
            colorbar = None
        if annotate_front and labels is not None:
            label_arr = np.asarray(labels)
            for index in np.flatnonzero(mask):
                if matrix.shape[1] == 2:
                    axis.annotate(str(label_arr[index]), matrix[index, :2], xytext=(3, 3), textcoords="offset points")
                else:
                    axis.text(*matrix[index, :3], str(label_arr[index]))
        return finalize(fig, axis, config=cfg, artists={"points": points, "front": front, "front_line": front_line, "colorbar": colorbar}, data={"objectives": matrix, "pareto_mask": mask, "objective_names": names}, metrics={"pareto_count": int(mask.sum()), "candidate_count": int(mask.size)})


@register_plot(category="optimization", aliases=("parallel-coordinates",))
def parallel_coordinates_plot(
    values: Any,
    *,
    dimensions: Sequence[str] | None = None,
    color: Any | None = None,
    labels: Sequence[str] | None = None,
    normalize: bool = True,
    highlight: Any | None = None,
    alpha: float = 0.2,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Parallel-coordinates plot for composition–process–property spaces."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (samples, dimensions)")
    names = make_labels(dimensions, matrix.shape[1], "Dimension")
    sample_labels = make_labels(labels, matrix.shape[0], "Sample")
    minimum, maximum = np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0)
    normalized = (matrix - minimum) / np.maximum(maximum - minimum, 1e-15) if normalize else matrix.copy()
    scalar = np.arange(matrix.shape[0]) if color is None else as_array(color)
    if scalar.size != matrix.shape[0]:
        raise ValueError("color must match samples")
    highlight_mask = np.zeros(matrix.shape[0], dtype=bool) if highlight is None else np.asarray(highlight, dtype=bool)
    if highlight_mask.size != matrix.shape[0]:
        raise ValueError("highlight must match samples")
    x = np.arange(matrix.shape[1])
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=np.nanmin(scalar), vmax=np.nanmax(scalar))
        cmap_obj = __import__("matplotlib").colormaps[cmap]
        lines, annotations = [], []
        for index, row in enumerate(normalized):
            line = axis.plot(x, row, color=cmap_obj(norm(scalar[index])), alpha=0.9 if highlight_mask[index] else alpha, linewidth=1.8 if highlight_mask[index] else 0.7, label=sample_labels[index] if highlight_mask[index] else "_nolegend_")[0]
            lines.append(line)
            if highlight_mask[index]:
                annotations.append(axis.annotate(sample_labels[index], (x[-1], row[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize="x-small", color=line.get_color()))
        axis.set_xticks(x, names, rotation=30, ha="right")
        if normalize:
            axis.set_ylim(0, 1); axis.set_ylabel("Normalized value")
        return finalize(fig, axis, config=cfg, artists={"lines": lines, "labels": annotations}, data={"values": matrix, "normalized": normalized, "dimensions": names, "labels": sample_labels, "minimum": minimum, "maximum": maximum}, theme=theme)


@register_plot(category="optimization", aliases=("ternary", "composition-ternary"))
def ternary_composition_plot(
    compositions: Any,
    *,
    property_values: Any | None = None,
    component_names: Sequence[str] = ("A", "B", "C"),
    normalize: bool = True,
    point_size: Any | float = 25.0,
    cmap: str = "viridis",
    contours: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Dependency-free ternary composition–property map."""
    composition = np.asarray(compositions, dtype=float)
    if composition.ndim != 2 or composition.shape[1] != 3:
        raise ValueError("compositions must have shape (samples, 3)")
    if normalize:
        composition = composition / np.maximum(composition.sum(axis=1, keepdims=True), 1e-15)
    a, b, c = composition.T
    x, y = b + 0.5 * c, np.sqrt(3.0) / 2.0 * c
    prop = None if property_values is None else as_array(property_values)
    if prop is not None and prop.size != composition.shape[0]:
        raise ValueError("property_values must match compositions")
    cfg = coerce_config(config, equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        triangle = axis.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="black", linewidth=0.9)[0]
        contour_artist = None
        if contours and prop is not None and prop.size >= 4:
            try:
                contour_artist = axis.tricontourf(x, y, prop, levels=15, cmap=cmap, alpha=0.55)
            except Exception:
                contour_artist = axis.tricontour(x, y, prop, levels=8, cmap=cmap, linewidths=0.7)
        points = axis.scatter(x, y, c=prop, s=point_size, cmap=cmap, edgecolors="none", alpha=0.85, rasterized=cfg.rasterized)
        axis.text(-0.03, -0.03, component_names[0], ha="right", va="top")
        axis.text(1.03, -0.03, component_names[1], ha="left", va="top")
        axis.text(0.5, np.sqrt(3) / 2 + 0.03, component_names[2], ha="center", va="bottom")
        grid_lines=[]
        for fraction in (0.2, 0.4, 0.6, 0.8):
            grid_lines.extend(axis.plot([fraction, 1 - fraction / 2], [0, np.sqrt(3) / 2 * fraction], color="0.85", linewidth=0.4))
            grid_lines.extend(axis.plot([fraction / 2, 1 - fraction], [np.sqrt(3) / 2 * fraction, 0], color="0.85", linewidth=0.4))
            grid_lines.extend(axis.plot([fraction / 2, 1 - fraction / 2], [np.sqrt(3) / 2 * fraction, np.sqrt(3) / 2 * fraction], color="0.85", linewidth=0.4))
        axis.set_axis_off()
        colorbar = fig.colorbar(points, ax=axis, label="Property") if prop is not None else None
        return finalize(fig, axis, config=cfg, artists={"triangle": triangle, "grid": grid_lines, "contours": contour_artist, "points": points, "colorbar": colorbar}, data={"compositions": composition, "cartesian": np.c_[x, y], "property": prop}, theme=theme)


@register_plot(category="optimization", aliases=("active-learning-progress", "discovery-progress"))
def active_learning_progress(
    iteration: Any,
    best_value: Any,
    *,
    evaluated_value: Any | None = None,
    uncertainty: Any | None = None,
    random_baseline: Any | None = None,
    maximize: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Active-learning discovery progress with best-so-far and uncertainty."""
    x = as_array(iteration)
    values = as_array(best_value)
    best = np.maximum.accumulate(values) if maximize else np.minimum.accumulate(values)
    cfg = coerce_config(config, xlabel="Iteration", ylabel="Best objective")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        evaluated = axis.scatter(x, as_array(evaluated_value), s=12, alpha=0.35, label="Evaluated") if evaluated_value is not None else None
        best_line = axis.step(x, best, where="post", label="Best so far")[0]
        band = None
        if uncertainty is not None:
            unc = as_array(uncertainty)
            band = axis.fill_between(x, best - unc, best + unc, step="post", color=best_line.get_color(), alpha=0.18)
        baseline_line = axis.plot(x, as_array(random_baseline), linestyle="--", label="Random baseline")[0] if random_baseline is not None else None
        return finalize(fig, axis, config=cfg, artists={"evaluated": evaluated, "best": best_line, "uncertainty": band, "baseline": baseline_line}, data={"iteration": x, "best": best, "evaluated": evaluated_value, "uncertainty": uncertainty})


@register_plot(category="optimization", aliases=("acquisition-map",))
def acquisition_landscape(
    x: Any,
    y: Any,
    acquisition: Any,
    *,
    evaluated: Any | None = None,
    proposed: Any | None = None,
    levels: int = 30,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Two-dimensional Bayesian-optimization acquisition landscape."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z = np.asarray(acquisition, dtype=float)
    if x_arr.ndim == 1 and y_arr.ndim == 1:
        xx, yy = np.meshgrid(x_arr, y_arr)
    else:
        xx, yy = x_arr, y_arr
    if z.shape != xx.shape:
        raise ValueError("acquisition shape must match mesh")
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        surface = axis.contourf(xx, yy, z, levels=levels, cmap=cmap)
        colorbar = fig.colorbar(surface, ax=axis, label="Acquisition")
        evaluated_artist = None
        proposed_artist = None
        if evaluated is not None:
            evaluated_arr = np.asarray(evaluated, dtype=float)
            evaluated_artist = axis.scatter(evaluated_arr[:, 0], evaluated_arr[:, 1], marker="x", color="white", linewidths=1.0, label="Evaluated")
        if proposed is not None:
            proposed_arr = np.asarray(proposed, dtype=float)
            proposed_artist = axis.scatter(proposed_arr[:, 0], proposed_arr[:, 1], marker="*", s=90, color="red", edgecolors="white", label="Proposed")
        return finalize(fig, axis, config=cfg, artists={"surface": surface, "colorbar": colorbar, "evaluated": evaluated_artist, "proposed": proposed_artist}, data={"x": xx, "y": yy, "acquisition": z})


@register_plot(category="optimization", aliases=("hypervolume",))
def hypervolume_curve(
    iteration: Any,
    hypervolume: Any,
    *,
    methods: Any | None = None,
    uncertainty: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Hypervolume versus query budget for multi-objective optimization."""
    x = as_array(iteration)
    values = np.asarray(hypervolume, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    labels = make_labels(None if methods is None else list(methods), values.shape[1], "Method")
    unc = None if uncertainty is None else np.asarray(uncertainty, dtype=float)
    cfg = coerce_config(config, xlabel="Iteration", ylabel="Hypervolume")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, bands = [], []
        for index, label in enumerate(labels):
            line = axis.plot(x, values[:, index], label=label)[0]
            lines.append(line)
            if unc is not None:
                error = unc[:, index] if unc.ndim > 1 else unc
                bands.append(axis.fill_between(x, values[:, index] - error, values[:, index] + error, color=line.get_color(), alpha=0.18))
        return finalize(fig, axis, config=cfg, artists={"lines": lines, "bands": bands}, data={"iteration": x, "hypervolume": values, "labels": labels})
