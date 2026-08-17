from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_xy, make_labels


def _field2d(values: Any, name: str = "field") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return array


def _coordinates(field: np.ndarray, x: Any | None, y: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        return np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim == y_arr.ndim == 1:
        if field.shape != (y_arr.size, x_arr.size):
            raise ValueError("field shape must match y and x")
        return np.meshgrid(x_arr, y_arr)
    if x_arr.shape != field.shape or y_arr.shape != field.shape:
        raise ValueError("2-D coordinates must match field")
    return x_arr, y_arr


@register_plot(category="phase-field", aliases=("dendrite-morphology", "phase-dendrite"))
def dendrite_morphology_plot(
    phase: Any,
    *,
    concentration: Any | None = None,
    potential: Any | None = None,
    x: Any | None = None,
    y: Any | None = None,
    interface_level: float = 0.5,
    cmap: str = "magma",
    field_cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Dendrite phase morphology with concentration or potential backdrop."""
    eta = _field2d(phase, "phase")
    backdrop = eta if concentration is None and potential is None else _field2d(concentration if concentration is not None else potential, "backdrop")
    if backdrop.shape != eta.shape:
        raise ValueError("phase and backdrop fields must have equal shape")
    xx, yy = _coordinates(eta, x, y)
    cfg = coerce_config(config, xlabel="x", ylabel="y", equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.pcolormesh(xx, yy, backdrop, cmap=field_cmap if backdrop is not eta else cmap, shading="auto")
        interface = axis.contour(xx, yy, eta, levels=[interface_level], colors="white", linewidths=1.1)
        phase_fill = axis.contourf(xx, yy, eta, levels=[interface_level, np.nanmax(eta) + 1e-12], colors=["black"], alpha=0.18)
        label = "Concentration" if concentration is not None else "Potential" if potential is not None else "Phase field"
        colorbar = fig.colorbar(image, ax=axis, label=label)
        solid_fraction = float(np.mean(eta >= interface_level))
        return finalize(fig, axis, config=cfg, artists={"backdrop": image, "interface": interface, "solid": phase_fill, "colorbar": colorbar}, data={"phase": eta, "backdrop": backdrop, "x": xx, "y": yy}, metrics={"solid_fraction": solid_fraction}, theme=theme)


@register_plot(category="phase-field", aliases=("phase-boundary", "interface-map"))
def phase_boundary_plot(
    phase: Any,
    *,
    levels: Sequence[float] = (0.1, 0.5, 0.9),
    x: Any | None = None,
    y: Any | None = None,
    background: Any | None = None,
    cmap: str = "coolwarm",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Diffuse-interface contours over a phase or coupled-field background."""
    eta = _field2d(phase, "phase")
    base = eta if background is None else _field2d(background, "background")
    if base.shape != eta.shape:
        raise ValueError("background must match phase")
    xx, yy = _coordinates(eta, x, y)
    cfg = coerce_config(config, xlabel="x", ylabel="y", equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.pcolormesh(xx, yy, base, cmap=cmap, shading="auto")
        contours = axis.contour(xx, yy, eta, levels=levels, colors="black", linewidths=np.linspace(0.5, 1.3, len(levels)))
        labels = axis.clabel(contours, inline=True, fontsize="x-small", fmt=lambda value: f"η={value:g}")
        colorbar = fig.colorbar(image, ax=axis, label="Background field")
        return finalize(fig, axis, config=cfg, artists={"background": image, "boundaries": contours, "labels": labels, "colorbar": colorbar}, data={"phase": eta, "background": base}, theme=theme)


@register_plot(category="phase-field", aliases=("phase-free-energy-landscape", "double-well"))
def free_energy_landscape_plot(
    order_parameter: Any,
    free_energy: Any,
    *,
    chemical_potential: Any | None = None,
    common_tangent: tuple[float, float] | None = None,
    minima: Sequence[float] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Bulk free-energy density, common tangent and chemical potential."""
    eta, energy, _ = finite_xy(order_parameter, free_energy)
    order = np.argsort(eta)
    eta, energy = eta[order], energy[order]
    cfg = coerce_config(config, xlabel="Order parameter / composition", ylabel="Free-energy density")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(eta, energy, label="Free energy")[0]
        tangent = None
        if common_tangent is not None:
            left, right = common_tangent
            f_left = np.interp(left, eta, energy)
            f_right = np.interp(right, eta, energy)
            tangent = axis.plot([left, right], [f_left, f_right], linestyle="--", color="black", label="Common tangent")[0]
        minimum_artists = []
        for value in minima or ():
            minimum_artists.append(axis.scatter([value], [np.interp(value, eta, energy)], marker="o", zorder=4, label=f"Minimum {value:g}"))
        mu_axis = None
        mu_line = None
        if chemical_potential is not None:
            mu = as_array(chemical_potential)[order]
            mu_axis = axis.twinx()
            mu_line = mu_axis.plot(eta, mu, color="#D1495B", alpha=0.75, label="Chemical potential")[0]
            mu_axis.set_ylabel("Chemical potential")
        return finalize(fig, axis, config=cfg, artists={"free_energy": line, "common_tangent": tangent, "minima": minimum_artists, "chemical_potential_axis": mu_axis, "chemical_potential": mu_line}, data={"order_parameter": eta, "free_energy": energy, "chemical_potential": chemical_potential}, theme=theme)


@register_plot(category="phase-field", aliases=("chemical-potential-profile", "mu-profile"))
def chemical_potential_profile_plot(
    position: Any,
    chemical_potential: Any,
    *,
    phase: Any | None = None,
    concentration: Any | None = None,
    interface_level: float = 0.5,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Chemical-potential profile with diffuse-interface locations."""
    x, mu, mask = finite_xy(position, chemical_potential)
    cfg = coerce_config(config, xlabel="Position", ylabel="Chemical potential")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(x, mu, label="Chemical potential")[0]
        interfaces = []
        if phase is not None:
            eta = as_array(phase)[mask]
            crossings = np.flatnonzero(np.diff(np.sign(eta - interface_level)) != 0)
            for index in crossings:
                interfaces.append(axis.axvline(x[index], color="0.4", linestyle="--", linewidth=0.8, label="Interface" if not interfaces else None))
        concentration_axis = None
        concentration_line = None
        if concentration is not None:
            c = as_array(concentration)[mask]
            concentration_axis = axis.twinx()
            concentration_line = concentration_axis.plot(x, c, color="#2A9D8F", alpha=0.7, label="Concentration")[0]
            concentration_axis.set_ylabel("Concentration")
        return finalize(fig, axis, config=cfg, artists={"chemical_potential": line, "interfaces": interfaces, "concentration_axis": concentration_axis, "concentration": concentration_line}, data={"position": x, "chemical_potential": mu, "phase": phase, "concentration": concentration}, theme=theme)


@register_plot(category="phase-field", aliases=("interfacial-energy-polar", "surface-energy-anisotropy"))
def interfacial_energy_polar_plot(
    angle: Any,
    energy: Any,
    *,
    stiffness: Any | None = None,
    labels: Sequence[str] | None = None,
    fill_alpha: float = 0.12,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Polar interfacial-energy anisotropy and optional surface stiffness."""
    theta = as_array(angle)
    matrix = np.asarray(energy, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != theta.size:
        raise ValueError("energy curves must match angle")
    names = make_labels(labels, matrix.shape[0], "Interface")
    cfg = coerce_config(config, figsize=(5.0, 5.0))
    with theme_context(theme):
        fig, axis, _ = create_axes(config=cfg, theme=theme, projection="polar")
        lines, fills = [], []
        for row, name in zip(matrix, names):
            line = axis.plot(theta, row, label=name)[0]
            lines.append(line)
            fills.append(axis.fill(theta, row, color=line.get_color(), alpha=fill_alpha))
        stiffness_line = None
        if stiffness is not None:
            stiffness_line = axis.plot(theta, as_array(stiffness), color="black", linestyle="--", label="Surface stiffness")[0]
        return finalize(fig, axis, config=cfg, artists={"energy": lines, "fills": fills, "stiffness": stiffness_line}, data={"angle": theta, "energy": matrix, "stiffness": stiffness}, theme=theme)


@register_plot(category="phase-field", aliases=("dendrite-tip-velocity", "front-velocity"))
def dendrite_tip_velocity_plot(
    time: Any,
    tip_position: Any,
    *,
    smooth_window: int = 1,
    labels: Sequence[str] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Dendrite-tip position and derived velocity versus time."""
    t = as_array(time)
    matrix = np.asarray(tip_position, dtype=float)
    if matrix.ndim == 1: matrix = matrix[None, :]
    if matrix.shape[1] != t.size: raise ValueError("tip positions must match time")
    if smooth_window > 1:
        kernel = np.ones(int(smooth_window)) / int(smooth_window)
        smooth = np.asarray([np.convolve(row, kernel, mode="same") for row in matrix])
    else: smooth = matrix
    velocity = np.gradient(smooth, t, axis=1)
    names = make_labels(labels, matrix.shape[0], "Condition")
    cfg = coerce_config(config, figsize=(6.2, 5.0))
    with theme_context(theme):
        if ax is None:
            fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=1, sharex=True)
            axes = np.atleast_1d(axes)
        else:
            axes = np.atleast_1d(ax)
            if axes.size == 1:
                primary = axes[0]; secondary = primary.twinx(); axes = np.asarray([primary, secondary], dtype=object)
            elif axes.size != 2:
                raise ValueError("ax must be one axis or a two-axis sequence")
            fig = axes[0].figure
        position_lines, velocity_lines = [], []
        for row, vel, name in zip(smooth, velocity, names):
            line = axes[0].plot(t, row, label=name)[0]
            position_lines.append(line)
            velocity_lines.append(axes[1].plot(t, vel, color=line.get_color(), linestyle="--", label=name)[0])
        axes[0].set_ylabel("Tip position")
        axes[1].set_ylabel("Tip velocity")
        axes[-1].set_xlabel("Time")
        return finalize(fig, axes, config=cfg, artists={"position": position_lines, "velocity": velocity_lines}, data={"time": t, "tip_position": matrix, "smoothed_position": smooth, "velocity": velocity}, metrics={"maximum_velocity": float(np.nanmax(velocity))}, theme=theme)


@register_plot(category="phase-field", aliases=("morphology-metrics", "dendrite-metrics"))
def morphology_metrics_plot(
    time: Any,
    metrics: Mapping[str, Any],
    *,
    normalize: bool = False,
    columns: int = 2,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Small-multiple morphology metrics such as roughness and branch count."""
    t = as_array(time)
    if not metrics:
        raise ValueError("metrics cannot be empty")
    rows = int(np.ceil(len(metrics) / columns))
    cfg = coerce_config(config, figsize=(3.3 * columns, 2.4 * rows))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=rows, ncols=columns, squeeze=False, sharex=True)
        lines = {}
        for axis, (name, values) in zip(axes.flat, metrics.items()):
            y = as_array(values)
            if y.size != t.size:
                raise ValueError(f"metric {name!r} must match time")
            if normalize:
                y = y / max(abs(y[0]), 1e-15)
            lines[name] = axis.plot(t, y, label=name)[0]
            axis.set_title(name)
            axis.set_ylabel("Normalized" if normalize else name)
        for axis in list(axes.flat)[len(metrics):]:
            axis.set_visible(False)
        for axis in axes[-1, :]:
            if axis.get_visible():
                axis.set_xlabel("Time")
        return finalize(fig, axes, config=cfg, artists={"metrics": lines}, data={"time": t, "metrics": dict(metrics)}, theme=theme)


@register_plot(category="phase-field", aliases=("nucleation-growth-map", "nucleation-regime"))
def nucleation_growth_map(
    driving_force: Any,
    interfacial_energy: Any,
    response: Any,
    *,
    threshold: float | None = None,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Nucleation/growth response over driving-force and interface-energy space."""
    values = np.asarray(response, dtype=float)
    x = as_array(driving_force)
    y = as_array(interfacial_energy)
    if values.shape != (y.size, x.size):
        raise ValueError("response must have shape (len(interfacial_energy), len(driving_force))")
    xx, yy = np.meshgrid(x, y)
    cfg = coerce_config(config, xlabel="Driving force", ylabel="Interfacial energy")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.contourf(xx, yy, values, levels=30, cmap=cmap)
        boundary = axis.contour(xx, yy, values, levels=[threshold], colors="white", linewidths=1.1) if threshold is not None else None
        colorbar = fig.colorbar(mesh, ax=axis, label="Nucleation/growth response")
        return finalize(fig, axis, config=cfg, artists={"response": mesh, "boundary": boundary, "colorbar": colorbar}, data={"driving_force": xx, "interfacial_energy": yy, "response": values}, theme=theme)


@register_plot(category="phase-field", aliases=("stress-concentration-coupling", "chemo-mechanical-map"))
def stress_concentration_coupling_plot(
    concentration: Any,
    stress: Any,
    *,
    phase: Any | None = None,
    mode: str = "scatter",
    bins: int = 55,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Coupled concentration–stress relation from chemo-mechanical simulations."""
    c = as_array(concentration)
    s = as_array(stress)
    if c.size != s.size:
        raise ValueError("concentration and stress must have equal size")
    mask = np.isfinite(c) & np.isfinite(s)
    c, s = c[mask], s[mask]
    phase_arr = None if phase is None else as_array(phase)[mask]
    correlation = float(np.corrcoef(c, s)[0, 1]) if c.size > 1 else float("nan")
    cfg = coerce_config(config, xlabel="Concentration", ylabel="Stress")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        if mode == "hexbin":
            points = axis.hexbin(c, s, C=phase_arr, reduce_C_function=np.mean if phase_arr is not None else np.size, gridsize=bins, mincnt=1, cmap="viridis")
            colorbar = fig.colorbar(points, ax=axis, label="Mean phase" if phase_arr is not None else "Count")
        else:
            points = axis.scatter(c, s, c=phase_arr if phase_arr is not None else "0.45", cmap="viridis" if phase_arr is not None else None, s=10, alpha=0.55, rasterized=cfg.rasterized)
            colorbar = fig.colorbar(points, ax=axis, label="Phase") if phase_arr is not None else None
        fit = None
        if c.size > 1:
            coeff = np.polyfit(c, s, 1)
            grid = np.linspace(c.min(), c.max(), 150)
            fit = axis.plot(grid, np.polyval(coeff, grid), color="black", linestyle="--", label=f"r={correlation:.3f}")[0]
        return finalize(fig, axis, config=cfg, artists={"points": points, "fit": fit, "colorbar": colorbar}, data={"concentration": c, "stress": s, "phase": phase_arr}, metrics={"pearson_r": correlation}, theme=theme)


@register_plot(category="phase-field", aliases=("grain-orientation-map", "orientation-field"))
def grain_orientation_map(
    orientation: Any,
    *,
    phase: Any | None = None,
    radians: bool = True,
    boundary_width: float = 0.8,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Periodic grain-orientation field with phase/interface overlay."""
    angles = _field2d(orientation, "orientation")
    displayed = np.mod(angles, 2 * np.pi if radians else 360.0)
    cfg = coerce_config(config, xlabel="x", ylabel="y", equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(displayed, origin="lower", cmap="hsv", aspect="equal")
        boundary = None
        if phase is not None:
            eta = _field2d(phase, "phase")
            if eta.shape != angles.shape:
                raise ValueError("phase must match orientation")
            boundary = axis.contour(eta, levels=[0.5], colors="black", linewidths=boundary_width)
        colorbar = fig.colorbar(image, ax=axis, label="Orientation (rad)" if radians else "Orientation (deg)")
        return finalize(fig, axis, config=cfg, artists={"orientation": image, "boundary": boundary, "colorbar": colorbar}, data={"orientation": angles, "phase": phase}, theme=theme)


@register_plot(category="phase-field", aliases=("interface-curvature", "curvature-map"))
def interface_curvature_plot(
    phase: Any,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    interface_level: float = 0.5,
    interface_width: float = 0.12,
    cmap: str = "coolwarm",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Approximate diffuse-interface curvature derived from the phase gradient."""
    eta = _field2d(phase, "phase")
    dy, dx = spacing
    gy, gx = np.gradient(eta, dy, dx)
    magnitude = np.sqrt(gx**2 + gy**2)
    nx = gx / np.maximum(magnitude, 1e-12)
    ny = gy / np.maximum(magnitude, 1e-12)
    curvature = np.gradient(nx, dx, axis=1) + np.gradient(ny, dy, axis=0)
    mask = np.abs(eta - interface_level) <= interface_width
    if not np.any(mask):
        # Sharp/binary phase fields may contain no value inside the diffuse
        # interface interval. Fall back to high-gradient pixels.
        threshold = np.nanpercentile(magnitude, 90)
        mask = magnitude >= threshold
    shown = np.where(mask, curvature, np.nan)
    finite_curvature = np.abs(shown[np.isfinite(shown)])
    limit = float(np.nanpercentile(finite_curvature, 98)) if finite_curvature.size else 1.0
    mean_absolute_curvature = float(np.mean(finite_curvature)) if finite_curvature.size else float("nan")
    cfg = coerce_config(config, xlabel="x", ylabel="y", equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(shown, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        interface = axis.contour(eta, levels=[interface_level], colors="black", linewidths=0.7)
        colorbar = fig.colorbar(image, ax=axis, label="Interface curvature")
        return finalize(fig, axis, config=cfg, artists={"curvature": image, "interface": interface, "colorbar": colorbar}, data={"phase": eta, "curvature": curvature, "interface_mask": mask}, metrics={"mean_absolute_curvature": mean_absolute_curvature}, theme=theme)


@register_plot(category="phase-field", aliases=("order-parameter-distribution", "phase-histogram"))
def order_parameter_distribution_plot(
    order_parameter: Any,
    *,
    times: Any | None = None,
    labels: Sequence[str] | None = None,
    bins: int = 60,
    density: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Evolution of order-parameter distributions during phase separation."""
    values = np.asarray(order_parameter, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    else:
        values = values.reshape(values.shape[0], -1)
    if labels is None and times is not None:
        labels = [f"t={value:g}" for value in as_array(times)]
    names = make_labels(labels, values.shape[0], "Frame")
    cfg = coerce_config(config, xlabel="Order parameter", ylabel="Density" if density else "Count")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = []
        shared_range = (float(np.nanmin(values)), float(np.nanmax(values)))
        for row, name in zip(values, names):
            hist, edges = np.histogram(row[np.isfinite(row)], bins=bins, range=shared_range, density=density)
            centers = 0.5 * (edges[:-1] + edges[1:])
            artists.append(axis.plot(centers, hist, label=name)[0])
        return finalize(fig, axis, config=cfg, artists={"distributions": artists}, data={"order_parameter": values, "labels": names}, theme=theme)


@register_plot(category="phase-field", aliases=("phase-field-convergence", "solver-convergence"))
def phase_field_convergence_plot(
    iteration: Any,
    residuals: Mapping[str, Any] | Any,
    *,
    labels: Sequence[str] | None = None,
    tolerance: float | None = None,
    log_scale: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Nonlinear/linear solver residual histories for phase-field systems."""
    x = as_array(iteration)
    if isinstance(residuals, Mapping):
        names = list(residuals)
        matrix = np.vstack([as_array(v) for v in residuals.values()])
    else:
        matrix = np.asarray(residuals, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        names = make_labels(labels, matrix.shape[0], "Residual")
    if matrix.shape[1] != x.size:
        raise ValueError("residual histories must match iteration")
    cfg = coerce_config(config, xlabel="Iteration", ylabel="Residual", yscale="log" if log_scale else "linear")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = [axis.plot(x, np.abs(row), label=name)[0] for row, name in zip(matrix, names)]
        tolerance_line = axis.axhline(tolerance, color="black", linestyle="--", label=f"Tolerance={tolerance:g}") if tolerance is not None else None
        converged = bool(np.all(np.abs(matrix[:, -1]) <= tolerance)) if tolerance is not None else False
        return finalize(fig, axis, config=cfg, artists={"residuals": lines, "tolerance": tolerance_line}, data={"iteration": x, "residuals": matrix}, metrics={"converged": converged, "final_max_residual": float(np.nanmax(np.abs(matrix[:, -1])))}, theme=theme)


@register_plot(category="phase-field", aliases=("morphology-comparison", "phase-field-comparison-grid"))
def morphology_comparison_grid(
    fields: Mapping[str, Any] | Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    columns: int = 3,
    cmap: str = "magma",
    shared_scale: bool = True,
    interface_level: float | None = 0.5,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Journal-style comparison grid for phase-field morphologies."""
    if isinstance(fields, Mapping):
        names = list(fields)
        arrays = [_field2d(value, name) for name, value in fields.items()]
    else:
        arrays = [_field2d(value, "field") for value in fields]
        names = make_labels(labels, len(arrays), "Case")
    rows = int(np.ceil(len(arrays) / columns))
    cfg = coerce_config(config, figsize=(3.0 * columns, 2.7 * rows))
    vmin = min(float(np.nanmin(array)) for array in arrays) if shared_scale else None
    vmax = max(float(np.nanmax(array)) for array in arrays) if shared_scale else None
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=rows, ncols=columns, squeeze=False)
        images, contours = [], []
        for axis, array, name in zip(axes.flat, arrays, names):
            image = axis.imshow(array, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            images.append(image)
            if interface_level is not None:
                contours.append(axis.contour(array, levels=[interface_level], colors="white", linewidths=0.65))
            axis.set_title(name)
            axis.set_axis_off()
        for axis in list(axes.flat)[len(arrays):]:
            axis.set_visible(False)
        colorbar = fig.colorbar(images[-1], ax=list(axes.flat), shrink=0.78, label="Phase field") if images else None
        return finalize(fig, axes, config=cfg, artists={"fields": images, "interfaces": contours, "colorbar": colorbar}, data={"fields": arrays, "labels": names}, theme=theme)


@register_plot(category="phase-field", aliases=("front-position", "interface-position"))
def front_position_plot(
    time: Any,
    front_position: Any,
    *,
    labels: Sequence[str] | None = None,
    scaling_exponent: float | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Interface/front position and optional power-law scaling reference."""
    t = as_array(time)
    matrix = np.asarray(front_position, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != t.size:
        raise ValueError("front positions must match time")
    names = make_labels(labels, matrix.shape[0], "Front")
    cfg = coerce_config(config, xlabel="Time", ylabel="Front position")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = [axis.plot(t, row, label=name)[0] for row, name in zip(matrix, names)]
        reference = None
        if scaling_exponent is not None:
            positive = t > 0
            amplitude = np.nanmedian(matrix[0, positive] / np.maximum(t[positive] ** scaling_exponent, 1e-15))
            reference = axis.plot(t[positive], amplitude * t[positive] ** scaling_exponent, color="black", linestyle="--", label=fr"$t^{{{scaling_exponent:g}}}$")[0]
        return finalize(fig, axis, config=cfg, artists={"fronts": lines, "reference": reference}, data={"time": t, "front_position": matrix}, theme=theme)


@register_plot(category="phase-field", aliases=("phase-field-energy", "energy-dissipation"))
def phase_field_energy_plot(
    time: Any,
    *,
    chemical_energy: Any | None = None,
    gradient_energy: Any | None = None,
    elastic_energy: Any | None = None,
    electrostatic_energy: Any | None = None,
    total_energy: Any | None = None,
    stacked: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Phase-field energy components and total free-energy dissipation."""
    t = as_array(time)
    components = {
        "Chemical": chemical_energy,
        "Gradient": gradient_energy,
        "Elastic": elastic_energy,
        "Electrostatic": electrostatic_energy,
    }
    components = {name: as_array(value) for name, value in components.items() if value is not None}
    if not components and total_energy is None:
        raise ValueError("provide at least one energy series")
    for value in components.values():
        if value.size != t.size:
            raise ValueError("energy series must match time")
    total = as_array(total_energy) if total_energy is not None else sum(components.values())
    cfg = coerce_config(config, xlabel="Time", ylabel="Free energy")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        if stacked and components:
            component_artists = axis.stackplot(t, np.vstack(list(components.values())), labels=list(components), alpha=0.72)
        else:
            component_artists = [axis.plot(t, value, label=name)[0] for name, value in components.items()]
        total_line = axis.plot(t, total, color="black", linewidth=1.2, label="Total")[0]
        monotonic_fraction = float(np.mean(np.diff(total) <= 0)) if total.size > 1 else float("nan")
        return finalize(fig, axis, config=cfg, artists={"components": component_artists, "total": total_line}, data={"time": t, "components": components, "total": total}, metrics={"monotonic_decrease_fraction": monotonic_fraction, "energy_change": float(total[-1] - total[0])}, theme=theme)


@register_plot(category="phase-field", aliases=("dendrite-branching", "branch-statistics"))
def dendrite_branching_plot(
    time: Any,
    branch_count: Any,
    *,
    tip_count: Any | None = None,
    mean_branch_length: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "phase-field",
) -> PlotResult:
    """Dendrite branch and tip statistics through phase-field evolution."""
    t = as_array(time)
    branches = as_array(branch_count)
    if branches.size != t.size:
        raise ValueError("branch_count must match time")
    cfg = coerce_config(config, xlabel="Time", ylabel="Count")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        branch_line = axis.step(t, branches, where="mid", label="Branches")[0]
        tip_line = axis.step(t, as_array(tip_count), where="mid", label="Tips")[0] if tip_count is not None else None
        length_axis = None
        length_line = None
        if mean_branch_length is not None:
            length_axis = axis.twinx()
            length_line = length_axis.plot(t, as_array(mean_branch_length), color="#D1495B", label="Mean branch length")[0]
            length_axis.set_ylabel("Mean branch length")
        return finalize(fig, axis, config=cfg, artists={"branches": branch_line, "tips": tip_line, "length_axis": length_axis, "mean_length": length_line}, data={"time": t, "branch_count": branches, "tip_count": tip_count, "mean_branch_length": mean_branch_length}, metrics={"maximum_branch_count": int(np.nanmax(branches))}, theme=theme)
