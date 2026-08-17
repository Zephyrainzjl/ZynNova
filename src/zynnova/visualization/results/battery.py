from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_xy, make_labels


def _as_series_matrix(values: Any, x_size: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError("values must be one- or two-dimensional")
    if x_size is not None and matrix.shape[1] != x_size:
        if matrix.shape[0] == x_size:
            matrix = matrix.T
        else:
            raise ValueError(f"series length must match x ({x_size}); received {matrix.shape}")
    return matrix


def _mesh_coordinates(x: Any, y: Any, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim == y_arr.ndim == 1:
        if values.shape != (y_arr.size, x_arr.size):
            raise ValueError("values must have shape (len(y), len(x))")
        return np.meshgrid(x_arr, y_arr)
    if x_arr.shape != y_arr.shape or x_arr.shape != values.shape:
        raise ValueError("2-D coordinates must match values")
    return x_arr, y_arr


@register_plot(category="battery", aliases=("electrode-profile", "through-thickness-profile"))
def electrode_profile_plot(
    position: Any,
    values: Any,
    *,
    labels: Sequence[str] | None = None,
    regions: Sequence[tuple[float, float, str]] | None = None,
    normalize_position: bool = False,
    reference_lines: Mapping[str, float] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Through-thickness electrode profiles with region shading."""
    x = as_array(position)
    matrix = _as_series_matrix(values, x.size)
    if normalize_position:
        span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-15)
        x = (x - np.nanmin(x)) / span
    names = make_labels(labels, matrix.shape[0], "Field")
    cfg = coerce_config(config, xlabel="Normalized position" if normalize_position else "Position", ylabel="Value")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        bands = []
        if regions:
            for index, (left, right, name) in enumerate(regions):
                bands.append(axis.axvspan(left, right, color=resolved.colors[index % len(resolved.colors)], alpha=0.08, label=name))
        lines = [axis.plot(x, row, label=name)[0] for row, name in zip(matrix, names)]
        refs = []
        for name, value in (reference_lines or {}).items():
            refs.append(axis.axhline(value, linestyle="--", linewidth=0.8, color="0.35", label=name))
        return finalize(fig, axis, config=cfg, artists={"profiles": lines, "regions": bands, "references": refs}, data={"position": x, "values": matrix, "labels": names}, theme=theme)


@register_plot(category="battery", aliases=("electrolyte-concentration-profile", "li-concentration-profile"))
def concentration_profile_plot(
    position: Any,
    concentration: Any,
    *,
    times: Any | None = None,
    labels: Sequence[str] | None = None,
    normalize: bool = False,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Electrolyte or solid-phase concentration profiles across the cell."""
    x = as_array(position)
    matrix = _as_series_matrix(concentration, x.size)
    if normalize:
        matrix = matrix / np.maximum(np.nanmax(np.abs(matrix), axis=1, keepdims=True), 1e-15)
    if labels is None and times is not None:
        time_arr = as_array(times)
        labels = [f"t={value:g}" for value in time_arr]
    names = make_labels(labels, matrix.shape[0], "Profile")
    cfg = coerce_config(config, xlabel="Position", ylabel="Normalized concentration" if normalize else "Concentration")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        colors = __import__("matplotlib").colormaps[cmap](np.linspace(0.05, 0.95, matrix.shape[0]))
        lines = [axis.plot(x, row, color=color, label=name)[0] for row, color, name in zip(matrix, colors, names)]
        return finalize(fig, axis, config=cfg, artists={"profiles": lines}, data={"position": x, "concentration": matrix, "labels": names}, theme=theme)


@register_plot(category="battery", aliases=("reaction-current-profile", "local-current-density"))
def reaction_current_profile_plot(
    position: Any,
    current_density: Any,
    *,
    labels: Sequence[str] | None = None,
    zero_line: bool = True,
    cumulative: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Local interfacial reaction-current distribution through an electrode."""
    x = as_array(position)
    matrix = _as_series_matrix(current_density, x.size)
    displayed = np.cumsum(matrix, axis=1) if cumulative else matrix
    names = make_labels(labels, displayed.shape[0], "Condition")
    cfg = coerce_config(config, xlabel="Position", ylabel="Cumulative reaction" if cumulative else "Reaction current density")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = [axis.plot(x, row, label=name)[0] for row, name in zip(displayed, names)]
        zero = axis.axhline(0, color="black", linewidth=0.7) if zero_line else None
        utilization = np.trapezoid(np.abs(matrix), x=x, axis=1)
        return finalize(fig, axis, config=cfg, artists={"profiles": lines, "zero": zero}, data={"position": x, "current_density": matrix, "displayed": displayed}, metrics={"integrated_absolute_current": utilization.tolist()}, theme=theme)


@register_plot(category="battery", aliases=("overpotential-stack", "voltage-loss-breakdown"))
def overpotential_decomposition_plot(
    coordinate: Any,
    components: Mapping[str, Any] | Any,
    *,
    labels: Sequence[str] | None = None,
    stacked: bool = True,
    absolute: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Activation, ohmic, diffusion and other overpotential contributions."""
    x = as_array(coordinate)
    if isinstance(components, Mapping):
        names = list(components)
        matrix = np.vstack([as_array(value) for value in components.values()])
    else:
        matrix = _as_series_matrix(components, x.size)
        names = make_labels(labels, matrix.shape[0], "Component")
    if matrix.shape[1] != x.size:
        raise ValueError("component length must match coordinate")
    shown = np.abs(matrix) if absolute else matrix
    total = np.sum(shown, axis=0)
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Overpotential")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        if stacked:
            artists = axis.stackplot(x, shown, labels=names, alpha=0.8)
        else:
            artists = [axis.plot(x, row, label=name)[0] for row, name in zip(shown, names)]
        total_line = axis.plot(x, total, color="black", linewidth=1.1, label="Total")[0]
        return finalize(fig, axis, config=cfg, artists={"components": artists, "total": total_line}, data={"coordinate": x, "components": matrix, "total": total, "labels": names}, metrics={"maximum_total": float(np.nanmax(total))}, theme=theme)


@register_plot(category="battery", aliases=("electrode-utilization", "particle-utilization"))
def electrode_utilization_plot(
    coordinate: Any,
    utilization: Any,
    *,
    groups: Any | None = None,
    target: float | None = None,
    fill: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Spatial active-material utilization with optional groups and target."""
    x = as_array(coordinate)
    matrix = _as_series_matrix(utilization, x.size)
    point_groups = None
    if groups is None:
        names = make_labels(None, matrix.shape[0], "Group")
    else:
        raw_groups = np.asarray(groups).reshape(-1)
        if raw_groups.size == matrix.shape[0]:
            names = [str(item) for item in raw_groups]
        elif raw_groups.size == x.size and matrix.shape[0] == 1:
            point_groups = raw_groups
            names = [str(item) for item in dict.fromkeys(raw_groups.tolist())]
        else:
            raise ValueError("groups must label utilization series or points of a single series")
    cfg = coerce_config(config, xlabel="Position", ylabel="Utilization", ylim=(0, 1) if np.nanmax(matrix) <= 1.05 else None)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, fills = [], []
        if point_groups is not None:
            row = matrix[0]
            for name in names:
                selected = point_groups == name
                line = axis.plot(x[selected], row[selected], marker="o", markersize=2.5, label=name)[0]
                lines.append(line)
                if fill:
                    fills.append(axis.fill_between(x[selected], 0, row[selected], color=line.get_color(), alpha=0.12))
        else:
            for row, name in zip(matrix, names):
                line = axis.plot(x, row, label=name)[0]
                lines.append(line)
                if fill:
                    fills.append(axis.fill_between(x, 0, row, color=line.get_color(), alpha=0.12))
        target_line = axis.axhline(target, color="black", linestyle="--", label=f"Target={target:g}") if target is not None else None
        return finalize(fig, axis, config=cfg, artists={"profiles": lines, "fills": fills, "target": target_line}, data={"coordinate": x, "utilization": matrix, "groups": point_groups if point_groups is not None else names}, metrics={"mean_utilization": float(np.nanmean(matrix)), "minimum_utilization": float(np.nanmin(matrix))}, theme=theme)


@register_plot(category="battery", aliases=("capacity-fade-components", "aging-breakdown"))
def capacity_fade_components_plot(
    cycle: Any,
    components: Mapping[str, Any] | Any,
    *,
    labels: Sequence[str] | None = None,
    normalize: bool = False,
    cumulative: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Capacity-loss attribution to SEI, plating, LAM and transport losses."""
    x = as_array(cycle)
    if isinstance(components, Mapping):
        names = list(components)
        matrix = np.vstack([as_array(v) for v in components.values()])
    else:
        matrix = _as_series_matrix(components, x.size)
        names = make_labels(labels, matrix.shape[0], "Mode")
    if normalize:
        matrix = matrix / np.maximum(np.sum(matrix, axis=0, keepdims=True), 1e-15) * 100.0
    shown = np.cumsum(matrix, axis=1) if cumulative else matrix
    cfg = coerce_config(config, xlabel="Cycle", ylabel="Loss fraction (%)" if normalize else "Capacity loss")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        areas = axis.stackplot(x, shown, labels=names, alpha=0.82)
        total = np.sum(shown, axis=0)
        return finalize(fig, axis, config=cfg, artists={"components": areas}, data={"cycle": x, "components": matrix, "displayed": shown, "labels": names}, metrics={"final_total_loss": float(total[-1])}, theme=theme)


@register_plot(category="battery", aliases=("degradation-mode-map", "aging-heatmap"))
def degradation_mode_map(
    cycle: Any,
    position: Any,
    degradation: Any,
    *,
    cmap: str = "magma",
    colorbar_label: str = "Degradation",
    contours: int | Sequence[float] | None = 8,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Cycle–position degradation heatmap for full-cell aging simulations."""
    values = np.asarray(degradation, dtype=float)
    xx, yy = _mesh_coordinates(position, cycle, values)
    cfg = coerce_config(config, xlabel="Position", ylabel="Cycle")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.pcolormesh(xx, yy, values, cmap=cmap, shading="auto")
        contour = axis.contour(xx, yy, values, levels=contours, colors="white", linewidths=0.45, alpha=0.6) if contours else None
        colorbar = fig.colorbar(mesh, ax=axis, label=colorbar_label)
        return finalize(fig, axis, config=cfg, artists={"map": mesh, "contours": contour, "colorbar": colorbar}, data={"cycle": yy, "position": xx, "degradation": values}, metrics={"maximum_degradation": float(np.nanmax(values))}, theme=theme)


@register_plot(category="battery", aliases=("dqdv-map", "incremental-capacity-map"))
def differential_capacity_map(
    voltage: Any,
    cycle: Any,
    differential_capacity: Any,
    *,
    cmap: str = "coolwarm",
    symmetric: bool = False,
    peak_tracks: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Cycle-resolved dQ/dV heatmap with optional tracked peak positions."""
    values = np.asarray(differential_capacity, dtype=float)
    xx, yy = _mesh_coordinates(voltage, cycle, values)
    limit = float(np.nanmax(np.abs(values))) if symmetric else None
    cfg = coerce_config(config, xlabel="Voltage (V)", ylabel="Cycle")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.pcolormesh(xx, yy, values, cmap=cmap, shading="auto", vmin=-limit if limit else None, vmax=limit)
        tracks = []
        if peak_tracks is not None:
            peaks = np.asarray(peak_tracks, dtype=float)
            if peaks.ndim == 1:
                peaks = peaks[:, None]
            cycle_arr = as_array(cycle)
            for index in range(peaks.shape[1]):
                tracks.append(axis.plot(peaks[:, index], cycle_arr, color="white", linewidth=0.8, linestyle="--", label=f"Peak {index + 1}")[0])
        colorbar = fig.colorbar(mesh, ax=axis, label="dQ/dV")
        return finalize(fig, axis, config=cfg, artists={"map": mesh, "peak_tracks": tracks, "colorbar": colorbar}, data={"voltage": xx, "cycle": yy, "differential_capacity": values}, theme=theme)


@register_plot(category="battery", aliases=("drt", "impedance-drt"))
def impedance_drt_plot(
    relaxation_time: Any,
    gamma: Any,
    *,
    labels: Sequence[str] | None = None,
    fill: bool = True,
    annotate_peaks: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Distribution-of-relaxation-times spectra for impedance analysis."""
    tau = as_array(relaxation_time)
    matrix = _as_series_matrix(gamma, tau.size)
    names = make_labels(labels, matrix.shape[0], "Spectrum")
    cfg = coerce_config(config, xlabel="Relaxation time (s)", ylabel=r"$\gamma(\tau)$", xscale="log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, fills, annotations = [], [], []
        for row, name in zip(matrix, names):
            line = axis.plot(tau, row, label=name)[0]
            lines.append(line)
            if fill:
                fills.append(axis.fill_between(tau, 0, row, color=line.get_color(), alpha=0.12))
            if annotate_peaks > 0:
                try:
                    from scipy.signal import find_peaks

                    peaks, _ = find_peaks(row)
                    selected = peaks[np.argsort(row[peaks])[-annotate_peaks:]] if peaks.size else []
                    for peak in selected:
                        annotations.append(axis.annotate(f"{tau[peak]:.2g}s", (tau[peak], row[peak]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize="x-small"))
                except Exception:
                    pass
        return finalize(fig, axis, config=cfg, artists={"spectra": lines, "fills": fills, "annotations": annotations}, data={"relaxation_time": tau, "gamma": matrix}, theme=theme)


@register_plot(category="battery", aliases=("plating-risk-map", "lithium-plating-map"))
def lithium_plating_risk_map(
    soc: Any,
    temperature: Any,
    risk: Any,
    *,
    threshold: float | None = None,
    cmap: str = "inferno",
    safe_label: str = "Safe",
    risk_label: str = "Plating risk",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """SOC–temperature lithium-plating risk surface and safe boundary."""
    values = np.asarray(risk, dtype=float)
    xx, yy = _mesh_coordinates(soc, temperature, values)
    cfg = coerce_config(config, xlabel="SOC", ylabel="Temperature")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.contourf(xx, yy, values, levels=32, cmap=cmap)
        boundary = None
        labels_artist = []
        if threshold is not None:
            boundary = axis.contour(xx, yy, values, levels=[threshold], colors="white", linewidths=1.2)
            labels_artist = axis.clabel(boundary, fmt={threshold: f"{risk_label}: {threshold:g}"}, fontsize="x-small")
        colorbar = fig.colorbar(mesh, ax=axis, label="Risk score")
        return finalize(fig, axis, config=cfg, artists={"risk": mesh, "boundary": boundary, "labels": labels_artist, "colorbar": colorbar}, data={"soc": xx, "temperature": yy, "risk": values}, metrics={"risk_fraction": float(np.mean(values >= threshold)) if threshold is not None else float("nan")}, metadata={"safe_label": safe_label}, theme=theme)


@register_plot(category="battery", aliases=("fast-charge-window", "charging-operating-window"))
def fast_charge_window_plot(
    current: Any,
    temperature: Any,
    objective: Any,
    *,
    plating_risk: Any | None = None,
    thermal_risk: Any | None = None,
    risk_threshold: float = 1.0,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Fast-charge objective surface with electrochemical and thermal constraints."""
    values = np.asarray(objective, dtype=float)
    xx, yy = _mesh_coordinates(current, temperature, values)
    cfg = coerce_config(config, xlabel="Charge current / C-rate", ylabel="Temperature")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.contourf(xx, yy, values, levels=30, cmap=cmap)
        boundaries = []
        if plating_risk is not None:
            boundaries.append(axis.contour(xx, yy, np.asarray(plating_risk), levels=[risk_threshold], colors="#7B2CBF", linewidths=1.2))
        if thermal_risk is not None:
            boundaries.append(axis.contour(xx, yy, np.asarray(thermal_risk), levels=[risk_threshold], colors="#D00000", linewidths=1.2, linestyles="--"))
        feasible = np.ones(values.shape, dtype=bool)
        if plating_risk is not None:
            feasible &= np.asarray(plating_risk) < risk_threshold
        if thermal_risk is not None:
            feasible &= np.asarray(thermal_risk) < risk_threshold
        optimum = None
        if np.any(feasible):
            score = np.where(feasible, values, -np.inf)
            index = np.unravel_index(np.nanargmax(score), score.shape)
            optimum = axis.scatter([xx[index]], [yy[index]], marker="*", s=100, color="white", edgecolor="black", label="Best feasible")
        colorbar = fig.colorbar(mesh, ax=axis, label="Fast-charge objective")
        return finalize(fig, axis, config=cfg, artists={"objective": mesh, "boundaries": boundaries, "optimum": optimum, "colorbar": colorbar}, data={"current": xx, "temperature": yy, "objective": values, "feasible": feasible}, metrics={"feasible_fraction": float(np.mean(feasible))}, theme=theme)


@register_plot(category="battery", aliases=("particle-utilization-histogram", "particle-soc-distribution"))
def particle_utilization_histogram(
    utilization: Any,
    *,
    groups: Any | None = None,
    bins: int = 30,
    density: bool = True,
    target: float | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Distribution of particle SOC or utilization by electrode region."""
    values = as_array(utilization)
    group_arr = np.full(values.size, "All", dtype=object) if groups is None else np.asarray(groups).reshape(-1)
    if group_arr.size != values.size:
        raise ValueError("groups must match utilization")
    cfg = coerce_config(config, xlabel="Particle utilization", ylabel="Density" if density else "Count")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = []
        for group in dict.fromkeys(group_arr.tolist()):
            selected = values[group_arr == group]
            artists.append(axis.hist(selected, bins=bins, density=density, histtype="stepfilled", alpha=0.28, label=str(group)))
        target_line = axis.axvline(target, color="black", linestyle="--", label=f"Target={target:g}") if target is not None else None
        return finalize(fig, axis, config=cfg, artists={"histograms": artists, "target": target_line}, data={"utilization": values, "groups": group_arr}, metrics={"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}, theme=theme)


@register_plot(category="battery", aliases=("tortuosity-porosity", "bruggeman-map"))
def tortuosity_porosity_plot(
    porosity: Any,
    tortuosity: Any,
    *,
    labels: Any | None = None,
    color: Any | None = None,
    bruggeman_exponent: float | None = None,
    annotate: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Porosity–tortuosity design map with optional Bruggeman reference."""
    eps, tau, mask = finite_xy(porosity, tortuosity)
    label_arr = None if labels is None else np.asarray(labels)[mask]
    color_arr = None if color is None else np.asarray(color)[mask]
    cfg = coerce_config(config, xlabel="Porosity", ylabel="Tortuosity")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        points = axis.scatter(eps, tau, c=color_arr if color_arr is not None else "0.45", cmap="viridis" if color_arr is not None else None, s=34, alpha=0.78)
        colorbar = fig.colorbar(points, ax=axis, label="Property") if color_arr is not None and np.issubdtype(color_arr.dtype, np.number) else None
        reference = None
        if bruggeman_exponent is not None:
            grid = np.linspace(max(np.nanmin(eps), 1e-3), min(np.nanmax(eps), 0.999), 250)
            prediction = grid ** (1.0 - bruggeman_exponent)
            reference = axis.plot(grid, prediction, linestyle="--", color="black", label=f"Bruggeman n={bruggeman_exponent:g}")[0]
        if annotate and label_arr is not None:
            for x, y, label in zip(eps, tau, label_arr):
                axis.annotate(str(label), (x, y), xytext=(3, 3), textcoords="offset points", fontsize="x-small")
        return finalize(fig, axis, config=cfg, artists={"points": points, "reference": reference, "colorbar": colorbar}, data={"porosity": eps, "tortuosity": tau}, theme=theme)


@register_plot(category="battery", aliases=("transport-profile", "effective-transport"))
def through_plane_transport_plot(
    position: Any,
    *,
    ionic_conductivity: Any | None = None,
    electronic_conductivity: Any | None = None,
    diffusivity: Any | None = None,
    normalize: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Ionic, electronic and diffusive transport coefficients through thickness."""
    x = as_array(position)
    series = {"Ionic conductivity": ionic_conductivity, "Electronic conductivity": electronic_conductivity, "Diffusivity": diffusivity}
    series = {name: as_array(value) for name, value in series.items() if value is not None}
    if not series:
        raise ValueError("provide at least one transport field")
    for value in series.values():
        if value.size != x.size:
            raise ValueError("transport fields must match position")
    if normalize:
        series = {name: value / max(float(np.nanmax(np.abs(value))), 1e-15) for name, value in series.items()}
    cfg = coerce_config(config, xlabel="Position", ylabel="Normalized transport" if normalize else "Transport coefficient", yscale="linear" if normalize else "log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = [axis.plot(x, value, label=name)[0] for name, value in series.items()]
        return finalize(fig, axis, config=cfg, artists={"transport": lines}, data={"position": x, **series}, theme=theme)


@register_plot(category="battery", aliases=("cell-temperature-map", "collector-temperature"))
def current_collector_temperature_plot(
    x: Any,
    y: Any,
    temperature: Any,
    *,
    current_density: Any | None = None,
    cmap: str = "inferno",
    quiver_stride: int = 5,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "thermal",
) -> PlotResult:
    """Cell or current-collector temperature field with current-flow overlay."""
    values = np.asarray(temperature, dtype=float)
    xx, yy = _mesh_coordinates(x, y, values)
    cfg = coerce_config(config, xlabel="x", ylabel="y", equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.pcolormesh(xx, yy, values, cmap=cmap, shading="auto")
        vectors = None
        if current_density is not None:
            field = np.asarray(current_density, dtype=float)
            if field.shape == values.shape + (2,):
                u, v = field[..., 0], field[..., 1]
            elif field.shape == (2,) + values.shape:
                u, v = field[0], field[1]
            else:
                raise ValueError("current_density must have shape temperature+(2,) or (2,)+temperature")
            vectors = axis.quiver(xx[::quiver_stride, ::quiver_stride], yy[::quiver_stride, ::quiver_stride], u[::quiver_stride, ::quiver_stride], v[::quiver_stride, ::quiver_stride], color="white", alpha=0.75)
        colorbar = fig.colorbar(mesh, ax=axis, label="Temperature")
        return finalize(fig, axis, config=cfg, artists={"temperature": mesh, "current": vectors, "colorbar": colorbar}, data={"x": xx, "y": yy, "temperature": values}, metrics={"maximum_temperature": float(np.nanmax(values)), "temperature_spread": float(np.nanmax(values) - np.nanmin(values))}, theme=theme)


@register_plot(category="battery", aliases=("voltage-breakdown", "cell-polarization-breakdown"))
def cell_voltage_breakdown_plot(
    coordinate: Any,
    open_circuit_voltage: Any,
    *,
    ohmic_loss: Any | None = None,
    activation_loss: Any | None = None,
    concentration_loss: Any | None = None,
    measured_voltage: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Cell-voltage decomposition into OCV and polarization losses."""
    x = as_array(coordinate)
    ocv = as_array(open_circuit_voltage)
    if ocv.size != x.size:
        raise ValueError("open_circuit_voltage must match coordinate")
    losses = {
        "Ohmic": np.zeros_like(ocv) if ohmic_loss is None else as_array(ohmic_loss),
        "Activation": np.zeros_like(ocv) if activation_loss is None else as_array(activation_loss),
        "Concentration": np.zeros_like(ocv) if concentration_loss is None else as_array(concentration_loss),
    }
    for value in losses.values():
        if value.size != x.size:
            raise ValueError("loss arrays must match coordinate")
    simulated = ocv - sum(losses.values())
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Voltage (V)")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        ocv_line = axis.plot(x, ocv, color="black", linestyle="--", label="OCV")[0]
        lines = []
        cumulative = ocv.copy()
        fills = []
        for name, loss in losses.items():
            next_curve = cumulative - loss
            fills.append(axis.fill_between(x, cumulative, next_curve, alpha=0.22, label=name))
            cumulative = next_curve
        lines.append(axis.plot(x, simulated, label="Simulated terminal voltage")[0])
        measured_line = axis.plot(x, as_array(measured_voltage), marker="o", markersize=2.5, linestyle="none", label="Measured") [0] if measured_voltage is not None else None
        return finalize(fig, axis, config=cfg, artists={"ocv": ocv_line, "losses": fills, "simulated": lines, "measured": measured_line}, data={"coordinate": x, "ocv": ocv, "losses": losses, "simulated": simulated}, metrics={"maximum_polarization": float(np.nanmax(ocv - simulated))}, theme=theme)


@register_plot(category="battery", aliases=("cycling-waterfall", "voltage-waterfall"))
def cycle_waterfall_plot(
    capacity: Any,
    voltage: Any,
    *,
    cycles: Any,
    selected_cycles: Sequence[Any] | None = None,
    offset: float = 0.08,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Vertically offset voltage profiles that reveal cycle evolution."""
    q = as_array(capacity)
    v = as_array(voltage)
    cycle_arr = np.asarray(cycles).reshape(-1)
    if not (q.size == v.size == cycle_arr.size):
        raise ValueError("capacity, voltage and cycles must have equal length")
    unique = list(dict.fromkeys(cycle_arr.tolist()))
    chosen = unique if selected_cycles is None else list(selected_cycles)
    colors = __import__("matplotlib").colormaps[cmap](np.linspace(0.05, 0.95, max(len(chosen), 1)))
    cfg = coerce_config(config, xlabel="Capacity", ylabel="Voltage + offset")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = []
        for index, (cycle_value, color) in enumerate(zip(chosen, colors)):
            selected = cycle_arr == cycle_value
            lines.append(axis.plot(q[selected], v[selected] + index * offset, color=color, label=f"Cycle {cycle_value}")[0])
        return finalize(fig, axis, config=cfg, artists={"profiles": lines}, data={"capacity": q, "voltage": v, "cycles": cycle_arr, "selected_cycles": chosen}, theme=theme)


@register_plot(category="battery", aliases=("operando-map", "spatiotemporal-electrode-map"))
def operando_map_plot(
    coordinate: Any,
    time: Any,
    signal: Any,
    *,
    voltage: Any | None = None,
    cmap: str = "viridis",
    colorbar_label: str = "Signal",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Operando spatial signal map with optional synchronized voltage trace."""
    values = np.asarray(signal, dtype=float)
    xx, yy = _mesh_coordinates(coordinate, time, values)
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Time", figsize=(6.5, 4.8))
    with theme_context(theme):
        if ax is None and voltage is not None:
            fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=1, sharex=False, gridspec_kw={"height_ratios": [4, 1]})
            main, trace = axes
        else:
            fig, main, _ = create_axes(ax=ax, config=cfg, theme=theme)
            axes, trace = main, None
        mesh = main.pcolormesh(xx, yy, values, cmap=cmap, shading="auto")
        colorbar = fig.colorbar(mesh, ax=main, label=colorbar_label)
        voltage_line = None
        if trace is not None:
            t = as_array(time)
            voltage_line = trace.plot(t, as_array(voltage), color="black")[0]
            trace.set_xlabel("Time")
            trace.set_ylabel("Voltage")
        return finalize(fig, axes, config=cfg, artists={"map": mesh, "voltage": voltage_line, "colorbar": colorbar}, data={"coordinate": xx, "time": yy, "signal": values, "voltage": voltage}, theme=theme)


@register_plot(category="battery", aliases=("electrode-phase-fraction", "lithiation-phase-fraction"))
def electrode_phase_fraction_plot(
    coordinate: Any,
    fractions: Mapping[str, Any] | Any,
    *,
    labels: Sequence[str] | None = None,
    normalize: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Spatial or temporal phase fractions in phase-separating electrodes."""
    x = as_array(coordinate)
    if isinstance(fractions, Mapping):
        names = list(fractions)
        matrix = np.vstack([as_array(v) for v in fractions.values()])
    else:
        matrix = _as_series_matrix(fractions, x.size)
        names = make_labels(labels, matrix.shape[0], "Phase")
    if normalize:
        matrix = matrix / np.maximum(np.sum(matrix, axis=0, keepdims=True), 1e-15)
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Phase fraction", ylim=(0, 1) if normalize else None)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        areas = axis.stackplot(x, matrix, labels=names, alpha=0.82)
        return finalize(fig, axis, config=cfg, artists={"fractions": areas}, data={"coordinate": x, "fractions": matrix, "labels": names}, metrics={"closure_error": float(np.nanmax(np.abs(np.sum(matrix, axis=0) - 1))) if normalize else float("nan")}, theme=theme)


@register_plot(category="battery", aliases=("stack-pressure-performance", "pressure-cell-map"))
def stack_pressure_performance_plot(
    pressure: Any,
    performance: Any,
    *,
    labels: Sequence[str] | None = None,
    uncertainty: Any | None = None,
    optimum: str = "max",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Stack-pressure trade-off curve for solid-state battery simulations."""
    x = as_array(pressure)
    matrix = _as_series_matrix(performance, x.size)
    names = make_labels(labels, matrix.shape[0], "Metric")
    error = None if uncertainty is None else _as_series_matrix(uncertainty, x.size)
    cfg = coerce_config(config, xlabel="Stack pressure", ylabel="Performance")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, bands, optima = [], [], []
        for index, (row, name) in enumerate(zip(matrix, names)):
            line = axis.plot(x, row, marker="o", label=name)[0]
            lines.append(line)
            if error is not None:
                bands.append(axis.fill_between(x, row - error[index], row + error[index], color=line.get_color(), alpha=0.18))
            best = int(np.nanargmax(row) if optimum == "max" else np.nanargmin(row))
            optima.append(axis.scatter([x[best]], [row[best]], marker="*", s=70, color=line.get_color()))
        return finalize(fig, axis, config=cfg, artists={"curves": lines, "uncertainty": bands, "optima": optima}, data={"pressure": x, "performance": matrix, "uncertainty": error}, theme=theme)
