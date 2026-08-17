from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_xy, make_labels, regression_metrics


@register_plot(category="atomistic", aliases=("rdf", "pair-correlation"))
def radial_distribution_plot(
    radius: Any,
    rdf: Any,
    *,
    pairs: Sequence[str] | None = None,
    coordination: Any | None = None,
    reference: Any | None = None,
    peak_labels: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Radial distribution functions with optional coordination-number axis."""
    r = as_array(radius)
    matrix = np.asarray(rdf, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    names = make_labels(pairs, matrix.shape[0], "Pair")
    cfg = coerce_config(config, xlabel="r", ylabel="g(r)")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = [axis.plot(r, row, label=name)[0] for row, name in zip(matrix, names)]
        reference_lines = []
        if reference is not None:
            ref = np.asarray(reference, dtype=float)
            if ref.ndim == 1:
                ref = ref[None, :]
            for index, row in enumerate(ref):
                reference_lines.append(axis.plot(r, row, linestyle="--", color=lines[index % len(lines)].get_color(), alpha=0.75, label=f"{names[index % len(names)]} reference")[0])
        coord_axis = None
        if coordination is not None:
            coord = np.asarray(coordination, dtype=float)
            if coord.ndim == 1:
                coord = coord[None, :]
            coord_axis = axis.twinx()
            for index, row in enumerate(coord):
                coord_axis.plot(r, row, linestyle=":", color=lines[index % len(lines)].get_color(), alpha=0.7)
            coord_axis.set_ylabel("Coordination number")
        if peak_labels > 0:
            try:
                from scipy.signal import find_peaks

                peaks, _ = find_peaks(matrix[0], prominence=max(np.ptp(matrix[0]) * 0.04, 1e-12))
                selected = peaks[np.argsort(matrix[0][peaks])[-peak_labels:]]
                for peak in selected:
                    axis.annotate(f"{r[peak]:.2f}", (r[peak], matrix[0, peak]), xytext=(0, 5), textcoords="offset points", ha="center")
            except Exception:
                pass
        return finalize(fig, axis, config=cfg, artists={"rdf": lines, "reference": reference_lines, "coordination_axis": coord_axis}, data={"radius": r, "rdf": matrix, "pairs": names, "coordination": coordination})


@register_plot(category="atomistic", aliases=("msd", "mean-square-displacement"))
def mean_squared_displacement_plot(
    time: Any,
    msd: Any,
    *,
    species: Sequence[str] | None = None,
    dimensions: int = 3,
    fit_range: tuple[float, float] | None = None,
    loglog: bool = False,
    uncertainty: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """MSD curves with Einstein diffusion fits and diffusion coefficients."""
    t = as_array(time)
    matrix = np.asarray(msd, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    names = make_labels(species, matrix.shape[0], "Species")
    cfg = coerce_config(config, xlabel="Time", ylabel="MSD", xscale="log" if loglog else None, yscale="log" if loglog else None)
    diffusion = {}
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, fits, bands = [], [], []
        for index, (row, name) in enumerate(zip(matrix, names)):
            line = axis.plot(t, row, label=name)[0]
            lines.append(line)
            if uncertainty is not None:
                unc = np.asarray(uncertainty, dtype=float)
                err = unc[index] if unc.ndim > 1 else unc
                bands.append(axis.fill_between(t, row - err, row + err, color=line.get_color(), alpha=0.18))
            if fit_range is not None:
                selected = (t >= fit_range[0]) & (t <= fit_range[1])
                if selected.sum() >= 2:
                    slope, intercept = np.polyfit(t[selected], row[selected], 1)
                    diffusion[name] = float(slope / (2.0 * dimensions))
                    fits.append(axis.plot(t[selected], slope * t[selected] + intercept, linestyle="--", color=line.get_color(), label=f"{name}: D={diffusion[name]:.3g}")[0])
        return finalize(fig, axis, config=cfg, artists={"curves": lines, "fits": fits, "bands": bands}, data={"time": t, "msd": matrix, "species": names}, metrics={f"diffusion_{key}": value for key, value in diffusion.items()})


@register_plot(category="atomistic", aliases=("vacf",))
def velocity_autocorrelation_plot(
    time: Any,
    vacf: Any,
    *,
    labels: Sequence[str] | None = None,
    normalize: bool = True,
    spectral_density: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Velocity autocorrelation with optional vibrational density spectrum."""
    t = as_array(time)
    matrix = np.asarray(vacf, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if normalize:
        matrix = matrix / np.maximum(np.abs(matrix[:, [0]]), 1e-15)
    names = make_labels(labels, matrix.shape[0], "Series")
    cfg = coerce_config(config, xlabel="Time", ylabel="Normalized VACF" if normalize else "VACF", figsize=(7.0, 3.5) if spectral_density else None)
    with theme_context(theme):
        if spectral_density:
            fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=1, ncols=2)
            time_axis, freq_axis = axes
        else:
            fig, time_axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
            axes, freq_axis = time_axis, None
        lines = [time_axis.plot(t, row, label=name)[0] for row, name in zip(matrix, names)]
        spectra = []
        frequencies = None
        if freq_axis is not None:
            dt = float(np.mean(np.diff(t)))
            frequencies = np.fft.rfftfreq(t.size, d=dt)
            for row, name, line in zip(matrix, names, lines):
                spectrum = np.real(np.fft.rfft(row))
                spectra.append(freq_axis.plot(frequencies, spectrum, label=name, color=line.get_color())[0])
            freq_axis.set_xlabel("Frequency")
            freq_axis.set_ylabel("Spectral density")
        return finalize(fig, axes, config=cfg, artists={"vacf": lines, "spectra": spectra}, data={"time": t, "vacf": matrix, "frequency": frequencies})


@register_plot(category="atomistic", aliases=("fes", "free-energy-landscape"))
def free_energy_surface(
    coordinate_x: Any,
    coordinate_y: Any,
    *,
    weights: Any | None = None,
    temperature: float = 300.0,
    bins: int | tuple[int, int] = 80,
    k_b: float = 8.617333262e-5,
    max_free_energy: float | None = None,
    levels: int = 25,
    cmap: str = "viridis_r",
    scatter: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """2-D potential of mean force from collective-variable samples."""
    x, y, mask = finite_xy(coordinate_x, coordinate_y)
    sample_weights = None if weights is None else as_array(weights)[mask]
    histogram, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=sample_weights, density=True)
    probability = histogram / max(np.sum(histogram), 1e-300)
    free_energy = -k_b * temperature * np.log(np.clip(probability, 1e-300, None))
    free_energy -= np.nanmin(free_energy[np.isfinite(free_energy)])
    if max_free_energy is not None:
        free_energy[free_energy > max_free_energy] = np.nan
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    cfg = coerce_config(config, xlabel="Collective variable 1", ylabel="Collective variable 2")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        contour = axis.contourf(x_centers, y_centers, free_energy.T, levels=levels, cmap=cmap)
        line_contour = axis.contour(x_centers, y_centers, free_energy.T, levels=max(5, levels // 5), colors="black", linewidths=0.35, alpha=0.5)
        points = axis.scatter(x, y, s=1, color="white", alpha=0.04, rasterized=True) if scatter else None
        colorbar = fig.colorbar(contour, ax=axis, label="Free energy")
        return finalize(fig, axis, config=cfg, artists={"surface": contour, "contours": line_contour, "points": points, "colorbar": colorbar}, data={"x": x_centers, "y": y_centers, "free_energy": free_energy, "probability": probability})


@register_plot(category="atomistic", aliases=("pmf",))
def potential_of_mean_force(
    coordinate: Any,
    *,
    weights: Any | None = None,
    temperature: float = 300.0,
    bins: int = 100,
    k_b: float = 8.617333262e-5,
    smooth_sigma: float | None = 1.0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """One-dimensional PMF from sampled coordinates."""
    values = as_array(coordinate)
    hist, edges = np.histogram(values, bins=bins, weights=None if weights is None else as_array(weights), density=True)
    if smooth_sigma is not None:
        try:
            from scipy.ndimage import gaussian_filter1d

            hist = gaussian_filter1d(hist, smooth_sigma)
        except Exception:
            pass
    centers = 0.5 * (edges[:-1] + edges[1:])
    free_energy = -k_b * temperature * np.log(np.clip(hist, 1e-300, None))
    free_energy -= np.nanmin(free_energy[np.isfinite(free_energy)])
    cfg = coerce_config(config, xlabel="Reaction coordinate", ylabel="Free energy")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line = axis.plot(centers, free_energy)[0]
        fill = axis.fill_between(centers, 0, free_energy, color=line.get_color(), alpha=0.12)
        return finalize(fig, axis, config=cfg, artists={"curve": line, "fill": fill}, data={"coordinate": centers, "free_energy": free_energy, "density": hist})


@register_plot(category="atomistic", aliases=("energy-drift", "md-conservation"))
def energy_conservation_plot(
    time: Any,
    total_energy: Any,
    *,
    potential_energy: Any | None = None,
    kinetic_energy: Any | None = None,
    relative: bool = True,
    rolling_window: int | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """MD energy conservation and drift diagnostics with optional smoothing."""
    t, total, _ = finite_xy(time, total_energy)
    displayed = (total - total[0]) / max(abs(total[0]), 1e-15) if relative else total
    slope = float(np.polyfit(t, displayed, 1)[0]) if t.size > 1 else float("nan")
    smooth = None
    if rolling_window is not None and int(rolling_window) > 1:
        window = min(int(rolling_window), displayed.size)
        kernel = np.ones(window, dtype=float) / window
        smooth = np.convolve(displayed, kernel, mode="same")
    cfg = coerce_config(config, xlabel="Time", ylabel="Relative energy drift" if relative else "Energy")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        total_line = axis.plot(t, displayed, label="Total", alpha=0.45 if smooth is not None else 1.0)[0]
        smooth_line = axis.plot(t, smooth, label=f"Rolling mean ({rolling_window})", linewidth=1.5)[0] if smooth is not None else None
        potential_line = axis.plot(t, as_array(potential_energy)[:t.size], label="Potential", alpha=0.7)[0] if potential_energy is not None else None
        kinetic_line = axis.plot(t, as_array(kinetic_energy)[:t.size], label="Kinetic", alpha=0.7)[0] if kinetic_energy is not None else None
        trend = axis.plot(t, slope * t + (np.mean(displayed) - slope * np.mean(t)), linestyle="--", color="black", label=f"Drift={slope:.3g}/time")[0]
        return finalize(fig, axis, config=cfg, artists={"total": total_line, "rolling": smooth_line, "potential": potential_line, "kinetic": kinetic_line, "trend": trend}, data={"time": t, "total_energy": total, "displayed": displayed, "rolling_mean": smooth}, metrics={"drift_rate": slope, "peak_to_peak": float(np.ptp(displayed))}, theme=theme)


@register_plot(category="atomistic", aliases=("thermodynamic-trace",))
def thermodynamic_trace(
    time: Any,
    *,
    temperature: Any | None = None,
    pressure: Any | None = None,
    density: Any | None = None,
    volume: Any | None = None,
    targets: Mapping[str, float] | None = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Stacked MD traces for temperature, pressure, density and volume."""
    series = {name: value for name, value in {"Temperature": temperature, "Pressure": pressure, "Density": density, "Volume": volume}.items() if value is not None}
    if not series:
        raise ValueError("provide at least one thermodynamic series")
    t = as_array(time)
    cfg = coerce_config(config, figsize=(7.0, 1.8 * len(series) + 0.5))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=len(series), ncols=1, sharex=True)
        axes_arr = np.atleast_1d(axes)
        artists = {}
        for axis, (name, values) in zip(axes_arr, series.items()):
            line = axis.plot(t, as_array(values)[:t.size], label=name)[0]
            axis.set_ylabel(name)
            if targets and name in targets:
                axis.axhline(targets[name], color="black", linestyle="--", linewidth=0.8)
            artists[name] = line
        axes_arr[-1].set_xlabel("Time")
        return finalize(fig, axes, config=cfg, artists=artists, data={"time": t, **series})


@register_plot(category="atomistic", aliases=("angular-distribution",))
def angular_distribution_plot(
    angle: Any,
    *,
    groups: Any | None = None,
    bins: int = 90,
    degrees: bool = True,
    polar: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Bond-angle or orientation distribution in Cartesian or polar form."""
    values = as_array(angle)
    radians = np.deg2rad(values) if degrees else values
    range_values = (0.0, 180.0) if degrees else (0.0, np.pi)
    group_arr = None if groups is None else np.asarray(groups).reshape(-1)
    if group_arr is not None and group_arr.size != values.size:
        raise ValueError("groups must match angle")
    cfg = coerce_config(config, xlabel="Angle (deg)" if degrees else "Angle (rad)", ylabel="Probability density")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme, projection="polar" if polar else None)
        displayed = radians if polar else values
        histograms = []
        if group_arr is None:
            histograms.append(axis.hist(displayed, bins=bins, density=True, range=(0, np.pi) if polar else range_values, alpha=0.65))
        else:
            for group in dict.fromkeys(group_arr.tolist()):
                selected = group_arr == group
                histograms.append(axis.hist(displayed[selected], bins=bins, density=True, range=(0, np.pi) if polar else range_values, histtype="step", linewidth=1.4, label=str(group)))
        return finalize(fig, axis, config=cfg, artists={"histogram": histograms}, data={"angle": values, "groups": group_arr}, theme=theme)


@register_plot(category="atomistic", aliases=("contact-map", "distance-map"))
def contact_map_plot(
    matrix_or_positions: Any,
    *,
    cutoff: float | None = None,
    labels: Sequence[str] | None = None,
    cmap: str = "magma_r",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Residue/atom contact or pair-distance map."""
    array = np.asarray(matrix_or_positions, dtype=float)
    if array.ndim == 2 and array.shape[0] != array.shape[1] and array.shape[1] == 3:
        differences = array[:, None, :] - array[None, :, :]
        matrix = np.linalg.norm(differences, axis=-1)
    else:
        matrix = array
    displayed = (matrix <= cutoff).astype(float) if cutoff is not None else matrix
    cfg = coerce_config(config, xlabel="Index", ylabel="Index")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(displayed, cmap="binary" if cutoff is not None else cmap, origin="lower", aspect="equal")
        if labels is not None and len(labels) <= 50:
            axis.set_xticks(np.arange(len(labels)), labels, rotation=90)
            axis.set_yticks(np.arange(len(labels)), labels)
        colorbar = fig.colorbar(image, ax=axis, label="Contact" if cutoff is not None else "Distance")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"matrix": matrix, "displayed": displayed})


@register_plot(category="atomistic", aliases=("state-population",))
def state_population_plot(
    time: Any,
    states: Any,
    *,
    state_names: Mapping[Any, str] | None = None,
    window: int = 1,
    stacked: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Time-resolved conformational or phase-state populations."""
    t = as_array(time)
    state_arr = np.asarray(states)
    unique = list(dict.fromkeys(state_arr.reshape(-1).tolist()))
    populations = np.asarray([(state_arr == state).mean(axis=1) if state_arr.ndim > 1 else (state_arr == state).astype(float) for state in unique]).T
    if window > 1:
        kernel = np.ones(window) / window
        populations = np.asarray([np.convolve(populations[:, index], kernel, mode="same") for index in range(populations.shape[1])]).T
    labels = [(state_names or {}).get(state, str(state)) for state in unique]
    cfg = coerce_config(config, xlabel="Time", ylabel="Population", ylim=(0, 1))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = axis.stackplot(t, populations.T, labels=labels, alpha=0.75) if stacked else [axis.plot(t, populations[:, index], label=label)[0] for index, label in enumerate(labels)]
        return finalize(fig, axis, config=cfg, artists={"populations": artists}, data={"time": t, "populations": populations, "states": unique})


@register_plot(category="atomistic", aliases=("species-force-error",))
def force_error_by_species(
    reference_forces: Any,
    predicted_forces: Any,
    species: Any,
    *,
    component: str = "norm",
    kind: str = "violin",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Force-error distributions separated by chemical species."""
    reference = np.asarray(reference_forces, dtype=float)
    prediction = np.asarray(predicted_forces, dtype=float)
    if reference.shape != prediction.shape:
        raise ValueError("reference and predicted forces must have equal shape")
    error_vector = prediction - reference
    error = np.linalg.norm(error_vector, axis=-1) if component == "norm" else error_vector[..., int(component)]
    species_arr = np.asarray(species).reshape(-1)
    error = error.reshape(-1)
    if species_arr.size != error.size:
        if error.size % species_arr.size == 0:
            species_arr = np.tile(species_arr, error.size // species_arr.size)
        else:
            raise ValueError("species must match force-vector count")
    from .distributions import raincloud_plot, violin_box_scatter

    if kind == "raincloud":
        return raincloud_plot(error, groups=species_arr, ax=ax, config=coerce_config(config, ylabel="Force error"), theme=theme)
    return violin_box_scatter(error, groups=species_arr, ax=ax, config=coerce_config(config, ylabel="Force error"), theme=theme)


@register_plot(category="atomistic", aliases=("mlip-dashboard", "energy-force-stress-parity"))
def mlip_parity_dashboard(
    *,
    energy_reference: Any,
    energy_prediction: Any,
    force_reference: Any,
    force_prediction: Any,
    stress_reference: Any | None = None,
    stress_prediction: Any | None = None,
    species: Any | None = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Multi-panel machine-learning potential validation dashboard."""
    panels = 3 if stress_reference is not None and stress_prediction is not None else 2
    cfg = coerce_config(config, figsize=(4.0 * panels, 3.7))
    from .model_evaluation import parity_plot
    from ._utils import regression_metrics

    ref_force = np.asarray(force_reference, dtype=float).reshape(-1)
    pred_force = np.asarray(force_prediction, dtype=float).reshape(-1)
    force_groups = None
    if species is not None:
        species_arr = np.asarray(species).reshape(-1)
        if species_arr.size == ref_force.size:
            force_groups = species_arr
        elif species_arr.size * 3 == ref_force.size:
            force_groups = np.repeat(species_arr, 3)
        else:
            raise ValueError("species must match force atoms or force components")
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=1, ncols=panels)
        axes_arr = np.atleast_1d(axes)
        energy_result = parity_plot(energy_reference, energy_prediction, mode="density", ax=axes_arr[0], config=PlotConfig(title="Energy", xlabel="Reference", ylabel="Prediction", equal_aspect=True, legend=False), theme=theme)
        force_result = parity_plot(ref_force, pred_force, groups=force_groups, mode="hexbin", ax=axes_arr[1], config=PlotConfig(title="Force components", xlabel="Reference", ylabel="Prediction", equal_aspect=True, legend=force_groups is not None), theme=theme)
        stress_result = None
        if panels == 3:
            stress_result = parity_plot(np.asarray(stress_reference).reshape(-1), np.asarray(stress_prediction).reshape(-1), mode="density", ax=axes_arr[2], config=PlotConfig(title="Stress", xlabel="Reference", ylabel="Prediction", equal_aspect=True, legend=False), theme=theme)
        metrics = {f"energy_{key}": value for key, value in energy_result.metrics.items()}
        metrics.update({f"force_{key}": value for key, value in force_result.metrics.items()})
        if force_groups is not None:
            for group in dict.fromkeys(force_groups.tolist()):
                selected = force_groups == group
                for key, value in regression_metrics(ref_force[selected], pred_force[selected]).items():
                    metrics[f"force_{group}_{key}"] = value
        if stress_result is not None:
            metrics.update({f"stress_{key}": value for key, value in stress_result.metrics.items()})
        return finalize(fig, axes, config=cfg, artists={"energy": energy_result.artists, "force": force_result.artists, "stress": None if stress_result is None else stress_result.artists}, data={"species": force_groups}, metrics=metrics, theme=theme)
