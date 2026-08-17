from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, finite_xy, make_labels


@register_plot(category="multiscale", aliases=("scale-bridge", "multiscale-bridge"))
def scale_bridge_plot(
    scales: Sequence[str],
    *,
    length_ranges: Sequence[tuple[float, float]] | None = None,
    time_ranges: Sequence[tuple[float, float]] | None = None,
    methods: Sequence[str] | None = None,
    couplings: Sequence[tuple[int, int, str]] | None = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Visual map from electronic/atomistic scales to electrode, cell and pack."""
    import matplotlib.patches as patches

    n = len(scales)
    if n == 0:
        raise ValueError("scales cannot be empty")
    if length_ranges is None:
        length_ranges = [(10.0 ** (-10 + 2 * i), 10.0 ** (-8 + 2 * i)) for i in range(n)]
    if len(length_ranges) != n:
        raise ValueError("length_ranges must match scales")
    methods = list(methods or [""] * n)
    if len(methods) != n:
        raise ValueError("methods must match scales")
    cfg = coerce_config(config, figsize=(8.0, 4.8), xlabel="Length scale (m)", ylabel="Model level", xscale="log", ylim=(-0.7, n - 0.3), despine=False)
    with theme_context(theme):
        fig, axis, resolved = create_axes(config=cfg, theme=theme)
        rectangles, arrows, texts = [], [], []
        for index, (name, bounds, method) in enumerate(zip(scales, length_ranges, methods)):
            left, right = bounds
            rect = patches.Rectangle((left, index - 0.28), right - left, 0.56, facecolor=resolved.colors[index % len(resolved.colors)], alpha=0.65, edgecolor="none")
            axis.add_patch(rect)
            rectangles.append(rect)
            texts.append(axis.text(np.sqrt(left * right), index, f"{name}\n{method}" if method else name, ha="center", va="center", fontsize="small"))
        for source, target, label in couplings or [(i, i + 1, "upscale") for i in range(n - 1)]:
            x0 = length_ranges[source][1]
            x1 = length_ranges[target][0]
            arrow = patches.FancyArrowPatch((x0, source), (x1, target), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color="0.25", connectionstyle="arc3,rad=0.08")
            axis.add_patch(arrow)
            arrows.append(arrow)
            if label:
                texts.append(axis.text(np.sqrt(x0 * x1), (source + target) / 2 + 0.18, label, ha="center", va="bottom", fontsize="x-small"))
        axis.set_yticks(range(n), scales)
        if time_ranges is not None:
            axis.text(1.01, 0.5, "Time ranges: " + "; ".join(f"{a:g}–{b:g}s" for a, b in time_ranges), transform=axis.transAxes, rotation=90, va="center", fontsize="x-small")
        return finalize(fig, axis, config=cfg, artists={"scale_bands": rectangles, "couplings": arrows, "labels": texts}, data={"scales": list(scales), "length_ranges": list(length_ranges), "time_ranges": time_ranges, "methods": methods}, theme=theme)


@register_plot(category="multiscale", aliases=("model-hierarchy", "fidelity-hierarchy"))
def model_hierarchy_plot(
    models: Sequence[str],
    accuracy: Any,
    cost: Any,
    *,
    coverage: Any | None = None,
    labels: Sequence[str] | None = None,
    connect: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Accuracy–cost hierarchy from DFT/MD to continuum and system models."""
    acc = as_array(accuracy)
    computational_cost = as_array(cost)
    if len(models) != acc.size or acc.size != computational_cost.size:
        raise ValueError("models, accuracy and cost must have equal length")
    sizes = 80.0 if coverage is None else 30.0 + 120.0 * np.asarray(coverage, dtype=float) / max(float(np.nanmax(coverage)), 1e-15)
    cfg = coerce_config(config, xlabel="Computational cost", ylabel="Accuracy / fidelity", xscale="log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        points = axis.scatter(computational_cost, acc, s=sizes, c=np.arange(acc.size), cmap="viridis", alpha=0.8)
        line = axis.plot(computational_cost, acc, color="0.45", linestyle="--", linewidth=0.8)[0] if connect else None
        annotations = []
        names = labels or models
        for x, y, name in zip(computational_cost, acc, names):
            annotations.append(axis.annotate(str(name), (x, y), xytext=(4, 4), textcoords="offset points", fontsize="small"))
        return finalize(fig, axis, config=cfg, artists={"models": points, "connection": line, "labels": annotations}, data={"models": list(models), "accuracy": acc, "cost": computational_cost, "coverage": coverage}, theme=theme)


@register_plot(category="multiscale", aliases=("homogenization-comparison", "resolved-vs-homogenized"))
def homogenization_comparison_plot(
    coordinate: Any,
    resolved: Any,
    homogenized: Any,
    *,
    reference: Any | None = None,
    labels: Sequence[str] = ("Microstructure-resolved", "Homogenized"),
    residual_panel: bool = True,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Compare microstructure-resolved and homogenized model predictions."""
    x = as_array(coordinate)
    a = as_array(resolved)
    b = as_array(homogenized)
    if not (x.size == a.size == b.size):
        raise ValueError("all series must have equal length")
    residual = b - a
    cfg = coerce_config(config, figsize=(6.2, 5.0 if residual_panel else 3.8), xlabel="Coordinate", ylabel="Response")
    with theme_context(theme):
        if residual_panel:
            fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            main, residual_axis = axes
        else:
            fig, main, _ = create_axes(config=cfg, theme=theme)
            axes, residual_axis = main, None
        resolved_line = main.plot(x, a, label=labels[0])[0]
        homogenized_line = main.plot(x, b, label=labels[1], linestyle="--")[0]
        reference_line = main.plot(x, as_array(reference), marker="o", markersize=3, linestyle="none", color="black", label="Reference")[0] if reference is not None else None
        residual_line = None
        if residual_axis is not None:
            residual_line = residual_axis.plot(x, residual, color=homogenized_line.get_color())[0]
            residual_axis.axhline(0, color="black", linewidth=0.7)
            residual_axis.set_ylabel("Difference")
            residual_axis.set_xlabel("Coordinate")
        rmse = float(np.sqrt(np.mean(residual**2)))
        return finalize(fig, axes, config=cfg, artists={"resolved": resolved_line, "homogenized": homogenized_line, "reference": reference_line, "residual": residual_line}, data={"coordinate": x, "resolved": a, "homogenized": b, "residual": residual}, metrics={"rmse_between_models": rmse, "maximum_absolute_difference": float(np.max(np.abs(residual)))}, theme=theme)


@register_plot(category="multiscale", aliases=("rve-convergence", "representative-volume-convergence"))
def representative_volume_convergence_plot(
    volume_size: Any,
    property_value: Any,
    *,
    uncertainty: Any | None = None,
    reference: float | None = None,
    tolerance: float | None = None,
    labels: Sequence[str] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Representative-volume convergence of effective properties."""
    x = as_array(volume_size)
    matrix = np.asarray(property_value, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != x.size:
        raise ValueError("property curves must match volume_size")
    names = make_labels(labels, matrix.shape[0], "Property")
    errors = None if uncertainty is None else np.asarray(uncertainty, dtype=float)
    if errors is not None and errors.ndim == 1:
        errors = errors[None, :]
    cfg = coerce_config(config, xlabel="RVE size", ylabel="Effective property", xscale="log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, bands = [], []
        for index, (row, name) in enumerate(zip(matrix, names)):
            line = axis.plot(x, row, marker="o", label=name)[0]
            lines.append(line)
            if errors is not None:
                bands.append(axis.fill_between(x, row - errors[index], row + errors[index], color=line.get_color(), alpha=0.18))
        ref_line = axis.axhline(reference, color="black", linestyle="--", label="Reference") if reference is not None else None
        tolerance_band = None
        if reference is not None and tolerance is not None:
            tolerance_band = axis.axhspan(reference - tolerance, reference + tolerance, color="0.5", alpha=0.12, label="Tolerance")
        convergence_size = float("nan")
        if reference is not None and tolerance is not None:
            valid = np.all(np.abs(matrix - reference) <= tolerance, axis=0)
            if np.any(valid):
                convergence_size = float(x[np.flatnonzero(valid)[0]])
        return finalize(fig, axis, config=cfg, artists={"curves": lines, "uncertainty": bands, "reference": ref_line, "tolerance": tolerance_band}, data={"volume_size": x, "property": matrix, "uncertainty": errors}, metrics={"convergence_size": convergence_size}, theme=theme)


@register_plot(category="multiscale", aliases=("mesh-convergence", "discretization-convergence"))
def mesh_convergence_plot(
    degrees_of_freedom: Any,
    quantity: Any,
    *,
    reference: float | None = None,
    runtime: Any | None = None,
    labels: Sequence[str] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Finite-element/finite-volume convergence against mesh size or DOFs."""
    dof = as_array(degrees_of_freedom)
    matrix = np.asarray(quantity, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != dof.size:
        raise ValueError("quantity curves must match degrees_of_freedom")
    names = make_labels(labels, matrix.shape[0], "Quantity")
    cfg = coerce_config(config, xlabel="Degrees of freedom", ylabel="Quantity", xscale="log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = [axis.plot(dof, row, marker="o", label=name)[0] for row, name in zip(matrix, names)]
        reference_line = axis.axhline(reference, color="black", linestyle="--", label="Reference") if reference is not None else None
        runtime_axis = None
        runtime_line = None
        if runtime is not None:
            runtime_axis = axis.twinx()
            runtime_line = runtime_axis.plot(dof, as_array(runtime), color="#D1495B", marker="s", label="Runtime")[0]
            runtime_axis.set_ylabel("Runtime")
            runtime_axis.set_yscale("log")
        relative_error = None if reference is None else np.abs(matrix - reference) / max(abs(reference), 1e-15)
        return finalize(fig, axis, config=cfg, artists={"curves": lines, "reference": reference_line, "runtime_axis": runtime_axis, "runtime": runtime_line}, data={"degrees_of_freedom": dof, "quantity": matrix, "runtime": runtime, "relative_error": relative_error}, metrics={"finest_relative_error": float(np.nanmax(relative_error[:, -1])) if relative_error is not None else float("nan")}, theme=theme)


@register_plot(category="multiscale", aliases=("sensitivity-tornado", "parameter-tornado"))
def parameter_sensitivity_tornado(
    parameters: Sequence[str],
    low_effect: Any,
    high_effect: Any,
    *,
    baseline: float = 0.0,
    sort: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Tornado chart for one-at-a-time multiscale parameter sensitivity."""
    low = as_array(low_effect)
    high = as_array(high_effect)
    if len(parameters) != low.size or low.size != high.size:
        raise ValueError("parameters and effects must have equal length")
    names = np.asarray(parameters, dtype=object)
    span = np.abs(high - low)
    order = np.argsort(span) if sort else np.arange(span.size)
    names, low, high = names[order], low[order], high[order]
    y = np.arange(names.size)
    cfg = coerce_config(config, xlabel="Effect on output", ylabel="")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        low_bars = axis.barh(y, low - baseline, left=baseline, color=resolved.colors[0], alpha=0.72, label="Low")
        high_bars = axis.barh(y, high - baseline, left=baseline, color=resolved.colors[1], alpha=0.72, label="High")
        axis.axvline(baseline, color="black", linewidth=0.8)
        axis.set_yticks(y, names)
        return finalize(fig, axis, config=cfg, artists={"low": low_bars, "high": high_bars}, data={"parameters": names, "low_effect": low, "high_effect": high, "baseline": baseline}, metrics={"most_sensitive": str(names[-1]) if names.size else ""}, theme=theme)


@register_plot(category="multiscale", aliases=("sobol-indices", "global-sensitivity"))
def sobol_indices_plot(
    parameters: Sequence[str],
    first_order: Any,
    total_order: Any,
    *,
    confidence: Any | None = None,
    sort: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """First-order and total Sobol sensitivity indices."""
    first = as_array(first_order)
    total = as_array(total_order)
    if len(parameters) != first.size or first.size != total.size:
        raise ValueError("parameters and Sobol arrays must have equal length")
    names = np.asarray(parameters, dtype=object)
    order = np.argsort(total)[::-1] if sort else np.arange(total.size)
    names, first, total = names[order], first[order], total[order]
    errors = None
    if confidence is not None:
        raw = np.asarray(confidence, dtype=float)
        if raw.ndim == 0:
            errors = float(raw)
        elif raw.ndim == 1:
            if raw.size != total.size:
                raise ValueError("confidence must be scalar, length n, (n, 2), or (2, n)")
            errors = raw[order]
        elif raw.shape == (total.size, 2):
            errors = raw[order].T
        elif raw.shape == (2, total.size):
            errors = raw[:, order]
        else:
            raise ValueError("confidence must be scalar, length n, (n, 2), or (2, n)")
    x = np.arange(names.size)
    width = 0.38
    cfg = coerce_config(config, xlabel="Parameter", ylabel="Sobol index", ylim=(0, max(1.0, float(np.nanmax(total) * 1.15))))
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        bars_first = axis.bar(x - width / 2, first, width, label="First order", color=resolved.colors[0])
        bars_total = axis.bar(x + width / 2, total, width, label="Total order", color=resolved.colors[1], yerr=errors, capsize=2 if errors is not None else 0)
        axis.set_xticks(x, names, rotation=35, ha="right")
        interaction = total - first
        return finalize(fig, axis, config=cfg, artists={"first_order": bars_first, "total_order": bars_total}, data={"parameters": names, "first_order": first, "total_order": total, "interaction": interaction}, metrics={"dominant_parameter": str(names[0]) if names.size else ""}, theme=theme)


@register_plot(category="multiscale", aliases=("uncertainty-fan", "ensemble-fan"))
def uncertainty_fan_plot(
    coordinate: Any,
    ensemble: Any,
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    reference: Any | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Nested ensemble-quantile fan for propagated multiscale uncertainty."""
    x = as_array(coordinate)
    samples = np.asarray(ensemble, dtype=float)
    if samples.ndim != 2:
        raise ValueError("ensemble must have shape (samples, coordinate)")
    if samples.shape[1] != x.size and samples.shape[0] == x.size:
        samples = samples.T
    if samples.shape[1] != x.size:
        raise ValueError("ensemble coordinate dimension must match")
    q = np.asarray(quantiles, dtype=float)
    if np.any((q < 0) | (q > 1)) or q.size < 3:
        raise ValueError("quantiles must contain at least three values in [0, 1]")
    curves = np.quantile(samples, q, axis=0)
    median_index = int(np.argmin(np.abs(q - 0.5)))
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Prediction")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        bands = []
        pairs = min(median_index, q.size - median_index - 1)
        for layer in range(pairs):
            low = curves[layer]
            high = curves[-layer - 1]
            alpha = 0.10 + 0.10 * (layer + 1)
            bands.append(axis.fill_between(x, low, high, color=resolved.colors[0], alpha=alpha, label=f"{q[layer]:.0%}–{q[-layer-1]:.0%}"))
        median = axis.plot(x, curves[median_index], color=resolved.colors[0], label="Median")[0]
        reference_line = axis.plot(x, as_array(reference), color="black", linestyle="--", label="Reference")[0] if reference is not None else None
        return finalize(fig, axis, config=cfg, artists={"bands": bands, "median": median, "reference": reference_line}, data={"coordinate": x, "ensemble": samples, "quantiles": q, "curves": curves}, metrics={"mean_interval_width": float(np.mean(curves[-1] - curves[0]))}, theme=theme)


@register_plot(category="multiscale", aliases=("spatial-error-map", "field-error-map"))
def spatial_error_map(
    reference: Any,
    prediction: Any,
    *,
    relative: bool = False,
    mask: Any | None = None,
    cmap: str = "coolwarm",
    symmetric: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Spatial absolute or relative error between resolved simulation fields."""
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if ref.shape != pred.shape or ref.ndim != 2:
        raise ValueError("reference and prediction must be equal-shape 2-D arrays")
    error = pred - ref
    if relative:
        error = error / np.maximum(np.abs(ref), 1e-12)
    if mask is not None:
        error = np.ma.array(error, mask=np.asarray(mask, dtype=bool))
    finite_error = error.compressed() if np.ma.isMaskedArray(error) else np.asarray(error).reshape(-1)
    finite_error = finite_error[np.isfinite(finite_error)]
    limit = float(np.nanpercentile(np.abs(finite_error), 99)) if symmetric and finite_error.size else None
    cfg = coerce_config(config, xlabel="x", ylabel="y", equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(error, origin="lower", cmap=cmap, vmin=-limit if limit else None, vmax=limit)
        colorbar = fig.colorbar(image, ax=axis, label="Relative error" if relative else "Error")
        values = finite_error
        mae = float(np.mean(np.abs(values))) if values.size else float("nan")
        rmse = float(np.sqrt(np.mean(values**2))) if values.size else float("nan")
        return finalize(fig, axis, config=cfg, artists={"error": image, "colorbar": colorbar}, data={"reference": ref, "prediction": pred, "error": error}, metrics={"mae": mae, "rmse": rmse}, theme=theme)


@register_plot(category="multiscale", aliases=("field-profile-comparison", "profile-validation"))
def field_profile_comparison(
    coordinate: Any,
    profiles: Mapping[str, Any],
    *,
    reference_name: str | None = None,
    normalize: bool = False,
    inset: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Overlay field profiles from multiple scales, solvers or experiments."""
    x = as_array(coordinate)
    if not profiles:
        raise ValueError("profiles cannot be empty")
    series = {name: as_array(value) for name, value in profiles.items()}
    for name, value in series.items():
        if value.size != x.size:
            raise ValueError(f"profile {name!r} must match coordinate")
    if normalize:
        series = {name: value / max(float(np.nanmax(np.abs(value))), 1e-15) for name, value in series.items()}
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Normalized field" if normalize else "Field")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines = {}
        for name, values in series.items():
            kwargs = {"linewidth": 1.5, "color": "black"} if name == reference_name else {}
            lines[name] = axis.plot(x, values, label=name, **kwargs)[0]
        inset_axis = None
        if inset is not None:
            from .panels import inset_zoom

            inset_axis = inset_zoom(axis, bounds=(0.55, 0.48, 0.4, 0.42), xlim=inset[0], ylim=inset[1])
        return finalize(fig, axis, config=cfg, artists={"profiles": lines, "inset": inset_axis}, data={"coordinate": x, "profiles": series}, theme=theme)


@register_plot(category="multiscale", aliases=("scale-transfer-matrix", "coupling-matrix"))
def scale_transfer_matrix_plot(
    matrix: Any,
    *,
    source_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
    annotate: bool = True,
    cmap: str = "Blues",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Information/parameter transfer strength between model scales."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows = make_labels(source_labels, values.shape[0], "Source")
    cols = make_labels(target_labels, values.shape[1], "Target")
    cfg = coerce_config(config, xlabel="Target scale", ylabel="Source scale")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(values, cmap=cmap, aspect="auto", origin="upper")
        axis.set_xticks(np.arange(len(cols)), cols, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(rows)), rows)
        texts = []
        if annotate:
            midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    texts.append(axis.text(j, i, f"{values[i, j]:.2g}", ha="center", va="center", color="white" if values[i, j] > midpoint else "black", fontsize="x-small"))
        colorbar = fig.colorbar(image, ax=axis, label="Transfer strength")
        return finalize(fig, axis, config=cfg, artists={"matrix": image, "annotations": texts, "colorbar": colorbar}, data={"matrix": values, "source_labels": rows, "target_labels": cols}, theme=theme)


@register_plot(category="multiscale", aliases=("cost-scaling", "runtime-scaling"))
def computational_cost_scaling_plot(
    problem_size: Any,
    runtime: Any,
    *,
    labels: Sequence[str] | None = None,
    memory: Any | None = None,
    fit_power_law: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Runtime and memory scaling for atomistic-to-continuum workflows."""
    size = as_array(problem_size)
    matrix = np.asarray(runtime, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != size.size:
        raise ValueError("runtime curves must match problem_size")
    names = make_labels(labels, matrix.shape[0], "Method")
    cfg = coerce_config(config, xlabel="Problem size", ylabel="Runtime", xscale="log", yscale="log")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lines, fits = [], []
        exponents = {}
        positive = size > 0
        for row, name in zip(matrix, names):
            line = axis.plot(size, row, marker="o", label=name)[0]
            lines.append(line)
            valid = positive & (row > 0)
            if fit_power_law and np.sum(valid) >= 2:
                slope, intercept = np.polyfit(np.log(size[valid]), np.log(row[valid]), 1)
                fits.append(axis.plot(size[valid], np.exp(intercept) * size[valid] ** slope, color=line.get_color(), linestyle="--", label=f"{name}: O(N^{slope:.2f})")[0])
                exponents[name] = float(slope)
        memory_axis = None
        memory_lines = []
        memory_matrix = None
        if memory is not None:
            memory_matrix = np.asarray(memory, dtype=float)
            if memory_matrix.ndim == 1:
                memory_matrix = memory_matrix[None, :]
            if memory_matrix.shape[1] != size.size:
                raise ValueError("memory curves must match problem_size")
            if memory_matrix.shape[0] not in {1, matrix.shape[0]}:
                raise ValueError("memory must contain one curve or one curve per runtime method")
            memory_axis = axis.twinx()
            for index, row in enumerate(memory_matrix):
                name = "Memory" if memory_matrix.shape[0] == 1 else f"{names[index]} memory"
                color = "black" if memory_matrix.shape[0] == 1 else lines[index].get_color()
                memory_lines.append(memory_axis.plot(size, row, color=color, marker="s", linestyle=":", alpha=0.7, label=name)[0])
            memory_axis.set_ylabel("Memory")
            memory_axis.set_yscale("log")
        axes = np.asarray([axis, memory_axis], dtype=object) if memory_axis is not None else axis
        return finalize(fig, axes, config=cfg, artists={"runtime": lines, "fits": fits, "memory": memory_lines}, data={"problem_size": size, "runtime": matrix, "memory": memory_matrix}, metrics={"scaling_exponents": exponents}, theme=theme)


@register_plot(category="multiscale", aliases=("experiment-simulation-overlay", "validation-overlay"))
def experiment_simulation_overlay(
    coordinate: Any,
    experiment: Any,
    simulation: Any,
    *,
    experiment_uncertainty: Any | None = None,
    simulation_uncertainty: Any | None = None,
    labels: tuple[str, str] = ("Experiment", "Simulation"),
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = "battery",
) -> PlotResult:
    """Experiment/simulation overlay with uncertainty and goodness-of-fit."""
    x, exp, mask = finite_xy(coordinate, experiment)
    sim = as_array(simulation)[mask]
    if sim.size != x.size:
        raise ValueError("simulation must match coordinate")
    def _masked_uncertainty(values: Any | None) -> Any | None:
        if values is None:
            return None
        raw = np.asarray(values, dtype=float)
        if raw.ndim == 0:
            return np.full(x.size, float(raw))
        if raw.ndim == 1:
            if raw.size != mask.size:
                raise ValueError("uncertainty must be scalar or match coordinate")
            return raw[mask]
        if raw.shape[-1] == mask.size:
            return raw[..., mask]
        raise ValueError("uncertainty must be scalar, length n, or have n as its last dimension")
    exp_error = _masked_uncertainty(experiment_uncertainty)
    sim_error = _masked_uncertainty(simulation_uncertainty)
    residual = sim - exp
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    cfg = coerce_config(config, xlabel="Coordinate", ylabel="Response")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        experiment_artist = axis.errorbar(x, exp, yerr=exp_error, fmt="o", markersize=3.5, color="black", capsize=2, label=labels[0])
        simulation_line = axis.plot(x, sim, color=resolved.colors[0], label=labels[1])[0]
        simulation_band = axis.fill_between(x, sim - sim_error, sim + sim_error, color=simulation_line.get_color(), alpha=0.18) if sim_error is not None else None
        axis.text(0.04, 0.96, f"RMSE={rmse:.3g}\nMAE={mae:.3g}", transform=axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        return finalize(fig, axis, config=cfg, artists={"experiment": experiment_artist, "simulation": simulation_line, "simulation_uncertainty": simulation_band}, data={"coordinate": x, "experiment": exp, "simulation": sim, "residual": residual}, metrics={"rmse": rmse, "mae": mae}, theme=theme)
