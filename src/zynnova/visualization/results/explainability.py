from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, make_labels


@register_plot(category="explainability", aliases=("importance",))
def feature_importance_plot(
    importance: Any,
    *,
    feature_names: Sequence[str] | None = None,
    uncertainty: Any | None = None,
    signed: bool = False,
    top_k: int | None = 20,
    sort: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Horizontal feature-importance ranking with optional confidence bars."""
    values = as_array(importance)
    names = np.asarray(make_labels(feature_names, values.size, "Feature"), dtype=object)
    errors = None if uncertainty is None else as_array(uncertainty)
    order = np.argsort(np.abs(values) if signed else values)
    if not sort:
        order = np.arange(values.size)
    if top_k is not None:
        order = order[-int(top_k):]
    values, names = values[order], names[order]
    if errors is not None:
        errors = errors[order]
    cfg = coerce_config(config, xlabel="Importance")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        colors = [resolved.colors[1] if value < 0 else resolved.colors[0] for value in values] if signed else resolved.colors[0]
        bars = axis.barh(np.arange(values.size), values, xerr=errors, color=colors, alpha=0.8, capsize=2)
        axis.set_yticks(np.arange(values.size), names)
        axis.axvline(0.0, color="black", linewidth=0.7)
        return finalize(fig, axis, config=cfg, artists={"bars": bars}, data={"importance": values, "feature_names": names, "uncertainty": errors})


@register_plot(category="explainability", aliases=("shap-summary", "attribution-beeswarm"))
def shap_beeswarm(
    shap_values: Any,
    feature_values: Any | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    top_k: int = 20,
    cmap: str = "coolwarm",
    point_size: float = 8.0,
    alpha: float = 0.65,
    jitter: float = 0.28,
    seed: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Dependency-free SHAP-style beeswarm for local attribution matrices."""
    shap = np.asarray(shap_values, dtype=float)
    if shap.ndim != 2:
        raise ValueError("shap_values must have shape (samples, features)")
    features = shap if feature_values is None else np.asarray(feature_values, dtype=float)
    if features.shape != shap.shape:
        raise ValueError("feature_values must match shap_values shape")
    names = np.asarray(make_labels(feature_names, shap.shape[1], "Feature"), dtype=object)
    importance = np.mean(np.abs(shap), axis=0)
    order = np.argsort(importance)[-min(top_k, shap.shape[1]):]
    rng = np.random.default_rng(seed)
    cfg = coerce_config(config, xlabel="Attribution value")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        scatters = []
        for row, feature_index in enumerate(order):
            values = shap[:, feature_index]
            feature = features[:, feature_index]
            normalized = (feature - np.nanmin(feature)) / max(np.nanmax(feature) - np.nanmin(feature), 1e-15)
            y = row + rng.normal(0.0, jitter / 3.0, size=values.size)
            scatters.append(axis.scatter(values, y, c=normalized, cmap=cmap, vmin=0, vmax=1, s=point_size, alpha=alpha, edgecolors="none", rasterized=cfg.rasterized))
        axis.axvline(0.0, color="black", linewidth=0.7)
        axis.set_yticks(np.arange(order.size), names[order])
        colorbar = fig.colorbar(scatters[-1], ax=axis, label="Feature value (normalized)") if scatters else None
        return finalize(fig, axis, config=cfg, artists={"points": scatters, "colorbar": colorbar}, data={"shap_values": shap[:, order], "feature_values": features[:, order], "feature_names": names[order], "importance": importance[order]})


@register_plot(category="explainability", aliases=("attribution-waterfall",))
def attribution_waterfall(
    contributions: Any,
    *,
    feature_names: Sequence[str] | None = None,
    baseline: float = 0.0,
    prediction: float | None = None,
    top_k: int | None = 15,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Local additive-explanation waterfall from baseline to prediction."""
    values = as_array(contributions)
    names = np.asarray(make_labels(feature_names, values.size, "Feature"), dtype=object)
    order = np.argsort(np.abs(values))[::-1]
    if top_k is not None and values.size > top_k:
        keep = order[: top_k - 1]
        other = np.sum(values[order[top_k - 1:]])
        values = np.r_[values[keep], other]
        names = np.r_[names[keep], "Other"]
    else:
        values, names = values[order], names[order]
    starts = baseline + np.r_[0.0, np.cumsum(values[:-1])]
    final_prediction = baseline + np.sum(values) if prediction is None else float(prediction)
    cfg = coerce_config(config, xlabel="Model output")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        colors = [resolved.colors[0] if value >= 0 else resolved.colors[1] for value in values]
        bars = axis.barh(np.arange(values.size), values, left=starts, color=colors, alpha=0.82)
        for y, start, value in zip(np.arange(values.size), starts, values):
            axis.annotate(f"{value:+.3g}", (start + value, y), xytext=(3 if value >= 0 else -3, 0), textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize="small")
        axis.set_yticks(np.arange(values.size), names)
        axis.invert_yaxis()
        baseline_line = axis.axvline(baseline, color="0.4", linestyle="--", label=f"Baseline {baseline:.3g}")
        prediction_line = axis.axvline(final_prediction, color="black", linewidth=1.2, label=f"Prediction {final_prediction:.3g}")
        return finalize(fig, axis, config=cfg, artists={"bars": bars, "baseline": baseline_line, "prediction": prediction_line}, data={"contributions": values, "feature_names": names, "starts": starts}, metrics={"baseline": baseline, "prediction": final_prediction})


@register_plot(category="explainability", aliases=("partial-dependence", "ice"))
def partial_dependence_plot(
    x: Any,
    average: Any,
    *,
    individual: Any | None = None,
    lower: Any | None = None,
    upper: Any | None = None,
    feature_name: str = "Feature",
    target_name: str = "Prediction",
    center: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Partial-dependence curve with ICE trajectories and uncertainty band."""
    x_arr = as_array(x)
    avg = as_array(average)
    ice = None if individual is None else np.asarray(individual, dtype=float)
    if center:
        avg = avg - avg[0]
        if ice is not None:
            ice = ice - ice[:, [0]]
    cfg = coerce_config(config, xlabel=feature_name, ylabel=target_name)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        ice_lines = []
        if ice is not None:
            for row in ice:
                ice_lines.extend(axis.plot(x_arr, row, color="0.65", alpha=0.15, linewidth=0.6))
        line = axis.plot(x_arr, avg, color="black", linewidth=1.8, label="Average")[0]
        band = None
        if lower is not None and upper is not None:
            band = axis.fill_between(x_arr, as_array(lower), as_array(upper), color="black", alpha=0.15)
        return finalize(fig, axis, config=cfg, artists={"individual": ice_lines, "average": line, "band": band}, data={"x": x_arr, "average": avg, "individual": ice})


@register_plot(category="explainability", aliases=("interaction-map",))
def interaction_heatmap(
    interactions: Any,
    *,
    feature_names: Sequence[str] | None = None,
    cluster: bool = True,
    cmap: str = "magma",
    annotate: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Symmetric interaction-strength heatmap with optional clustering."""
    matrix = np.asarray(interactions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("interactions must be a square matrix")
    names = np.asarray(make_labels(feature_names, matrix.shape[0], "Feature"), dtype=object)
    order = np.arange(matrix.shape[0])
    if cluster and matrix.shape[0] > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform

            maximum = np.nanmax(np.abs(matrix))
            distance = maximum - np.abs(matrix)
            order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
            matrix, names = matrix[np.ix_(order, order)], names[order]
        except Exception:
            pass
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        axis.set_xticks(np.arange(names.size), names, rotation=45, ha="right")
        axis.set_yticks(np.arange(names.size), names)
        if annotate and names.size <= 20:
            for i in range(names.size):
                for j in range(names.size):
                    axis.text(j, i, f"{matrix[i, j]:.2g}", ha="center", va="center", fontsize="x-small")
        colorbar = fig.colorbar(image, ax=axis, label="Interaction strength")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"interactions": matrix, "feature_names": names, "order": order})


@register_plot(category="explainability", aliases=("attention",))
def attention_map(
    attention: Any,
    *,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    normalize: str | None = None,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Attention or attribution matrix for tokens, atoms, residues or graph nodes."""
    matrix = np.asarray(attention, dtype=float)
    if matrix.ndim == 3:
        matrix = np.mean(matrix, axis=0)
    if matrix.ndim != 2:
        raise ValueError("attention must be 2-D or heads × query × key")
    if normalize == "row":
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-15)
    elif normalize == "column":
        matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-15)
    cfg = coerce_config(config)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(matrix, cmap=cmap, aspect="auto")
        if x_labels is not None:
            axis.set_xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        if y_labels is not None:
            axis.set_yticks(np.arange(len(y_labels)), y_labels)
        colorbar = fig.colorbar(image, ax=axis, label="Attention")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"attention": matrix})
