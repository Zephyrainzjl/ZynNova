from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import (
    as_array,
    classification_metrics,
    density_values,
    finite_xy,
    format_metric,
    make_labels,
    regression_metrics,
    robust_limits,
)


@register_plot(category="model-evaluation", aliases=("prediction-parity", "predicted-vs-true"))
def parity_plot(
    reference: Any,
    prediction: Any,
    *,
    groups: Any | None = None,
    mode: str = "density",
    bins: int = 70,
    point_size: float = 10.0,
    alpha: float = 0.65,
    identity: bool = True,
    tolerance: float | None = None,
    annotate_metrics: bool = True,
    metric_position: tuple[float, float] = (0.04, 0.96),
    cmap: str = "viridis",
    labels: Mapping[Any, str] | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Publication parity plot with density, hexbin or grouped scatter modes."""
    y_true, y_pred, mask = finite_xy(reference, prediction)
    metrics = regression_metrics(y_true, y_pred)
    low, high = robust_limits(np.concatenate([y_true, y_pred]), quantiles=(0.0, 1.0), pad=0.04)
    cfg = coerce_config(config, xlabel="Reference", ylabel="Prediction", xlim=(low, high), ylim=(low, high), equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists: dict[str, Any] = {}
        if groups is not None:
            group_arr = np.asarray(groups).reshape(-1)[mask]
            scatters = []
            for group in dict.fromkeys(group_arr.tolist()):
                selected = group_arr == group
                scatters.append(axis.scatter(y_true[selected], y_pred[selected], s=point_size, alpha=alpha, label=(labels or {}).get(group, str(group)), rasterized=cfg.rasterized))
            artists["points"] = scatters
        elif mode == "hexbin":
            artists["points"] = axis.hexbin(y_true, y_pred, gridsize=bins, mincnt=1, cmap=cmap, bins="log")
            artists["colorbar"] = fig.colorbar(artists["points"], ax=axis, label="log count")
        elif mode == "density":
            density = density_values(y_true, y_pred)
            order = np.argsort(density)
            artists["points"] = axis.scatter(y_true[order], y_pred[order], c=density[order], s=point_size, alpha=alpha, cmap=cmap, rasterized=cfg.rasterized)
            artists["colorbar"] = fig.colorbar(artists["points"], ax=axis, label="Point density")
        else:
            artists["points"] = axis.scatter(y_true, y_pred, s=point_size, alpha=alpha, rasterized=cfg.rasterized)
        if identity:
            artists["identity"] = axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9, label="Ideal")[0]
        if tolerance is not None:
            artists["tolerance"] = axis.fill_between([low, high], [low - tolerance, high - tolerance], [low + tolerance, high + tolerance], color="0.5", alpha=0.12, label=f"±{tolerance:g}")
        if annotate_metrics:
            text = "\n".join([
                f"$R^2$ = {format_metric(float(metrics['r2']))}",
                f"MAE = {format_metric(float(metrics['mae']))}",
                f"RMSE = {format_metric(float(metrics['rmse']))}",
                f"n = {int(metrics['n'])}",
            ])
            artists["metrics"] = axis.text(metric_position[0], metric_position[1], text, transform=axis.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.5})
        return finalize(fig, axis, config=cfg, artists=artists, data={"reference": y_true, "prediction": y_pred}, metrics=metrics)


@register_plot(category="model-evaluation", aliases=("residual-dashboard",))
def residual_diagnostics(
    reference: Any,
    prediction: Any,
    *,
    features: Any | None = None,
    feature_label: str = "Feature",
    bins: int = 30,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Four-panel residual diagnostics: parity, residual, QQ and distribution."""
    y_true, y_pred, _ = finite_xy(reference, prediction)
    residual = y_pred - y_true
    cfg = coerce_config(config, figsize=(8.0, 6.5))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=2)
        parity_plot(y_true, y_pred, ax=axes[0, 0], config=PlotConfig(equal_aspect=True, legend=False), theme=theme)
        x = y_pred if features is None else as_array(features)[: residual.size]
        axes[0, 1].scatter(x, residual, s=10, alpha=0.45, rasterized=cfg.rasterized)
        axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[0, 1].set_xlabel("Prediction" if features is None else feature_label)
        axes[0, 1].set_ylabel("Residual")
        try:
            from scipy import stats

            stats.probplot(residual, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Normal Q–Q")
        except Exception:
            ordered = np.sort(residual)
            probability = (np.arange(ordered.size) + 0.5) / ordered.size
            axes[1, 0].plot(probability, ordered, marker="o", linestyle="none")
        axes[1, 1].hist(residual, bins=bins, density=True, alpha=0.6)
        axes[1, 1].axvline(np.mean(residual), color="black", linestyle="--", label=f"Bias={np.mean(residual):.3g}")
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("Density")
        for axis in axes.flat:
            axis.grid(False)
        return finalize(fig, axes, config=cfg, data={"reference": y_true, "prediction": y_pred, "residual": residual}, metrics=regression_metrics(y_true, y_pred))


@register_plot(category="model-evaluation", aliases=("confusion",))
def confusion_matrix_plot(
    matrix_or_true: Any,
    predicted: Any | None = None,
    *,
    labels: Sequence[str] | None = None,
    normalize: str | None = None,
    cmap: str = "Blues",
    annotate: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Confusion matrix from raw labels or a precomputed square matrix."""
    if predicted is None:
        matrix = np.asarray(matrix_or_true, dtype=float)
    else:
        truth = np.asarray(matrix_or_true).reshape(-1)
        pred = np.asarray(predicted).reshape(-1)
        classes = np.unique(np.concatenate([truth, pred]))
        matrix = np.zeros((classes.size, classes.size), dtype=float)
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i, j] = np.sum((truth == true_class) & (pred == pred_class))
        if labels is None:
            labels = [str(item) for item in classes]
    raw = matrix.copy()
    if normalize == "true":
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-15)
    elif normalize == "pred":
        matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-15)
    elif normalize == "all":
        matrix = matrix / max(matrix.sum(), 1e-15)
    names = make_labels(labels, matrix.shape[0], "Class")
    cfg = coerce_config(config, xlabel="Predicted", ylabel="True")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(matrix, cmap=cmap, aspect="equal")
        axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(names)), names)
        if annotate:
            threshold = np.nanmax(matrix) * 0.55 if matrix.size else 0.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = f"{matrix[i, j]:.2f}" if normalize else f"{int(raw[i, j])}"
                    axis.text(j, i, text, ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")
        colorbar = fig.colorbar(image, ax=axis, label="Fraction" if normalize else "Count")
        accuracy = float(np.trace(raw) / max(raw.sum(), 1.0))
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"matrix": matrix, "raw_matrix": raw}, metrics={"accuracy": accuracy})


@register_plot(category="model-evaluation", aliases=("roc",))
def roc_curve_plot(
    y_true: Any,
    scores: Any,
    *,
    label: str | None = None,
    bootstrap: int = 0,
    seed: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """ROC curve with AUC and optional bootstrap confidence interval."""
    truth = as_array(y_true, dtype=int)
    score = as_array(scores)
    try:
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, thresholds = roc_curve(truth, score)
        auc_value = float(auc(fpr, tpr))
    except Exception:
        thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
        tpr, fpr = [], []
        for threshold in thresholds:
            predicted = score >= threshold
            tpr.append(np.sum(predicted & (truth == 1)) / max(np.sum(truth == 1), 1))
            fpr.append(np.sum(predicted & (truth == 0)) / max(np.sum(truth == 0), 1))
        fpr, tpr = np.asarray(fpr), np.asarray(tpr)
        order = np.argsort(fpr)
        fpr, tpr = fpr[order], tpr[order]
        auc_value = float(np.trapezoid(tpr, fpr))
    ci_low = ci_high = float("nan")
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(bootstrap):
            indices = rng.integers(0, truth.size, truth.size)
            if np.unique(truth[indices]).size < 2:
                continue
            try:
                from sklearn.metrics import roc_auc_score

                estimates.append(roc_auc_score(truth[indices], score[indices]))
            except Exception:
                pass
        if estimates:
            ci_low, ci_high = np.quantile(estimates, [0.025, 0.975])
    cfg = coerce_config(config, xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        line_label = label or f"AUC = {auc_value:.3f}"
        if bootstrap > 0 and np.isfinite(ci_low):
            line_label += f" [{ci_low:.3f}, {ci_high:.3f}]"
        curve = axis.plot(fpr, tpr, label=line_label)[0]
        diagonal = axis.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=0.8)[0]
        return finalize(fig, axis, config=cfg, artists={"curve": curve, "chance": diagonal}, data={"fpr": fpr, "tpr": tpr, "thresholds": thresholds}, metrics={"auc": auc_value, "auc_ci_low": float(ci_low), "auc_ci_high": float(ci_high)})


@register_plot(category="model-evaluation", aliases=("pr-curve", "precision-recall"))
def precision_recall_plot(
    y_true: Any,
    scores: Any,
    *,
    label: str | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Precision–recall curve with average precision."""
    truth = as_array(y_true, dtype=int)
    score = as_array(scores)
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(truth, score)
        average_precision = float(average_precision_score(truth, score))
    except Exception:
        thresholds = np.sort(np.unique(score))[::-1]
        precision, recall = [], []
        for threshold in thresholds:
            metrics = classification_metrics(truth, score, threshold=threshold)
            precision.append(metrics["precision"])
            recall.append(metrics["recall"])
        precision, recall = np.asarray(precision), np.asarray(recall)
        order = np.argsort(recall)
        average_precision = float(np.trapezoid(precision[order], recall[order]))
    prevalence = float(np.mean(truth == 1))
    cfg = coerce_config(config, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        curve = axis.plot(recall, precision, label=label or f"AP = {average_precision:.3f}")[0]
        baseline = axis.axhline(prevalence, color="0.4", linestyle="--", linewidth=0.8, label=f"Prevalence = {prevalence:.3f}")
        return finalize(fig, axis, config=cfg, artists={"curve": curve, "baseline": baseline}, data={"precision": precision, "recall": recall, "thresholds": thresholds}, metrics={"average_precision": average_precision, "prevalence": prevalence})


@register_plot(category="model-evaluation", aliases=("benchmark-radar",))
def radar_plot(
    values: Any,
    *,
    categories: Sequence[str],
    series_labels: Sequence[str] | None = None,
    normalize: bool = False,
    fill_alpha: float = 0.12,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Radar chart for compact multi-metric benchmark comparisons."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[1] != len(categories):
        raise ValueError("categories must match metric count")
    if normalize:
        lo = np.nanmin(matrix, axis=0)
        hi = np.nanmax(matrix, axis=0)
        matrix = (matrix - lo) / np.maximum(hi - lo, 1e-15)
    labels = make_labels(series_labels, matrix.shape[0], "Model")
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    cfg = coerce_config(config, figsize=(5.0, 5.0))
    with theme_context(theme):
        fig, axis, _ = create_axes(config=cfg, theme=theme, projection="polar")
        artists = []
        for row, label in zip(matrix, labels):
            closed = np.r_[row, row[0]]
            line = axis.plot(closed_angles, closed, label=label)[0]
            axis.fill(closed_angles, closed, color=line.get_color(), alpha=fill_alpha)
            artists.append(line)
        axis.set_xticks(angles, categories)
        return finalize(fig, axis, config=cfg, artists={"series": artists}, data={"values": matrix, "categories": list(categories), "labels": labels})


@register_plot(category="model-evaluation", aliases=("performance-profile",))
def performance_profile(
    scores: Any,
    *,
    method_labels: Sequence[str] | None = None,
    lower_is_better: bool = True,
    max_ratio: float | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Dolan–Moré style performance profile across benchmark tasks."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("scores must have shape (tasks, methods)")
    best = np.nanmin(matrix, axis=1, keepdims=True) if lower_is_better else np.nanmax(matrix, axis=1, keepdims=True)
    ratios = matrix / np.maximum(best, 1e-15) if lower_is_better else best / np.maximum(matrix, 1e-15)
    finite = ratios[np.isfinite(ratios)]
    upper = float(max_ratio or np.nanquantile(finite, 0.98))
    tau = np.linspace(1.0, max(upper, 1.01), 300)
    labels = make_labels(method_labels, matrix.shape[1], "Method")
    profiles = np.asarray([np.mean(ratios[:, j, None] <= tau[None, :], axis=0) for j in range(matrix.shape[1])])
    cfg = coerce_config(config, xlabel="Performance ratio τ", ylabel="Fraction of tasks", ylim=(0, 1))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = [axis.plot(tau, profile, label=label)[0] for profile, label in zip(profiles, labels)]
        return finalize(fig, axis, config=cfg, artists={"profiles": artists}, data={"tau": tau, "profiles": profiles, "ratios": ratios, "labels": labels})
