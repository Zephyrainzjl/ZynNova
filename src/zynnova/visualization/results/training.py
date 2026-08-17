from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, make_labels, rolling_statistics


@register_plot(category="training", aliases=("learning-curves", "history"))
def training_history_plot(
    history: Mapping[str, Any],
    *,
    x: Any | None = None,
    metric_groups: Sequence[Sequence[str]] | None = None,
    log_y: bool = False,
    smooth_window: int = 1,
    best: str | None = None,
    best_mode: str = "min",
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Flexible training-history panels for losses, metrics and learning rates."""
    metrics = list(history)
    groups = [metrics] if metric_groups is None else [list(group) for group in metric_groups]
    length = max(len(as_array(history[name])) for name in metrics)
    x_arr = np.arange(length) if x is None else as_array(x)
    cfg = coerce_config(config, figsize=(7.0, 2.8 * len(groups)))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=len(groups), ncols=1, sharex=True)
        axes_arr = np.atleast_1d(axes)
        artists = {}
        for axis, group in zip(axes_arr, groups):
            for name in group:
                values = as_array(history[name])
                xx = x_arr[: values.size]
                if smooth_window > 1 and values.size >= smooth_window:
                    mean, _ = rolling_statistics(values, smooth_window)
                    xx_plot = xx[smooth_window - 1:]
                    line = axis.plot(xx_plot, mean, label=name)[0]
                    axis.plot(xx, values, color=line.get_color(), alpha=0.12, linewidth=0.6)
                else:
                    line = axis.plot(xx, values, label=name)[0]
                artists[name] = line
            if log_y:
                axis.set_yscale("log")
            axis.set_ylabel(" / ".join(group))
        axes_arr[-1].set_xlabel("Epoch / step")
        metrics_out = {}
        if best and best in history:
            values = as_array(history[best])
            index = int(np.nanargmin(values) if best_mode == "min" else np.nanargmax(values))
            for axis in axes_arr:
                axis.axvline(x_arr[index], color="black", linestyle="--", linewidth=0.8, label=f"Best {best}")
            metrics_out = {"best_metric": best, "best_index": index, "best_value": float(values[index])}
        return finalize(fig, axes, config=cfg, artists=artists, data={"history": dict(history), "x": x_arr}, metrics=metrics_out)


@register_plot(category="training", aliases=("lr-schedule", "learning-rate"))
def learning_rate_schedule_plot(
    step: Any,
    learning_rate: Any,
    *,
    loss: Any | None = None,
    log_y: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Learning-rate schedule with optional synchronized loss."""
    x = as_array(step)
    lr = as_array(learning_rate)
    cfg = coerce_config(config, xlabel="Step", ylabel="Learning rate", yscale="log" if log_y else None)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        lr_line = axis.plot(x, lr, label="Learning rate")[0]
        loss_axis = None
        loss_line = None
        if loss is not None:
            loss_axis = axis.twinx()
            loss_line = loss_axis.plot(x, as_array(loss), color="black", alpha=0.5, label="Loss")[0]
            loss_axis.set_ylabel("Loss")
        return finalize(fig, axis, config=cfg, artists={"learning_rate": lr_line, "loss_axis": loss_axis, "loss": loss_line}, data={"step": x, "learning_rate": lr, "loss": loss})


@register_plot(category="training", aliases=("gradient-flow",))
def gradient_flow_plot(
    gradients: Mapping[str, Any],
    *,
    log_y: bool = True,
    show_max: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Layerwise gradient magnitude diagnostic for vanishing or exploding gradients."""
    names = list(gradients)
    means = np.asarray([np.mean(np.abs(as_array(gradients[name]))) for name in names])
    maxima = np.asarray([np.max(np.abs(as_array(gradients[name]))) for name in names])
    cfg = coerce_config(config, xlabel="Layer", ylabel="Gradient magnitude", yscale="log" if log_y else None, figsize=(max(6.0, len(names) * 0.25), 3.5))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        x = np.arange(len(names))
        mean_bars = axis.bar(x, means, label="Mean |grad|", alpha=0.75)
        max_points = axis.scatter(x, maxima, marker="_", s=50, color="black", label="Max |grad|") if show_max else None
        axis.set_xticks(x, names, rotation=90)
        return finalize(fig, axis, config=cfg, artists={"mean": mean_bars, "max": max_points}, data={"layer_names": names, "mean_gradient": means, "max_gradient": maxima})


@register_plot(category="training", aliases=("dataset-cartography",))
def dataset_cartography_plot(
    confidence: Any,
    variability: Any,
    *,
    correctness: Any | None = None,
    labels: Sequence[str] | None = None,
    annotate: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Dataset cartography: confidence versus variability and correctness."""
    confidence_arr = as_array(confidence)
    variability_arr = as_array(variability)
    color = correctness
    cfg = coerce_config(config, xlabel="Mean confidence", ylabel="Variability")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        points = axis.scatter(confidence_arr, variability_arr, c=color, cmap="viridis", s=12, alpha=0.65, edgecolors="none", rasterized=True)
        colorbar = fig.colorbar(points, ax=axis, label="Correctness") if correctness is not None else None
        if labels is not None and annotate > 0:
            score = variability_arr + np.abs(confidence_arr - 0.5)
            selected = np.argsort(score)[-annotate:]
            label_arr = np.asarray(labels)
            for index in selected:
                axis.annotate(str(label_arr[index]), (confidence_arr[index], variability_arr[index]), xytext=(3, 3), textcoords="offset points", fontsize="x-small")
        return finalize(fig, axis, config=cfg, artists={"points": points, "colorbar": colorbar}, data={"confidence": confidence_arr, "variability": variability_arr, "correctness": correctness})


@register_plot(category="training", aliases=("benchmark-heatmap",))
def benchmark_heatmap(
    scores: Any,
    *,
    model_names: Sequence[str],
    task_names: Sequence[str],
    higher_is_better: bool = True,
    rank: bool = False,
    annotate: bool = True,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Model × task benchmark matrix with raw scores or within-task ranks."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.shape != (len(model_names), len(task_names)):
        raise ValueError("scores must have shape (models, tasks)")
    displayed = matrix.copy()
    if rank:
        try:
            from scipy.stats import rankdata

            displayed = np.asarray([rankdata(-column if higher_is_better else column) for column in matrix.T]).T
        except Exception:
            displayed = np.asarray([np.argsort(np.argsort(-column if higher_is_better else column)) + 1 for column in matrix.T]).T
    cfg = coerce_config(config, figsize=(max(5.0, len(task_names) * 0.55), max(3.0, len(model_names) * 0.42)))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(displayed, cmap=cmap, aspect="auto")
        axis.set_xticks(np.arange(len(task_names)), task_names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(model_names)), model_names)
        if annotate and matrix.size <= 500:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, f"{displayed[i, j]:.2g}", ha="center", va="center", fontsize="x-small")
        colorbar = fig.colorbar(image, ax=axis, label="Rank" if rank else "Score")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"scores": matrix, "displayed": displayed})
