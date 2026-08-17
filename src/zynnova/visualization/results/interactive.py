from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._registry import register_plot
from ._utils import as_array, pareto_mask, regression_metrics


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go

        return go
    except Exception as exc:
        raise ImportError("interactive result plots require plotly>=5") from exc


@register_plot(category="interactive", aliases=("interactive-parity",))
def interactive_parity_plot(reference: Any, prediction: Any, *, labels: Sequence[str] | None = None, color: Any | None = None, title: str = "Prediction parity") -> Any:
    """Interactive Plotly parity plot with hover labels and regression metrics."""
    go = _plotly()
    ref = as_array(reference)
    pred = as_array(prediction)
    metrics = regression_metrics(ref, pred)
    low = float(min(ref.min(), pred.min()))
    high = float(max(ref.max(), pred.max()))
    figure = go.Figure()
    figure.add_trace(go.Scattergl(x=ref, y=pred, mode="markers", marker={"color": color, "colorscale": "Viridis", "showscale": color is not None, "size": 6, "opacity": 0.65}, text=labels, name="Samples"))
    figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line={"dash": "dash", "color": "black"}, name="Ideal"))
    figure.update_layout(title=f"{title}<br><sup>R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3g}, RMSE={metrics['rmse']:.3g}</sup>", xaxis_title="Reference", yaxis_title="Prediction", template="plotly_white")
    return figure


@register_plot(category="interactive", aliases=("interactive-embedding",))
def interactive_embedding_plot(embedding: Any, *, color: Any | None = None, labels: Sequence[str] | None = None, size: Any | float = 6, title: str = "Latent embedding") -> Any:
    """Interactive 2-D or 3-D embedding with rich hover information."""
    go = _plotly()
    coordinates = np.asarray(embedding, dtype=float)
    marker = {"color": color, "colorscale": "Viridis", "showscale": color is not None, "size": size, "opacity": 0.72}
    if coordinates.shape[1] >= 3:
        trace = go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2], mode="markers", marker=marker, text=labels)
    else:
        trace = go.Scattergl(x=coordinates[:, 0], y=coordinates[:, 1], mode="markers", marker=marker, text=labels)
    figure = go.Figure(trace)
    figure.update_layout(title=title, template="plotly_white")
    return figure


@register_plot(category="interactive", aliases=("interactive-pareto",))
def interactive_pareto_plot(objectives: Any, *, names: Sequence[str] | None = None, maximize: Sequence[bool] | bool = True, labels: Sequence[str] | None = None, color: Any | None = None) -> Any:
    """Interactive 2-D/3-D Pareto front for candidate exploration."""
    go = _plotly()
    matrix = np.asarray(objectives, dtype=float)
    mask = pareto_mask(matrix, maximize=maximize)
    objective_names = list(names or [f"Objective {index + 1}" for index in range(matrix.shape[1])])
    marker = {"color": color, "colorscale": "Viridis", "showscale": color is not None, "size": 6, "opacity": 0.55}
    front_marker = {"size": 9, "symbol": "circle-open", "line": {"width": 2, "color": "red"}}
    figure = go.Figure()
    if matrix.shape[1] == 2:
        figure.add_trace(go.Scattergl(x=matrix[:, 0], y=matrix[:, 1], mode="markers", marker=marker, text=labels, name="Candidates"))
        figure.add_trace(go.Scattergl(x=matrix[mask, 0], y=matrix[mask, 1], mode="markers", marker=front_marker, text=None if labels is None else np.asarray(labels)[mask], name="Pareto front"))
        figure.update_xaxes(title=objective_names[0])
        figure.update_yaxes(title=objective_names[1])
    elif matrix.shape[1] == 3:
        figure.add_trace(go.Scatter3d(x=matrix[:, 0], y=matrix[:, 1], z=matrix[:, 2], mode="markers", marker=marker, text=labels, name="Candidates"))
        figure.add_trace(go.Scatter3d(x=matrix[mask, 0], y=matrix[mask, 1], z=matrix[mask, 2], mode="markers", marker=front_marker, text=None if labels is None else np.asarray(labels)[mask], name="Pareto front"))
        figure.update_layout(scene={"xaxis_title": objective_names[0], "yaxis_title": objective_names[1], "zaxis_title": objective_names[2]})
    else:
        raise ValueError("interactive_pareto_plot supports 2 or 3 objectives")
    figure.update_layout(template="plotly_white", title="Pareto front")
    return figure


@register_plot(category="interactive", aliases=("interactive-volume",))
def interactive_volume_plot(volume: Any, *, isomin: float | None = None, isomax: float | None = None, surface_count: int = 12, opacity: float = 0.12, title: str = "3-D scalar field") -> Any:
    """Interactive Plotly volume rendering for fields and tomography."""
    go = _plotly()
    data = np.asarray(volume, dtype=float)
    z, y, x = np.indices(data.shape)
    figure = go.Figure(go.Volume(x=x.ravel(), y=y.ravel(), z=z.ravel(), value=data.ravel(), isomin=float(np.nanquantile(data, 0.2) if isomin is None else isomin), isomax=float(np.nanquantile(data, 0.95) if isomax is None else isomax), opacity=opacity, surface_count=surface_count, colorscale="Viridis"))
    figure.update_layout(title=title, template="plotly_white")
    return figure


@register_plot(category="interactive", aliases=("interactive-sankey",))
def interactive_sankey_plot(source: Sequence[str], target: Sequence[str], value: Any, *, title: str = "Flow diagram") -> Any:
    """Interactive Sankey flow diagram for mechanisms, states or data pipelines."""
    go = _plotly()
    source_arr = np.asarray(source, dtype=object)
    target_arr = np.asarray(target, dtype=object)
    values = as_array(value)
    labels = list(dict.fromkeys(np.concatenate([source_arr, target_arr]).tolist()))
    index = {label: position for position, label in enumerate(labels)}
    figure = go.Figure(go.Sankey(node={"label": labels}, link={"source": [index[item] for item in source_arr], "target": [index[item] for item in target_arr], "value": values}))
    figure.update_layout(title=title, template="plotly_white")
    return figure
