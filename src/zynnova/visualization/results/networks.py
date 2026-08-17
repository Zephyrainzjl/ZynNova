from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, make_labels, validate_square


@register_plot(category="networks", aliases=("graph", "network"))
def network_plot(
    adjacency_or_graph: Any,
    *,
    node_labels: Sequence[str] | None = None,
    node_values: Any | None = None,
    node_sizes: Any | float = 80.0,
    edge_values: Any | None = None,
    layout: str = "spring",
    directed: bool = False,
    weighted: bool = True,
    cmap: str = "viridis",
    seed: int = 0,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """General graph visualization with reusable node and edge styling."""
    try: import networkx as nx
    except Exception as exc: raise ImportError("network_plot requires networkx") from exc
    if hasattr(adjacency_or_graph,"nodes"): graph=adjacency_or_graph.copy()
    else:
        adjacency=np.asarray(adjacency_or_graph,dtype=float); create=nx.DiGraph if directed else nx.Graph; graph=nx.from_numpy_array(adjacency,create_using=create)
    layouts={"spring":lambda:nx.spring_layout(graph,seed=seed,weight="weight" if weighted else None),"kamada-kawai":lambda:nx.kamada_kawai_layout(graph,weight="weight" if weighted else None),"circular":lambda:nx.circular_layout(graph),"shell":lambda:nx.shell_layout(graph),"spectral":lambda:nx.spectral_layout(graph),"planar":lambda:nx.planar_layout(graph)}
    if layout not in layouts: raise ValueError(f"unknown layout {layout!r}; available={sorted(layouts)}")
    positions=layouts[layout](); nodes=list(graph.nodes); edges=list(graph.edges)
    labels={node:str(node) for node in nodes} if node_labels is None else {node:str(node_labels[i]) for i,node in enumerate(nodes)}
    if node_labels is not None and len(node_labels)!=len(nodes): raise ValueError("node_labels must match nodes")
    if edge_values is None: edge_scalar=np.asarray([graph.edges[e].get("weight",1.0) for e in edges],dtype=float)
    else:
        raw=np.asarray(edge_values,dtype=float)
        if raw.ndim==2: edge_scalar=np.asarray([raw[u,v] for u,v in edges],dtype=float)
        else: edge_scalar=raw.reshape(-1)
        if edge_scalar.size!=len(edges): raise ValueError("edge_values must match edges or be an adjacency-shaped matrix")
    magnitude=np.abs(edge_scalar); widths=0.5+2.5*magnitude/max(float(np.nanmax(magnitude)) if magnitude.size else 1.0,1e-15)
    cfg=coerce_config(config,equal_aspect=True)
    with theme_context(theme):
        fig,axis,_=create_axes(ax=ax,config=cfg,theme=theme)
        node_artist=nx.draw_networkx_nodes(graph,positions,node_color=node_values if node_values is not None else np.arange(len(nodes)),node_size=node_sizes,cmap=cmap,ax=axis)
        edge_artist=nx.draw_networkx_edges(graph,positions,width=widths,edge_color=edge_scalar,edge_cmap=__import__('matplotlib').colormaps[cmap],alpha=0.55,arrows=directed,ax=axis)
        label_artist=nx.draw_networkx_labels(graph,positions,labels=labels,font_size=7,ax=axis); axis.set_axis_off()
        edge_colorbar=None
        if edge_scalar.size and np.nanmax(edge_scalar)!=np.nanmin(edge_scalar):
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            edge_colorbar=fig.colorbar(ScalarMappable(norm=Normalize(np.nanmin(edge_scalar),np.nanmax(edge_scalar)),cmap=cmap),ax=axis,label="Edge value")
        return finalize(fig,axis,config=cfg,artists={"nodes":node_artist,"edges":edge_artist,"labels":label_artist,"edge_colorbar":edge_colorbar},data={"graph":graph,"positions":positions,"edge_values":edge_scalar},theme=theme)


@register_plot(category="networks", aliases=("reaction-network", "pathway-network"))
def reaction_network_plot(
    species: Sequence[str],
    reactions: Sequence[tuple[str, str, float]],
    *,
    node_values: Mapping[str, float] | None = None,
    edge_labels: bool = True,
    layout: str = "spring",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Directed chemical or biochemical reaction network with rate-weighted edges."""
    try:
        import networkx as nx
    except Exception as exc:
        raise ImportError("reaction_network_plot requires networkx") from exc
    graph = nx.DiGraph()
    graph.add_nodes_from(species)
    for source, target, rate in reactions:
        graph.add_edge(source, target, weight=float(rate), label=f"{rate:.3g}")
    result = network_plot(graph, node_labels=list(species), node_values=None if node_values is None else [node_values.get(item, 0.0) for item in species], layout=layout, directed=True, ax=ax, config=config, theme=theme)
    if edge_labels:
        positions = result.data["positions"]
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=nx.get_edge_attributes(graph, "label"), font_size=6, ax=result.ax)
    result.data["reactions"] = reactions
    return result


@register_plot(category="networks", aliases=("adjacency",))
def adjacency_matrix_plot(
    adjacency: Any,
    *,
    labels: Sequence[str] | None = None,
    reorder: bool = True,
    log_scale: bool = False,
    cmap: str = "magma",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Adjacency or contact matrix with community-aware reordering."""
    matrix = validate_square(adjacency)
    names = np.asarray(make_labels(labels, matrix.shape[0], "Node"), dtype=object)
    order = np.arange(matrix.shape[0])
    if reorder and matrix.shape[0] > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage

            order = leaves_list(linkage(matrix, method="average", metric="euclidean"))
            matrix, names = matrix[np.ix_(order, order)], names[order]
        except Exception:
            pass
    displayed = np.log1p(np.abs(matrix)) * np.sign(matrix) if log_scale else matrix
    cfg = coerce_config(config, equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.imshow(displayed, cmap=cmap, origin="lower", aspect="equal")
        if names.size <= 50:
            axis.set_xticks(np.arange(names.size), names, rotation=90)
            axis.set_yticks(np.arange(names.size), names)
        colorbar = fig.colorbar(image, ax=axis, label="log(1+weight)" if log_scale else "Weight")
        return finalize(fig, axis, config=cfg, artists={"image": image, "colorbar": colorbar}, data={"adjacency": matrix, "order": order, "labels": names})


@register_plot(category="networks", aliases=("chord",))
def chord_diagram(
    matrix: Any,
    *,
    labels: Sequence[str] | None = None,
    threshold: float = 0.0,
    node_width: float = 0.08,
    alpha: float = 0.35,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Circular chord diagram for transitions, interactions or flows."""
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    values = validate_square(matrix)
    names = make_labels(labels, values.shape[0], "Node")
    n = values.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cfg = coerce_config(config, figsize=(6.0, 6.0), equal_aspect=True)
    with theme_context(theme):
        fig, axis, resolved = create_axes(config=cfg, theme=theme)
        axis.set_aspect("equal")
        axis.set_axis_off()
        nodes, chords = [], []
        for index, angle in enumerate(angles):
            x, y = np.cos(angle), np.sin(angle)
            node = mpatches.Circle((x, y), node_width, facecolor=resolved.colors[index % len(resolved.colors)], edgecolor="white", linewidth=0.8, zorder=3)
            axis.add_patch(node)
            axis.text(1.16 * x, 1.16 * y, names[index], ha="center", va="center")
            nodes.append(node)
        maximum = max(np.max(np.abs(values)), 1e-15)
        for i in range(n):
            for j in range(i + 1, n):
                weight = values[i, j] + values[j, i]
                if abs(weight) <= threshold:
                    continue
                start = np.array([np.cos(angles[i]), np.sin(angles[i])])
                end = np.array([np.cos(angles[j]), np.sin(angles[j])])
                path = mpath.Path([start, [0, 0], end], [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])
                patch = mpatches.PathPatch(path, facecolor="none", edgecolor=resolved.colors[i % len(resolved.colors)], linewidth=0.3 + 4.0 * abs(weight) / maximum, alpha=alpha)
                axis.add_patch(patch)
                chords.append(patch)
        axis.set_xlim(-1.3, 1.3)
        axis.set_ylim(-1.3, 1.3)
        return finalize(fig, axis, config=cfg, artists={"nodes": nodes, "chords": chords}, data={"matrix": values, "labels": names})


@register_plot(category="networks", aliases=("alluvial", "sankey"))
def alluvial_plot(
    source: Sequence[str],
    target: Sequence[str],
    value: Any,
    *,
    source_order: Sequence[str] | None = None,
    target_order: Sequence[str] | None = None,
    alpha: float = 0.45,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Two-stage alluvial/Sankey diagram implemented with Matplotlib ribbons."""
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    source_arr = np.asarray(source, dtype=object)
    target_arr = np.asarray(target, dtype=object)
    weights = as_array(value)
    sources = list(source_order or dict.fromkeys(source_arr.tolist()))
    targets = list(target_order or dict.fromkeys(target_arr.tolist()))
    total = max(float(np.sum(weights)), 1e-15)
    source_totals = {name: float(np.sum(weights[source_arr == name])) for name in sources}
    target_totals = {name: float(np.sum(weights[target_arr == name])) for name in targets}
    source_start = {}
    cursor = 0.0
    gap = 0.025
    for name in sources:
        source_start[name] = cursor
        cursor += source_totals[name] / total * (1 - gap * max(len(sources) - 1, 0)) + gap
    target_start = {}
    cursor = 0.0
    for name in targets:
        target_start[name] = cursor
        cursor += target_totals[name] / total * (1 - gap * max(len(targets) - 1, 0)) + gap
    cfg = coerce_config(config, figsize=(7.0, 5.0), equal_aspect=False)
    with theme_context(theme):
        fig, axis, resolved = create_axes(config=cfg, theme=theme)
        rectangles, ribbons = [], []
        source_cursor = source_start.copy()
        target_cursor = target_start.copy()
        scale_s = (1 - gap * max(len(sources) - 1, 0)) / total
        scale_t = (1 - gap * max(len(targets) - 1, 0)) / total
        for index, name in enumerate(sources):
            height = source_totals[name] * scale_s
            rect = mpatches.Rectangle((0.02, source_start[name]), 0.05, height, facecolor=resolved.colors[index % len(resolved.colors)], edgecolor="none")
            axis.add_patch(rect)
            axis.text(0.0, source_start[name] + height / 2, name, ha="right", va="center")
            rectangles.append(rect)
        for index, name in enumerate(targets):
            height = target_totals[name] * scale_t
            rect = mpatches.Rectangle((0.93, target_start[name]), 0.05, height, facecolor=resolved.colors[index % len(resolved.colors)], edgecolor="none")
            axis.add_patch(rect)
            axis.text(1.0, target_start[name] + height / 2, name, ha="left", va="center")
            rectangles.append(rect)
        for s, t, w in zip(source_arr, target_arr, weights):
            h_s, h_t = w * scale_s, w * scale_t
            y0a, y0b = source_cursor[s], source_cursor[s] + h_s
            y1a, y1b = target_cursor[t], target_cursor[t] + h_t
            vertices = [(0.07, y0a), (0.50, y0a), (0.50, y1a), (0.93, y1a), (0.93, y1b), (0.50, y1b), (0.50, y0b), (0.07, y0b), (0.07, y0a)]
            codes = [mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.LINETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CLOSEPOLY]
            patch = mpatches.PathPatch(mpath.Path(vertices, codes), facecolor=resolved.colors[sources.index(s) % len(resolved.colors)], edgecolor="none", alpha=alpha)
            axis.add_patch(patch)
            ribbons.append(patch)
            source_cursor[s] += h_s
            target_cursor[t] += h_t
        axis.set_xlim(-0.1, 1.1)
        axis.set_ylim(0, 1)
        axis.set_axis_off()
        return finalize(fig, axis, config=cfg, artists={"nodes": rectangles, "ribbons": ribbons}, data={"source": source_arr, "target": target_arr, "value": weights})
