from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array, density_values, make_labels


@register_plot(category="embeddings", aliases=("latent-scatter", "umap-plot", "tsne-plot"))
def embedding_scatter(
    embedding: Any,
    *,
    color: Any | None = None,
    labels: Any | None = None,
    label_names: Mapping[Any, str] | None = None,
    size: Any | float = 10.0,
    alpha: float = 0.7,
    cmap: str = "viridis",
    density_background: bool = False,
    annotate_centroids: bool = False,
    rasterized: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """General UMAP/t-SNE/PCA/latent-space scatter for materials and biology."""
    coordinates = np.asarray(embedding, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("embedding must have shape (n_samples, >=2)")
    x, y = coordinates[:, 0], coordinates[:, 1]
    cfg = coerce_config(config, xlabel="Embedding 1", ylabel="Embedding 2", rasterized=rasterized)
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        artists: dict[str, Any] = {}
        if density_background:
            density = density_values(x, y)
            order = np.argsort(density)
            artists["points"] = axis.scatter(x[order], y[order], c=density[order], s=size, cmap=cmap, alpha=alpha, edgecolors="none", rasterized=rasterized)
            artists["colorbar"] = fig.colorbar(artists["points"], ax=axis, label="Density")
        elif labels is not None:
            label_arr = np.asarray(labels).reshape(-1)
            scatters = []
            for index, label in enumerate(dict.fromkeys(label_arr.tolist())):
                selected = label_arr == label
                scatter = axis.scatter(x[selected], y[selected], s=np.asarray(size)[selected] if np.ndim(size) else size, alpha=alpha, label=(label_names or {}).get(label, str(label)), color=resolved.colors[index % len(resolved.colors)], edgecolors="none", rasterized=rasterized)
                scatters.append(scatter)
                if annotate_centroids:
                    axis.text(np.mean(x[selected]), np.mean(y[selected]), (label_names or {}).get(label, str(label)), ha="center", va="center", fontweight="bold")
            artists["points"] = scatters
        else:
            artists["points"] = axis.scatter(x, y, c=color, s=size, alpha=alpha, cmap=cmap, edgecolors="none", rasterized=rasterized)
            if color is not None and np.issubdtype(np.asarray(color).dtype, np.number):
                artists["colorbar"] = fig.colorbar(artists["points"], ax=axis, label="Value")
        return finalize(fig, axis, config=cfg, artists=artists, data={"embedding": coordinates, "color": color, "labels": labels})


@register_plot(category="embeddings", aliases=("embedding-density", "latent-density"))
def embedding_density(
    embedding: Any,
    *,
    bins: int = 100,
    log_density: bool = True,
    cmap: str = "magma",
    contours: int = 8,
    show_points: bool = False,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Hexbin or smooth density map for very large embeddings."""
    coordinates = np.asarray(embedding, dtype=float)
    x, y = coordinates[:, 0], coordinates[:, 1]
    cfg = coerce_config(config, xlabel="Embedding 1", ylabel="Embedding 2")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        hexbin = axis.hexbin(x, y, gridsize=bins, mincnt=1, bins="log" if log_density else None, cmap=cmap)
        colorbar = fig.colorbar(hexbin, ax=axis, label="log count" if log_density else "Count")
        contour_artist = None
        if contours:
            try:
                from scipy.stats import gaussian_kde

                xi = np.linspace(x.min(), x.max(), 120)
                yi = np.linspace(y.min(), y.max(), 120)
                xx, yy = np.meshgrid(xi, yi)
                density = gaussian_kde(np.vstack([x, y]))(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                contour_artist = axis.contour(xx, yy, density, levels=contours, colors="white", linewidths=0.5, alpha=0.6)
            except Exception:
                pass
        points = axis.scatter(x, y, s=2, color="white", alpha=0.1, rasterized=True) if show_points else None
        return finalize(fig, axis, config=cfg, artists={"hexbin": hexbin, "colorbar": colorbar, "contours": contour_artist, "points": points}, data={"embedding": coordinates})


@register_plot(category="embeddings", aliases=("pseudotime-trajectory", "latent-trajectory"))
def trajectory_embedding(
    embedding: Any,
    *,
    time: Any | None = None,
    trajectories: Any | None = None,
    labels: Any | None = None,
    arrows: bool = True,
    arrow_every: int = 10,
    cmap: str = "viridis",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Latent trajectory or pseudotime path with directional arrows."""
    coordinates=np.asarray(embedding,dtype=float)
    if coordinates.ndim!=2 or coordinates.shape[1]<2: raise ValueError("embedding must contain at least two coordinates")
    x,y=coordinates[:,0],coordinates[:,1]; scalar=np.arange(len(x)) if time is None else as_array(time)
    if scalar.size!=len(x): raise ValueError("time must match embedding")
    label_arr=None if labels is None else np.asarray(labels).reshape(-1)
    if label_arr is not None and label_arr.size!=len(x): raise ValueError("labels must match embedding")
    cfg=coerce_config(config,xlabel="Embedding 1",ylabel="Embedding 2")
    with theme_context(theme):
        fig,axis,_=create_axes(ax=ax,config=cfg,theme=theme)
        points=axis.scatter(x,y,c=scalar,cmap=cmap,s=12,alpha=0.75,rasterized=cfg.rasterized)
        colorbar=fig.colorbar(points,ax=axis,label="Time" if time is not None else "Order")
        lines=[]; arrow_artists=[]; annotations=[]
        if trajectories is None: paths=[np.arange(coordinates.shape[0])]
        else:
            trajectory_arr=np.asarray(trajectories).reshape(-1)
            if trajectory_arr.size!=len(x): raise ValueError("trajectories must match embedding")
            paths=[np.flatnonzero(trajectory_arr==item) for item in dict.fromkeys(trajectory_arr.tolist())]
        for path in paths:
            if path.size<2: continue
            line=axis.plot(x[path],y[path],color="black",alpha=0.5,linewidth=0.8)[0]; lines.append(line)
            if arrows:
                for offset in range(0,path.size-1,max(1,arrow_every)):
                    start,end=path[offset],path[min(offset+1,path.size-1)]
                    arrow_artists.append(axis.annotate("",xy=(x[end],y[end]),xytext=(x[start],y[start]),arrowprops={"arrowstyle":"->","color":"black","lw":0.8,"alpha":0.7}))
            if label_arr is not None:
                for index in (path[0],path[-1]): annotations.append(axis.annotate(str(label_arr[index]),(x[index],y[index]),xytext=(3,3),textcoords="offset points",fontsize="x-small"))
        return finalize(fig,axis,config=cfg,artists={"points":points,"colorbar":colorbar,"paths":lines,"arrows":arrow_artists,"labels":annotations},data={"embedding":coordinates,"time":scalar,"trajectories":trajectories,"labels":label_arr},theme=theme)


@register_plot(category="embeddings", aliases=("latent-property-map",))
def latent_property_surface(
    embedding: Any,
    property_values: Any,
    *,
    resolution: int = 150,
    method: str = "linear",
    cmap: str = "viridis",
    mask_distance: float | None = None,
    show_points: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Interpolate a property over a learned latent manifold."""
    coordinates = np.asarray(embedding, dtype=float)
    values = as_array(property_values)
    x, y = coordinates[:, 0], coordinates[:, 1]
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xx, yy = np.meshgrid(xi, yi)
    try:
        from scipy.interpolate import griddata

        zz = griddata((x, y), values, (xx, yy), method=method)
    except Exception:
        zz = np.full_like(xx, np.nan)
        for i in range(resolution):
            for j in range(resolution):
                nearest = np.argmin((x - xx[i, j]) ** 2 + (y - yy[i, j]) ** 2)
                zz[i, j] = values[nearest]
    if mask_distance is not None:
        try:
            from scipy.spatial import cKDTree

            distance, _ = cKDTree(coordinates[:, :2]).query(np.c_[xx.ravel(), yy.ravel()])
            zz.ravel()[distance > mask_distance] = np.nan
        except Exception:
            pass
    cfg = coerce_config(config, xlabel="Embedding 1", ylabel="Embedding 2")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        image = axis.contourf(xx, yy, zz, levels=30, cmap=cmap)
        points = axis.scatter(x, y, c=values, cmap=cmap, s=8, edgecolors="none", rasterized=True) if show_points else None
        colorbar = fig.colorbar(image, ax=axis, label="Property")
        return finalize(fig, axis, config=cfg, artists={"surface": image, "points": points, "colorbar": colorbar}, data={"x_grid": xx, "y_grid": yy, "property_grid": zz, "embedding": coordinates, "property": values})


@register_plot(category="embeddings", aliases=("cluster-hulls",))
def cluster_hulls(
    embedding: Any,
    labels: Any,
    *,
    alpha: float = 0.15,
    point_size: float = 8.0,
    annotate: bool = True,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Cluster scatter with convex hulls for phases, cell types or regimes."""
    coordinates = np.asarray(embedding, dtype=float)
    label_arr = np.asarray(labels).reshape(-1)
    cfg = coerce_config(config, xlabel="Embedding 1", ylabel="Embedding 2")
    with theme_context(theme):
        fig, axis, resolved = create_axes(ax=ax, config=cfg, theme=theme)
        scatters, polygons = [], []
        for index, label in enumerate(dict.fromkeys(label_arr.tolist())):
            selected = label_arr == label
            points = coordinates[selected, :2]
            color = resolved.colors[index % len(resolved.colors)]
            scatters.append(axis.scatter(points[:, 0], points[:, 1], s=point_size, color=color, label=str(label), alpha=0.65, edgecolors="none", rasterized=True))
            if points.shape[0] >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    from matplotlib.patches import Polygon

                    hull = ConvexHull(points)
                    patch = Polygon(points[hull.vertices], closed=True, facecolor=color, edgecolor=color, alpha=alpha)
                    axis.add_patch(patch)
                    polygons.append(patch)
                except Exception:
                    pass
            if annotate:
                axis.text(np.mean(points[:, 0]), np.mean(points[:, 1]), str(label), ha="center", va="center", fontweight="bold")
        return finalize(fig, axis, config=cfg, artists={"points": scatters, "hulls": polygons}, data={"embedding": coordinates, "labels": label_arr})


@register_plot(category="embeddings", aliases=("embedding-stability",))
def embedding_stability_plot(
    reference_embedding: Any,
    comparison_embeddings: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    neighbors: Sequence[int] = (5, 10, 20, 50),
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Neighborhood-overlap profile for comparing latent projections."""
    reference = np.asarray(reference_embedding, dtype=float)
    comparisons = [np.asarray(item, dtype=float) for item in comparison_embeddings]
    names = make_labels(labels, len(comparisons), "Embedding")
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        raise ImportError("embedding_stability_plot requires scikit-learn") from exc
    scores = np.zeros((len(comparisons), len(neighbors)), dtype=float)
    for j, k in enumerate(neighbors):
        ref_neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(reference))).fit(reference).kneighbors(return_distance=False)[:, 1:]
        for i, comparison in enumerate(comparisons):
            comp_neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(comparison))).fit(comparison).kneighbors(return_distance=False)[:, 1:]
            scores[i, j] = np.mean([len(set(a).intersection(b)) / max(k, 1) for a, b in zip(ref_neighbors, comp_neighbors)])
    cfg = coerce_config(config, xlabel="Number of neighbors", ylabel="Neighborhood overlap", ylim=(0, 1))
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        artists = [axis.plot(neighbors, row, marker="o", label=label)[0] for row, label in zip(scores, names)]
        return finalize(fig, axis, config=cfg, artists={"profiles": artists}, data={"neighbors": np.asarray(neighbors), "scores": scores, "labels": names})
