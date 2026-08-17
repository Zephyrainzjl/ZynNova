from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, theme_context
from ._registry import register_plot
from ._utils import as_array


@register_plot(category="fields", aliases=("scalar-map", "heatmap-field"))
def scalar_field_plot(
    field: Any,
    *,
    x: Any | None = None,
    y: Any | None = None,
    mask: Any | None = None,
    contours: int | Sequence[float] = 0,
    cmap: str = "viridis",
    colorbar_label: str = "Value",
    origin: str = "lower",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Scalar simulation field with optional contours and physical coordinates."""
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("scalar_field_plot expects a 2-D array")
    if mask is not None:
        values = np.ma.array(values, mask=np.asarray(mask, dtype=bool))
    cfg = coerce_config(config, equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        if x is None or y is None:
            image = axis.imshow(values, origin=origin, cmap=cmap, aspect="auto")
            xx, yy = np.meshgrid(np.arange(values.shape[1]), np.arange(values.shape[0]))
        else:
            x_arr, y_arr = np.asarray(x), np.asarray(y)
            xx, yy = np.meshgrid(x_arr, y_arr) if x_arr.ndim == y_arr.ndim == 1 else (x_arr, y_arr)
            image = axis.pcolormesh(xx, yy, values, cmap=cmap, shading="auto")
        contour = axis.contour(xx, yy, values, levels=contours, colors="black", linewidths=0.4) if contours else None
        colorbar = fig.colorbar(image, ax=axis, label=colorbar_label)
        return finalize(fig, axis, config=cfg, artists={"field": image, "contours": contour, "colorbar": colorbar}, data={"field": values, "x": xx, "y": yy})


@register_plot(category="fields", aliases=("quiver", "vector-map"))
def vector_field_plot(
    u: Any,
    v: Any,
    *,
    x: Any | None = None,
    y: Any | None = None,
    magnitude_background: bool = True,
    stride: int = 1,
    normalize: bool = False,
    cmap: str = "viridis",
    scale: float | None = None,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """2-D vector field with magnitude backdrop and quiver arrows."""
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if u_arr.shape != v_arr.shape or u_arr.ndim != 2:
        raise ValueError("u and v must be equal-shape 2-D arrays")
    xx, yy = np.meshgrid(np.arange(u_arr.shape[1]), np.arange(u_arr.shape[0])) if x is None or y is None else (np.meshgrid(np.asarray(x), np.asarray(y)) if np.asarray(x).ndim == np.asarray(y).ndim == 1 else (np.asarray(x), np.asarray(y)))
    magnitude = np.sqrt(u_arr**2 + v_arr**2)
    u_plot, v_plot = u_arr.copy(), v_arr.copy()
    if normalize:
        u_plot /= np.maximum(magnitude, 1e-15)
        v_plot /= np.maximum(magnitude, 1e-15)
    cfg = coerce_config(config, equal_aspect=True)
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        background = axis.pcolormesh(xx, yy, magnitude, cmap=cmap, shading="auto") if magnitude_background else None
        quiver = axis.quiver(xx[::stride, ::stride], yy[::stride, ::stride], u_plot[::stride, ::stride], v_plot[::stride, ::stride], scale=scale, color="white" if magnitude_background else None)
        colorbar = fig.colorbar(background, ax=axis, label="Magnitude") if background is not None else None
        return finalize(fig, axis, config=cfg, artists={"background": background, "vectors": quiver, "colorbar": colorbar}, data={"u": u_arr, "v": v_arr, "magnitude": magnitude})


@register_plot(category="fields", aliases=("orthogonal-slices", "volume-slices"))
def orthogonal_slices_plot(
    volume: Any,
    *,
    indices: tuple[int, int, int] | None = None,
    cmap: str = "viridis",
    shared_colorbar: bool = True,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Three orthogonal slices through a 3-D scalar volume."""
    data = np.asarray(volume, dtype=float)
    if data.ndim != 3:
        raise ValueError("volume must be three-dimensional")
    z, y, x = indices or tuple(size // 2 for size in data.shape)
    slices = [data[z, :, :], data[:, y, :], data[:, :, x]]
    titles = [f"z={z}", f"y={y}", f"x={x}"]
    cfg = coerce_config(config, figsize=(9.0, 3.2))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=1, ncols=3)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        images = []
        for axis, image_data, title in zip(axes, slices, titles):
            image = axis.imshow(image_data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_axis_off()
            images.append(image)
        colorbar = fig.colorbar(images[-1], ax=list(axes), shrink=0.8, label="Value") if shared_colorbar else None
        return finalize(fig, axes, config=cfg, artists={"slices": images, "colorbar": colorbar}, data={"volume": data, "indices": (z, y, x)})


@register_plot(category="fields", aliases=("phase-montage", "field-montage"))
def phase_field_montage(
    frames: Any,
    *,
    times: Any | None = None,
    frame_indices: Sequence[int] | None = None,
    columns: int = 4,
    cmap: str = "coolwarm",
    shared_colorbar: bool = True,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Snapshot montage for phase-field, damage or concentration evolution."""
    array = np.asarray(frames, dtype=float)
    if array.ndim != 3:
        raise ValueError("frames must have shape (time, y, x)")
    indices = np.linspace(0, len(array) - 1, min(8, len(array)), dtype=int) if frame_indices is None else np.asarray(frame_indices, dtype=int)
    rows = int(np.ceil(len(indices) / columns))
    cfg = coerce_config(config, figsize=(3.0 * columns, 2.7 * rows))
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=rows, ncols=columns, squeeze=False)
        images = []
        vmin, vmax = np.nanmin(array[indices]), np.nanmax(array[indices])
        time_arr = None if times is None else as_array(times)
        for axis, index in zip(axes.flat, indices):
            image = axis.imshow(array[index], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            title = f"Frame {index}" if time_arr is None else f"t={time_arr[index]:.3g}"
            axis.set_title(title)
            axis.set_axis_off()
            images.append(image)
        for axis in list(axes.flat)[len(indices):]:
            axis.set_visible(False)
        colorbar = fig.colorbar(images[-1], ax=list(axes.flat), shrink=0.75, label="Field") if shared_colorbar and images else None
        return finalize(fig, axes, config=cfg, artists={"images": images, "colorbar": colorbar}, data={"frames": array, "indices": indices})


@register_plot(category="fields", aliases=("kymograph", "space-time-map"))
def kymograph_plot(
    space: Any,
    time: Any,
    values: Any,
    *,
    cmap: str = "viridis",
    colorbar_label: str = "Value",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Space–time kymograph for fronts, waves and concentration profiles."""
    x = as_array(space)
    t = as_array(time)
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (t.size, x.size):
        raise ValueError("values must have shape (time, space)")
    cfg = coerce_config(config, xlabel="Space", ylabel="Time")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        mesh = axis.pcolormesh(x, t, matrix, cmap=cmap, shading="auto")
        colorbar = fig.colorbar(mesh, ax=axis, label=colorbar_label)
        return finalize(fig, axis, config=cfg, artists={"kymograph": mesh, "colorbar": colorbar}, data={"space": x, "time": t, "values": matrix})


@register_plot(category="fields", aliases=("mesh-quality",))
def mesh_quality_plot(
    quality: Any,
    *,
    threshold: float | None = None,
    metric_name: str = "Element quality",
    bins: int = 40,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Finite-element mesh-quality distribution with failure threshold."""
    values = as_array(quality)
    cfg = coerce_config(config, xlabel=metric_name, ylabel="Count")
    with theme_context(theme):
        fig, axis, _ = create_axes(ax=ax, config=cfg, theme=theme)
        histogram = axis.hist(values, bins=bins, alpha=0.7)
        threshold_line = axis.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:g}") if threshold is not None else None
        failure_fraction = float(np.mean(values < threshold)) if threshold is not None else float("nan")
        axis.text(0.96, 0.96, f"min={np.min(values):.3g}\nmedian={np.median(values):.3g}\nfail={failure_fraction:.2%}" if threshold is not None else f"min={np.min(values):.3g}\nmedian={np.median(values):.3g}", transform=axis.transAxes, va="top", ha="right")
        return finalize(fig, axis, config=cfg, artists={"histogram": histogram, "threshold": threshold_line}, data={"quality": values}, metrics={"minimum": float(np.min(values)), "median": float(np.median(values)), "failure_fraction": failure_fraction})


@register_plot(category="fields", aliases=("crack-path", "damage-map"))
def crack_path_plot(
    damage: Any,
    *,
    displacement: Any | None = None,
    threshold: float = 0.8,
    cmap: str = "inferno",
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Phase-field fracture damage map, crack path and displacement vectors."""
    field=np.asarray(damage,dtype=float)
    if field.ndim!=2: raise ValueError("damage must be 2-D")
    crack=field>=threshold; disp=None if displacement is None else np.asarray(displacement,dtype=float)
    if disp is not None:
        if disp.shape==(2,)+field.shape: disp=np.moveaxis(disp,0,-1)
        if disp.shape!=field.shape+(2,): raise ValueError("displacement must have shape (ny, nx, 2) or (2, ny, nx)")
    cfg=coerce_config(config,equal_aspect=True)
    with theme_context(theme):
        fig,axis,_=create_axes(ax=ax,config=cfg,theme=theme)
        image=axis.imshow(field,origin="lower",cmap=cmap,vmin=0,vmax=1); contour=axis.contour(crack.astype(float),levels=[0.5],colors="cyan",linewidths=1.0)
        quiver=None
        if disp is not None:
            step=max(1,min(field.shape)//24); yy,xx=np.mgrid[0:field.shape[0]:step,0:field.shape[1]:step]
            quiver=axis.quiver(xx,yy,disp[::step,::step,0],disp[::step,::step,1],color="white",alpha=0.65,scale=None)
        colorbar=fig.colorbar(image,ax=axis,label="Damage")
        return finalize(fig,axis,config=cfg,artists={"damage":image,"crack":contour,"displacement":quiver,"colorbar":colorbar},data={"damage":field,"crack_mask":crack,"displacement":disp},metrics={"crack_fraction":float(np.mean(crack))},theme=theme)
