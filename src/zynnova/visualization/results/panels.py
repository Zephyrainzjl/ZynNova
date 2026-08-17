from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._core import PlotConfig, PlotResult, coerce_config, create_axes, finalize, iter_axes, theme_context
from ._registry import get_plot, register_plot


@dataclass(slots=True)
class PanelSpec:
    """One panel in a composed figure.

    ``plot`` may be a plotting callable or any registered plot name/alias.
    ``kwargs`` can include a panel-specific ``PlotConfig``.
    """

    plot: Callable[..., PlotResult] | str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    title: str | None = None


@dataclass(slots=True)
class FigureComposer:
    """Compose reusable plot functions into journal-style multi-panel figures."""

    nrows: int
    ncols: int
    figsize: tuple[float, float] | None = None
    theme: Any = "nature"
    sharex: bool = False
    sharey: bool = False
    panel_labels: bool = True
    label_style: str = "alphabetic"
    panel_label_kwargs: dict[str, Any] = field(default_factory=dict)
    width_ratios: Sequence[float] | None = None
    height_ratios: Sequence[float] | None = None
    wspace: float = 0.28
    hspace: float = 0.32
    collect_legend: bool = False
    legend_loc: str = "upper center"
    legend_ncol: int = 3
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, 1.02)
    suptitle: str | None = None

    def compose(self, panels: Sequence[PanelSpec], *, config: PlotConfig | None = None) -> PlotResult:
        if len(panels) > self.nrows * self.ncols:
            raise ValueError("more panels supplied than grid cells")
        cfg = coerce_config(config, figsize=self.figsize, tight_layout=False)
        gridspec_kw: dict[str, Any] = {"wspace": self.wspace, "hspace": self.hspace}
        if self.width_ratios is not None:
            gridspec_kw["width_ratios"] = list(self.width_ratios)
        if self.height_ratios is not None:
            gridspec_kw["height_ratios"] = list(self.height_ratios)
        with theme_context(self.theme):
            fig, axes, _ = create_axes(
                config=cfg,
                theme=self.theme,
                nrows=self.nrows,
                ncols=self.ncols,
                squeeze=False,
                sharex=self.sharex,
                sharey=self.sharey,
                gridspec_kw=gridspec_kw,
            )
            results: list[PlotResult] = []
            for axis, panel in zip(axes.flat, panels):
                kwargs = dict(panel.kwargs)
                kwargs["ax"] = axis
                kwargs.setdefault("theme", self.theme)
                function = get_plot(panel.plot) if isinstance(panel.plot, str) else panel.plot
                result = function(*panel.args, **kwargs)
                if panel.title:
                    axis.set_title(panel.title)
                results.append(result)
            for axis in list(axes.flat)[len(panels):]:
                axis.set_visible(False)
            panel_label_artists = []
            if self.panel_labels:
                explicit = [panel.label for panel in panels]
                labels = explicit if all(label is not None for label in explicit) else None
                panel_label_artists = add_panel_labels(
                    list(axes.flat)[: len(panels)],
                    labels=labels,
                    style=self.label_style,
                    **self.panel_label_kwargs,
                )
            shared_legend_artist = None
            if self.collect_legend:
                shared_legend_artist = shared_legend(
                    fig,
                    list(axes.flat)[: len(panels)],
                    loc=self.legend_loc,
                    ncol=self.legend_ncol,
                    bbox_to_anchor=self.legend_bbox_to_anchor,
                )
                for axis in list(axes.flat)[: len(panels)]:
                    legend = axis.get_legend()
                    if legend is not None:
                        legend.remove()
            if self.suptitle:
                fig.suptitle(self.suptitle)
            if cfg.subplot_adjust:
                fig.subplots_adjust(**cfg.subplot_adjust)
            return finalize(
                fig,
                axes,
                config=cfg,
                artists={
                    "panels": [result.artists for result in results],
                    "panel_labels": panel_label_artists,
                    "shared_legend": shared_legend_artist,
                },
                data={"panel_results": results},
                theme=self.theme,
            )


@register_plot(category="panels", aliases=("compose-figure", "multi-panel-figure"))
def compose_plots(
    panels: Sequence[PanelSpec],
    *,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] | None = None,
    theme: Any = "nature",
    shared_legend_enabled: bool = False,
    config: PlotConfig | None = None,
    **composer_kwargs: Any,
) -> PlotResult:
    """Functional API for composing registered plots into one figure."""
    composer = FigureComposer(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        theme=theme,
        collect_legend=shared_legend_enabled,
        **composer_kwargs,
    )
    return composer.compose(panels, config=config)


def _alphabetic_label(index: int, uppercase: bool = False) -> str:
    # Excel-style sequence: a..z, aa..az, ba...
    chars = []
    number = index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        chars.append(chr((ord("A") if uppercase else ord("a")) + remainder))
    return "".join(reversed(chars))


@register_plot(category="panels", aliases=("panel-labels",))
def add_panel_labels(
    axes: Any,
    *,
    labels: Sequence[str] | None = None,
    style: str = "alphabetic",
    x: float = -0.12,
    y: float = 1.05,
    fontweight: str = "bold",
    fontsize: float | str = "medium",
    bbox: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Add journal-style a, b, c panel labels to one or many axes."""
    axes_list = list(iter_axes(axes))
    if labels is None:
        if style == "alphabetic":
            labels = [_alphabetic_label(index) for index in range(len(axes_list))]
        elif style == "uppercase":
            labels = [_alphabetic_label(index, uppercase=True) for index in range(len(axes_list))]
        else:
            labels = [str(index + 1) for index in range(len(axes_list))]
    if len(labels) != len(axes_list):
        raise ValueError("labels must match number of axes")
    artists = []
    for axis, label in zip(axes_list, labels):
        artists.append(
            axis.text(
                x,
                y,
                str(label),
                transform=axis.transAxes,
                fontweight=fontweight,
                fontsize=fontsize,
                va="top",
                ha="left",
                bbox=None if bbox is None else dict(bbox),
            )
        )
    return artists


@register_plot(category="panels", aliases=("shared-legend", "figure-legend"))
def shared_legend(
    figure: Any,
    axes: Any,
    *,
    loc: str = "upper center",
    ncol: int = 3,
    bbox_to_anchor: tuple[float, float] | None = (0.5, 1.02),
    deduplicate: bool = True,
    **kwargs: Any,
) -> Any:
    """Collect handles from multiple axes into one figure-level legend."""
    handles: list[Any] = []
    labels: list[str] = []
    seen: set[str] = set()
    for axis in iter_axes(axes):
        h, l = axis.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if not label or label.startswith("_"):
                continue
            if deduplicate and label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    if not labels:
        return None
    legend_kwargs = dict(kwargs)
    legend_kwargs.update({"loc": loc, "ncol": ncol})
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    return figure.legend(handles, labels, **legend_kwargs)


@register_plot(category="panels", aliases=("inset-zoom",))
def inset_zoom(
    ax: Any,
    *,
    bounds: tuple[float, float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    loc1: int = 2,
    loc2: int = 4,
    copy_lines: bool = True,
    connector_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Create a reusable zoomed inset and mark the selected region."""
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    inset = ax.inset_axes(bounds)
    if copy_lines:
        for line in ax.lines:
            inset.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                alpha=line.get_alpha(),
            )
        for collection in ax.collections:
            offsets = getattr(collection, "get_offsets", lambda: np.empty((0, 2)))()
            if len(offsets):
                inset.scatter(offsets[:, 0], offsets[:, 1], s=8, alpha=0.5)
    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.tick_params(labelsize="x-small")
    kwargs = {"fc": "none", "ec": "0.4", "lw": 0.7, **dict(connector_kwargs or {})}
    mark_inset(ax, inset, loc1=loc1, loc2=loc2, **kwargs)
    return inset


@register_plot(category="panels", aliases=("broken-axis",))
def broken_axis_plot(
    x: Any,
    y: Any,
    *,
    y_limits: tuple[tuple[float, float], tuple[float, float]],
    marker: str | None = "o",
    line_kwargs: Mapping[str, Any] | None = None,
    config: PlotConfig | None = None,
    theme: Any = None,
) -> PlotResult:
    """Two-panel broken y-axis for separated magnitude ranges."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    cfg = coerce_config(config, figsize=(5.0, 4.5), tight_layout=False)
    with theme_context(theme):
        fig, axes, _ = create_axes(config=cfg, theme=theme, nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05})
        top, bottom = axes
        kwargs = {"marker": marker, **dict(line_kwargs or {})}
        top_line = top.plot(x_arr, y_arr, **kwargs)[0]
        bottom_line = bottom.plot(x_arr, y_arr, **kwargs)[0]
        top.set_ylim(*y_limits[1])
        bottom.set_ylim(*y_limits[0])
        top.spines["bottom"].set_visible(False)
        bottom.spines["top"].set_visible(False)
        top.tick_params(labeltop=False, bottom=False)
        bottom.xaxis.tick_bottom()
        diagonal = 0.012
        diagonal_artists = []
        diagonal_kwargs = dict(color="black", clip_on=False, linewidth=0.8)
        diagonal_artists.extend(top.plot((-diagonal, diagonal), (-diagonal, diagonal), transform=top.transAxes, **diagonal_kwargs))
        diagonal_artists.extend(top.plot((1 - diagonal, 1 + diagonal), (-diagonal, diagonal), transform=top.transAxes, **diagonal_kwargs))
        diagonal_artists.extend(bottom.plot((-diagonal, diagonal), (1 - diagonal, 1 + diagonal), transform=bottom.transAxes, **diagonal_kwargs))
        diagonal_artists.extend(bottom.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), transform=bottom.transAxes, **diagonal_kwargs))
        return finalize(fig, axes, config=cfg, artists={"top": top_line, "bottom": bottom_line, "break_marks": diagonal_artists}, data={"x": x_arr, "y": y_arr, "y_limits": y_limits}, theme=theme)
