from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class PlotConfig:
    """Common publication-figure configuration.

    The object is intentionally reusable across every static plotting function.
    Plot-specific arguments still control the scientific representation, while
    this class controls typography, axes, legends and artist appearance.
    """

    figsize: tuple[float, float] | None = None
    dpi: int = 120
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xscale: str = "linear"
    yscale: str = "linear"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    grid: bool = False
    legend: bool = True
    legend_loc: str = "best"
    despine: bool = True
    equal_aspect: bool = False
    tight_layout: bool = True
    constrained_layout: bool = False
    rasterized: bool = False

    # Typography and axis placement.
    title_loc: str = "center"
    title_pad: float | None = None
    title_kwargs: dict[str, Any] = field(default_factory=dict)
    xlabel_kwargs: dict[str, Any] = field(default_factory=dict)
    ylabel_kwargs: dict[str, Any] = field(default_factory=dict)
    tick_params: dict[str, Any] = field(default_factory=dict)
    xtick_rotation: float | None = None
    ytick_rotation: float | None = None
    axes_position: tuple[float, float, float, float] | None = None
    spine_visibility: dict[str, bool] = field(default_factory=dict)
    background: str | None = None

    # Universal artist styling. ``None`` means preserve plot-specific styling.
    palette: tuple[str, ...] | None = None
    line_color: str | None = None
    line_width: float | None = None
    line_style: str | None = None
    marker: str | None = None
    marker_size: float | None = None
    alpha: float | None = None
    artist_styles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Legend customization.
    legend_title: str | None = None
    legend_ncol: int = 1
    legend_bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = None
    legend_frameon: bool | None = None
    legend_fontsize: float | str | None = None
    legend_labels: Mapping[str, str] | Sequence[str] | None = None
    legend_order: Sequence[int] | None = None
    legend_kwargs: dict[str, Any] = field(default_factory=dict)

    # Grid and layout customization.
    grid_kwargs: dict[str, Any] = field(default_factory=dict)
    subplot_adjust: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def merged(self, **updates: Any) -> "PlotConfig":
        clean = {key: value for key, value in updates.items() if value is not None}
        return replace(self, **clean)


@dataclass(slots=True)
class PlotResult:
    """A reusable plotting result returned by ZynNova result plots."""

    figure: Any
    axes: Any
    artists: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fig(self) -> Any:
        return self.figure

    @property
    def ax(self) -> Any:
        axes = np.asarray(self.axes, dtype=object).reshape(-1)
        return axes[0] if axes.size else self.axes

    def save(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        transparent: bool = False,
        bbox_inches: str | None = "tight",
        **kwargs: Any,
    ) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = dict(
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            **kwargs,
        )
        suffix = target.suffix.lower()
        # Matplotlib only accepts arbitrary metadata for selected vector formats.
        if self.metadata and suffix in {".pdf", ".svg", ".eps", ".ps", ".png"}:
            save_kwargs["metadata"] = {str(k): str(v) for k, v in self.metadata.items()}
        self.figure.savefig(target, **save_kwargs)
        return target

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.figure)

    def restyle(self, **kwargs: Any) -> "PlotResult":
        """Update selected artists or axes in-place and return ``self``.

        Only explicitly supplied keys are changed, so a style-only call does
        not reset logarithmic scales, axis limits, titles or layout choices.

        Examples
        --------
        ``result.restyle(line_width=2, line_style="--", legend_loc="upper left")``
        """
        unknown = set(kwargs) - set(PlotConfig.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown PlotConfig fields: {sorted(unknown)}")
        config = PlotConfig(**kwargs)
        _apply_artist_styles(self.artists, config, DEFAULT_THEME)
        for axis in iter_axes(self.axes):
            _apply_partial_axis_style(axis, config, set(kwargs))
        if "subplot_adjust" in kwargs and config.subplot_adjust:
            self.figure.subplots_adjust(**config.subplot_adjust)
        if kwargs.get("tight_layout") and not kwargs.get("constrained_layout", False):
            try:
                self.figure.tight_layout()
            except Exception:
                pass
        return self

    def combine_legend(
        self,
        *,
        ax: Any | None = None,
        axes: Any | None = None,
        **legend_kwargs: Any,
    ) -> Any:
        """Create one de-duplicated legend from several axes."""
        target = ax or self.ax
        source_axes = list(iter_axes(self.axes if axes is None else axes))
        handles: list[Any] = []
        labels: list[str] = []
        seen: set[str] = set()
        for axis in source_axes:
            h, l = axis.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label and not label.startswith("_") and label not in seen:
                    handles.append(handle)
                    labels.append(label)
                    seen.add(label)
        return target.legend(handles, labels, **legend_kwargs) if labels else None


@dataclass(slots=True)
class PlotTheme:
    """Theme with color cycle, typography and journal-size defaults."""

    name: str = "nature"
    font_family: str = "DejaVu Sans"
    font_size: float = 8.0
    title_size: float = 9.0
    label_size: float = 8.0
    tick_size: float = 7.0
    line_width: float = 1.2
    marker_size: float = 4.0
    axis_line_width: float = 0.8
    colors: tuple[str, ...] = (
        "#3B6FB6",
        "#D1495B",
        "#2A9D8F",
        "#F4A261",
        "#7E57C2",
        "#6C757D",
        "#E9C46A",
        "#264653",
    )
    linestyles: tuple[str, ...] = ("-", "--", "-.", ":")
    markers: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
    single_column_width: float = 3.35
    double_column_width: float = 7.0
    background: str = "white"
    foreground: str = "#1a1a1a"
    grid_color: str = "#d7d7d7"
    colorblind_safe: bool = True
    extra_rc: dict[str, Any] = field(default_factory=dict)

    def rc_params(self) -> dict[str, Any]:
        from cycler import cycler

        values: dict[str, Any] = {
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "axes.linewidth": self.axis_line_width,
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            "axes.prop_cycle": cycler(color=self.colors),
            "figure.facecolor": self.background,
            "axes.facecolor": self.background,
            "savefig.facecolor": self.background,
            "text.color": self.foreground,
            "axes.labelcolor": self.foreground,
            "axes.edgecolor": self.foreground,
            "xtick.color": self.foreground,
            "ytick.color": self.foreground,
            "grid.color": self.grid_color,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
        values.update(self.extra_rc)
        return values

    @contextmanager
    def context(self) -> Iterator[None]:
        import matplotlib as mpl

        with mpl.rc_context(self.rc_params()):
            yield


DEFAULT_THEME = PlotTheme()


def coerce_config(config: PlotConfig | None, **kwargs: Any) -> PlotConfig:
    base = config or PlotConfig()
    valid = PlotConfig.__dataclass_fields__
    updates = {key: value for key, value in kwargs.items() if key in valid and value is not None}
    return base.merged(**updates)


def get_theme(theme: PlotTheme | str | None) -> PlotTheme:
    if theme is None:
        return DEFAULT_THEME
    if isinstance(theme, PlotTheme):
        return theme
    from .themes import THEMES

    key = str(theme).strip().lower().replace("_", "-")
    if key not in THEMES:
        raise KeyError(f"unknown plot theme {theme!r}; available={sorted(THEMES)}")
    return THEMES[key]


@contextmanager
def theme_context(theme: PlotTheme | str | None = None) -> Iterator[PlotTheme]:
    resolved = get_theme(theme)
    with resolved.context():
        yield resolved


def create_axes(
    *,
    ax: Any = None,
    config: PlotConfig | None = None,
    theme: PlotTheme | str | None = None,
    projection: str | None = None,
    nrows: int = 1,
    ncols: int = 1,
    squeeze: bool = True,
    sharex: bool = False,
    sharey: bool = False,
    gridspec_kw: Mapping[str, Any] | None = None,
    subplot_kw: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, PlotTheme]:
    import matplotlib.pyplot as plt

    cfg = config or PlotConfig()
    resolved = get_theme(theme)
    if ax is not None:
        first = ax if hasattr(ax, "figure") else np.asarray(ax, dtype=object).flat[0]
        return first.figure, ax, resolved
    subplot_options = dict(subplot_kw or {})
    if projection is not None:
        subplot_options["projection"] = projection
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=cfg.figsize,
        dpi=cfg.dpi,
        squeeze=squeeze,
        sharex=sharex,
        sharey=sharey,
        subplot_kw=subplot_options or None,
        gridspec_kw=dict(gridspec_kw or {}),
        constrained_layout=cfg.constrained_layout,
    )
    return figure, axes, resolved


def iter_axes(axes: Any) -> Iterable[Any]:
    if isinstance(axes, np.ndarray):
        yield from axes.flat
    elif isinstance(axes, (list, tuple)):
        for item in axes:
            yield from iter_axes(item)
    else:
        yield axes


def _set_artist_style(artist: Any, style: Mapping[str, Any]) -> None:
    if artist is None:
        return
    if isinstance(artist, Mapping):
        for value in artist.values():
            _set_artist_style(value, style)
        return
    if isinstance(artist, (list, tuple, np.ndarray)):
        for value in artist:
            _set_artist_style(value, style)
        return
    # Matplotlib containers are iterable but some artists also implement
    # iteration; prefer ``set`` first and recurse only on failure.
    try:
        artist.set(**dict(style))
        return
    except Exception:
        pass
    try:
        for value in artist:
            _set_artist_style(value, style)
    except Exception:
        return


def _flatten_artists(artist: Any) -> list[Any]:
    """Return drawable leaf artists from nested mappings/sequences."""
    if artist is None:
        return []
    if isinstance(artist, Mapping):
        leaves: list[Any] = []
        for value in artist.values():
            leaves.extend(_flatten_artists(value))
        return leaves
    if isinstance(artist, (str, bytes)):
        return []
    if hasattr(artist, "set") or hasattr(artist, "set_color"):
        return [artist]
    try:
        leaves = []
        for value in artist:
            leaves.extend(_flatten_artists(value))
        return leaves
    except Exception:
        return []


def _apply_artist_styles(artists: Mapping[str, Any], config: PlotConfig, theme: PlotTheme) -> None:
    generic: dict[str, Any] = {}
    if config.line_color is not None:
        generic["color"] = config.line_color
    if config.line_width is not None:
        generic["linewidth"] = config.line_width
    if config.line_style is not None:
        generic["linestyle"] = config.line_style
    if config.marker is not None:
        generic["marker"] = config.marker
    if config.marker_size is not None:
        generic["markersize"] = config.marker_size
    if config.alpha is not None:
        generic["alpha"] = config.alpha
    if generic:
        for artist in artists.values():
            _set_artist_style(artist, generic)

    palette = config.palette
    if palette:
        # Cycle colors over individual leaf artists rather than top-level
        # containers, so a list of lines receives distinct reusable colors.
        leaves: list[Any] = []
        for artist in artists.values():
            leaves.extend(_flatten_artists(artist))
        for index, artist in enumerate(leaves):
            _set_artist_style(artist, {"color": palette[index % len(palette)]})

    for key, style in config.artist_styles.items():
        if key in artists:
            _set_artist_style(artists[key], style)


def _legend_labels(handles: Sequence[Any], labels: Sequence[str], config: PlotConfig) -> tuple[list[Any], list[str]]:
    h = list(handles)
    l = list(labels)
    if isinstance(config.legend_labels, Mapping):
        l = [str(config.legend_labels.get(label, label)) for label in l]
    elif config.legend_labels is not None:
        replacement = [str(value) for value in config.legend_labels]
        l[: len(replacement)] = replacement[: len(l)]
    if config.legend_order is not None:
        order = [int(i) for i in config.legend_order if 0 <= int(i) < len(h)]
        h = [h[i] for i in order]
        l = [l[i] for i in order]
    return h, l


def apply_axis_config(ax: Any, config: PlotConfig) -> None:
    if config.title:
        kwargs = dict(config.title_kwargs)
        kwargs.setdefault("loc", config.title_loc)
        if config.title_pad is not None:
            kwargs.setdefault("pad", config.title_pad)
        ax.set_title(config.title, **kwargs)
    if config.xlabel:
        ax.set_xlabel(config.xlabel, **config.xlabel_kwargs)
    if config.ylabel:
        ax.set_ylabel(config.ylabel, **config.ylabel_kwargs)
    if config.xscale:
        ax.set_xscale(config.xscale)
    if config.yscale:
        ax.set_yscale(config.yscale)
    if config.xlim is not None:
        ax.set_xlim(*config.xlim)
    if config.ylim is not None:
        ax.set_ylim(*config.ylim)
    if config.grid:
        ax.grid(True, **config.grid_kwargs)
    if config.equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if config.background is not None:
        ax.set_facecolor(config.background)
    if config.axes_position is not None:
        ax.set_position(config.axes_position)
    if config.tick_params:
        ax.tick_params(**config.tick_params)
    if config.xtick_rotation is not None:
        ax.tick_params(axis="x", labelrotation=config.xtick_rotation)
    if config.ytick_rotation is not None:
        ax.tick_params(axis="y", labelrotation=config.ytick_rotation)
    if hasattr(ax, "spines"):
        visibility = {"top": not config.despine, "right": not config.despine}
        visibility.update(config.spine_visibility)
        for side, visible in visibility.items():
            if side in ax.spines:
                ax.spines[side].set_visible(bool(visible))


def _apply_legend(axis: Any, config: PlotConfig) -> Any:
    if not config.legend or not hasattr(axis, "get_legend_handles_labels"):
        existing = axis.get_legend() if hasattr(axis, "get_legend") else None
        if existing is not None and not config.legend:
            existing.remove()
        return None
    handles, labels = axis.get_legend_handles_labels()
    pairs = [(h, l) for h, l in zip(handles, labels) if l and not str(l).startswith("_")]
    if not pairs:
        return None
    # De-duplicate labels while preserving order.
    unique_h: list[Any] = []
    unique_l: list[str] = []
    seen: set[str] = set()
    for handle, label in pairs:
        label = str(label)
        if label not in seen:
            unique_h.append(handle)
            unique_l.append(label)
            seen.add(label)
    unique_h, unique_l = _legend_labels(unique_h, unique_l, config)
    kwargs = dict(config.legend_kwargs)
    kwargs.setdefault("loc", config.legend_loc)
    kwargs.setdefault("ncol", config.legend_ncol)
    if config.legend_title is not None:
        kwargs.setdefault("title", config.legend_title)
    if config.legend_bbox_to_anchor is not None:
        kwargs.setdefault("bbox_to_anchor", config.legend_bbox_to_anchor)
    if config.legend_frameon is not None:
        kwargs.setdefault("frameon", config.legend_frameon)
    if config.legend_fontsize is not None:
        kwargs.setdefault("fontsize", config.legend_fontsize)
    return axis.legend(unique_h, unique_l, **kwargs)


def _apply_partial_axis_style(axis: Any, config: PlotConfig, supplied: set[str]) -> None:
    """Apply only explicitly supplied PlotConfig fields to an existing axis."""
    if "title" in supplied:
        kwargs = dict(config.title_kwargs)
        kwargs.setdefault("loc", config.title_loc)
        if config.title_pad is not None:
            kwargs.setdefault("pad", config.title_pad)
        axis.set_title(config.title or "", **kwargs)
    if "xlabel" in supplied:
        axis.set_xlabel(config.xlabel or "", **config.xlabel_kwargs)
    if "ylabel" in supplied:
        axis.set_ylabel(config.ylabel or "", **config.ylabel_kwargs)
    if "xscale" in supplied:
        axis.set_xscale(config.xscale)
    if "yscale" in supplied:
        axis.set_yscale(config.yscale)
    if "xlim" in supplied and config.xlim is not None:
        axis.set_xlim(*config.xlim)
    if "ylim" in supplied and config.ylim is not None:
        axis.set_ylim(*config.ylim)
    if "grid" in supplied:
        axis.grid(config.grid, **config.grid_kwargs)
    if "equal_aspect" in supplied:
        axis.set_aspect("equal" if config.equal_aspect else "auto", adjustable="box")
    if "background" in supplied:
        axis.set_facecolor(config.background or "none")
    if "axes_position" in supplied and config.axes_position is not None:
        axis.set_position(config.axes_position)
    if "tick_params" in supplied and config.tick_params:
        axis.tick_params(**config.tick_params)
    if "xtick_rotation" in supplied and config.xtick_rotation is not None:
        axis.tick_params(axis="x", labelrotation=config.xtick_rotation)
    if "ytick_rotation" in supplied and config.ytick_rotation is not None:
        axis.tick_params(axis="y", labelrotation=config.ytick_rotation)
    if hasattr(axis, "spines") and ({"despine", "spine_visibility"} & supplied):
        visibility = {"top": not config.despine, "right": not config.despine}
        visibility.update(config.spine_visibility)
        for side, visible in visibility.items():
            if side in axis.spines:
                axis.spines[side].set_visible(bool(visible))
    legend_fields = {
        "legend", "legend_loc", "legend_title", "legend_ncol",
        "legend_bbox_to_anchor", "legend_frameon", "legend_fontsize",
        "legend_labels", "legend_order", "legend_kwargs",
    }
    if supplied & legend_fields:
        _apply_legend(axis, config)


def _apply_result_style(result: PlotResult, config: PlotConfig) -> None:
    theme = DEFAULT_THEME
    _apply_artist_styles(result.artists, config, theme)
    for axis in iter_axes(result.axes):
        apply_axis_config(axis, config)
        _apply_legend(axis, config)
    if config.subplot_adjust:
        result.figure.subplots_adjust(**config.subplot_adjust)
    if config.tight_layout and not config.constrained_layout:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*", category=UserWarning)
                result.figure.tight_layout()
        except Exception:
            pass


def finalize(
    figure: Any,
    axes: Any,
    *,
    config: PlotConfig | None = None,
    artists: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    theme: PlotTheme | str | None = None,
) -> PlotResult:
    cfg = config or PlotConfig()
    artists_dict = dict(artists or {})
    resolved = get_theme(theme)
    _apply_artist_styles(artists_dict, cfg, resolved)
    for axis in iter_axes(axes):
        apply_axis_config(axis, cfg)
        _apply_legend(axis, cfg)
    if cfg.subplot_adjust:
        figure.subplots_adjust(**cfg.subplot_adjust)
    if cfg.tight_layout and not cfg.constrained_layout:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*", category=UserWarning)
                figure.tight_layout()
        except Exception:
            pass
    return PlotResult(
        figure=figure,
        axes=axes,
        artists=artists_dict,
        data=dict(data or {}),
        metrics=dict(metrics or {}),
        metadata={**cfg.metadata, **dict(metadata or {})},
    )


def journal_size(
    columns: int = 1,
    *,
    aspect: float = 0.72,
    theme: PlotTheme | str | None = None,
) -> tuple[float, float]:
    resolved = get_theme(theme)
    if columns not in {1, 2}:
        raise ValueError("columns must be 1 or 2")
    width = resolved.single_column_width if columns == 1 else resolved.double_column_width
    return width, width * float(aspect)
