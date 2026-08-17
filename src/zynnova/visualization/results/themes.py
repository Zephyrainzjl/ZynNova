from __future__ import annotations

from ._core import PlotTheme


THEMES: dict[str, PlotTheme] = {
    "nature": PlotTheme(name="nature"),
    "nature-dark": PlotTheme(
        name="nature-dark",
        background="#14171a",
        foreground="#f2f2f2",
        grid_color="#50555b",
        colors=("#77AADD", "#EE8866", "#44BB99", "#EEDD88", "#AA99CC", "#BBBBBB"),
    ),
    "science": PlotTheme(
        name="science",
        font_size=8.5,
        title_size=9.5,
        colors=("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"),
        single_column_width=3.42,
        double_column_width=7.18,
    ),
    "cell": PlotTheme(
        name="cell",
        font_size=8.0,
        colors=("#3366AA", "#DC3912", "#109618", "#FF9900", "#990099", "#0099C6", "#DD4477"),
        single_column_width=3.45,
        double_column_width=7.1,
    ),
    "colorblind": PlotTheme(
        name="colorblind",
        colors=("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"),
    ),
    "monochrome": PlotTheme(
        name="monochrome",
        colors=("#111111", "#3A3A3A", "#626262", "#8A8A8A", "#B2B2B2"),
        linestyles=("-", "--", "-.", ":"),
        markers=("o", "s", "^", "D", "v"),
    ),
    "poster": PlotTheme(
        name="poster",
        font_size=13.0,
        title_size=16.0,
        label_size=14.0,
        tick_size=12.0,
        line_width=2.0,
        marker_size=7.0,
        axis_line_width=1.2,
        single_column_width=6.0,
        double_column_width=12.0,
    ),
    "battery": PlotTheme(
        name="battery",
        colors=("#2F6690", "#D95D39", "#3A7D44", "#E5B25D", "#7A5195", "#4C956C", "#8D99AE", "#BC4749"),
        linestyles=("-", "--", "-.", ":"),
    ),
    "phase-field": PlotTheme(
        name="phase-field",
        colors=("#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D597A", "#355070"),
        linestyles=("-", "--", ":", "-."),
    ),
    "thermal": PlotTheme(
        name="thermal",
        colors=("#313695", "#4575B4", "#74ADD1", "#FDAE61", "#F46D43", "#A50026"),
    ),
}


def register_theme(theme: PlotTheme, *, overwrite: bool = False) -> None:
    key = theme.name.strip().lower().replace("_", "-")
    if key in THEMES and not overwrite:
        raise KeyError(f"theme {key!r} already exists")
    THEMES[key] = theme


def available_themes() -> tuple[str, ...]:
    return tuple(sorted(THEMES))
