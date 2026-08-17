from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR

from ..scene.schema import ColorRef


_THEME_MAP = {
    "dk1": MSO_THEME_COLOR.DARK_1,
    "lt1": MSO_THEME_COLOR.LIGHT_1,
    "dk2": MSO_THEME_COLOR.DARK_2,
    "lt2": MSO_THEME_COLOR.LIGHT_2,
    "tx1": MSO_THEME_COLOR.TEXT_1,
    "tx2": MSO_THEME_COLOR.TEXT_2,
    "bg1": MSO_THEME_COLOR.BACKGROUND_1,
    "bg2": MSO_THEME_COLOR.BACKGROUND_2,
    "accent1": MSO_THEME_COLOR.ACCENT_1,
    "accent2": MSO_THEME_COLOR.ACCENT_2,
    "accent3": MSO_THEME_COLOR.ACCENT_3,
    "accent4": MSO_THEME_COLOR.ACCENT_4,
    "accent5": MSO_THEME_COLOR.ACCENT_5,
    "accent6": MSO_THEME_COLOR.ACCENT_6,
    "hyperlink": MSO_THEME_COLOR.HYPERLINK,
    "followed_hyperlink": MSO_THEME_COLOR.FOLLOWED_HYPERLINK,
}


def _hex_to_rgb(value: str) -> RGBColor:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        raise ValueError(f"invalid RGB color {value!r}")
    return RGBColor.from_string(text.upper())


def apply_color(color_format: Any, color: ColorRef) -> None:
    """Apply a theme or explicit RGB color to a python-pptx color format."""

    key = color.value.strip().lower()
    if key in _THEME_MAP:
        color_format.theme_color = _THEME_MAP[key]
    else:
        color_format.rgb = _hex_to_rgb(color.value)
    if color.transparency > 0.0:
        # python-pptx does not expose transparency on ColorFormat. The caller
        # applies it to the fill XML when supported.
        pass


@dataclass(frozen=True, slots=True)
class ThemeFontPolicy:
    """Text policy used by the native renderer.

    ``inherit=True`` means no explicit typeface is written, so the run uses the
    presentation theme.  This is the safest way to keep generated text aligned
    with a PowerPoint 2019+ template.
    """

    inherit: bool = True
    fallback_latin: str | None = None
    fallback_east_asian: str | None = None
