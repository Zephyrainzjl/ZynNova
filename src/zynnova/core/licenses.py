"""Explicit license gates for third-party models and weight files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .exceptions import LicenseNotAcceptedError


@dataclass(frozen=True, slots=True)
class LicenseGate:
    backend: str
    license_id: str
    acceptance_environment_variable: str | None = None
    note: str = ""

    def accepted(self, *, explicit: bool = False) -> bool:
        if explicit:
            return True
        variable = self.acceptance_environment_variable
        if variable is None:
            return True
        return os.environ.get(variable, "").strip().lower() in {"1", "true", "yes", "accepted"}

    def require(self, *, explicit: bool = False) -> None:
        if self.accepted(explicit=explicit):
            return
        variable = self.acceptance_environment_variable
        guidance = ""
        if variable:
            guidance = f" Set {variable}=1 only after reviewing the terms."
        raise LicenseNotAcceptedError(
            f"{self.backend} uses {self.license_id}; explicit acceptance is required.{guidance} "
            f"{self.note}".strip()
        )


KNOWN_LICENSES: Mapping[str, LicenseGate] = {
    "map-anything-apache": LicenseGate("map-anything-apache", "Apache-2.0"),
    "trellis2": LicenseGate("trellis2", "MIT"),
    "pixal3d": LicenseGate("pixal3d", "MIT + third-party terms"),
    "meanvc2": LicenseGate("meanvc2", "Apache-2.0"),
    "x-vc": LicenseGate("x-vc", "MIT + dependency model terms"),
    "indextts-2.5": LicenseGate(
        "indextts-2.5",
        "bilibili Model Use License Agreement",
        "ZYNNOVA_ACCEPT_INDEXTTS_LICENSE",
        "Review the model-use license and disclaimer before enabling this backend.",
    ),
    "hy-world-2": LicenseGate(
        "hy-world-2",
        "Tencent Hunyuan Community License",
        "ZYNNOVA_ACCEPT_HY_WORLD_LICENSE",
        "The code and model terms must be reviewed separately before use.",
    ),
    "hunyuan3d": LicenseGate(
        "hunyuan3d",
        "Tencent Hunyuan Community License",
        "ZYNNOVA_ACCEPT_HUNYUAN3D_LICENSE",
        "Jurisdiction and use restrictions may apply.",
    ),
}


def require_known_license(name: str, *, explicit: bool = False) -> None:
    gate = KNOWN_LICENSES.get(name)
    if gate is not None:
        gate.require(explicit=explicit)


__all__ = ["KNOWN_LICENSES", "LicenseGate", "require_known_license"]
