from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..calculators.classical import create_classical_calculator


@dataclass(slots=True)
class LAMMPSLibConfig:
    """Configuration for an in-process LAMMPS library calculator."""

    commands: list[str]
    atom_types: dict[str, int]
    amendments: list[str] = field(default_factory=list)
    log_file: str | None = None
    keep_alive: bool = True
    lammps_header: list[str] | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_calculator(self):
        kwargs = dict(self.extra_kwargs)
        kwargs.update(
            {
                "lmpcmds": list(self.commands),
                "atom_types": dict(self.atom_types),
                "amendments": list(self.amendments),
                "log_file": self.log_file,
                "keep_alive": self.keep_alive,
            }
        )
        if self.lammps_header is not None:
            kwargs["lammps_header"] = list(self.lammps_header)
        return create_classical_calculator("lammpslib", **kwargs)
