from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...polymer_utils import polymer_record


@dataclass(slots=True)
class PolymerValidityReport:
    valid: bool
    reason: str | None
    record: Any | None
    port_count: int
    atom_count: int


def validate_generated_polymer(
    psmiles: str,
    *,
    require_two_ports: bool = True,
) -> PolymerValidityReport:
    if not psmiles.strip():
        return PolymerValidityReport(False, "empty PSMILES", None, 0, 0)
    try:
        record = polymer_record(psmiles)
        record.validate()
    except Exception as exc:
        return PolymerValidityReport(
            False,
            f"chemical parser rejected PSMILES: {exc}",
            None,
            0,
            0,
        )
    port_count = sum(len(unit.graph.ports) for unit in record.units.values())
    atom_count = sum(len(unit.graph.atoms) for unit in record.units.values())
    if require_two_ports and port_count != 2:
        return PolymerValidityReport(
            False,
            f"repeat unit requires exactly two polymerization ports, found {port_count}",
            record,
            port_count,
            atom_count,
        )
    if atom_count < 2:
        return PolymerValidityReport(
            False,
            "repeat unit contains fewer than two atoms",
            record,
            port_count,
            atom_count,
        )
    return PolymerValidityReport(True, None, record, port_count, atom_count)


__all__ = ["PolymerValidityReport", "validate_generated_polymer"]
