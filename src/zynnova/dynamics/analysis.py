from __future__ import annotations

from typing import Any

import numpy as np

from .results import ThermoSeries


def summarize_thermo(series: ThermoSeries, *, num_atoms: int | None = None) -> dict[str, Any]:
    arrays = series.as_arrays()
    if not len(arrays["step"]):
        return {"samples": 0}
    divisor = max(int(num_atoms or 1), 1)
    total = arrays["total_energy_eV"] / divisor
    temperature = arrays["temperature_K"]
    pressure = arrays["pressure_GPa"]
    summary = {
        "samples": int(len(total)),
        "first_step": int(arrays["step"][0]),
        "last_step": int(arrays["step"][-1]),
        "mean_temperature_K": float(np.nanmean(temperature)),
        "std_temperature_K": float(np.nanstd(temperature)),
        "mean_total_energy_eV_per_atom": float(np.nanmean(total)),
        "energy_drift_eV_per_atom": float(total[-1] - total[0]),
        "max_force_eV_per_A": float(np.nanmax(arrays["max_force_eV_per_A"])),
    }
    if np.any(np.isfinite(pressure)):
        summary["mean_pressure_GPa"] = float(np.nanmean(pressure))
        summary["std_pressure_GPa"] = float(np.nanstd(pressure))
    return summary
