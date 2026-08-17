# Results visualization changelog — 2026-08-01

## Added

- `battery.py`: 20 battery/electrochemical multiscale plots.
- `phase_field.py`: 17 phase-field and dendrite plots.
- `multiscale.py`: 13 scale-bridging, validation and sensitivity plots.
- `compose_plots` and `shared_legend` in `panels.py`.
- `battery`, `phase-field` and `thermal` themes.
- Universal `PlotConfig` controls for artist, legend, axis and layout styling.
- `PlotResult.restyle()` and `PlotResult.combine_legend()`.
- Audit, literature mapping, examples and smoke tests.

## Changed

- Results registry: 110 → 162 entries.
- Top-level visualization imports are lazy.
- Multi-panel composer accepts registry names and shared legends.
- Existing plot parameters now affect their plots and validate input shapes.

## Fixed

See `RESULTS_AUDIT.md` for the complete defect list and validation record.
