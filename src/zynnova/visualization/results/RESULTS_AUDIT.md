# ZynNova `visualization.results` audit and repair report

Audit date: 2026-08-01

## Scope

The audit covered registry loading, lazy imports, theme resolution, common
configuration, result saving/restyling, all static result modules, multi-panel
composition and the public API pattern used by `from zynnova.visualization
import results as zv`.

## Corrected defects

| Area | Previous behavior | Correction |
|---|---|---|
| Package import | Importing `visualization` eagerly imported structure adapters and failed when the complete `zynnova.structure` package was not present. | Top-level exports now use lazy imports; `visualization.results` can be imported and tested independently. |
| Monochrome theme | `axes.prop_cycle=None` caused a Matplotlib `ValueError`. | A valid monochrome color/line-style cycle is used. |
| Public parameters | More than twenty public parameters were accepted but ignored. | Every public parameter is now referenced and validated. |
| Restyling | `result.restyle(line_width=...)` silently reset log scales, limits and layout to `PlotConfig` defaults. | Restyling applies only explicitly supplied fields. |
| Palette application | A palette was applied once per top-level artist container, causing all lines inside a list to receive the same color. | Palettes now cycle over leaf artists. |
| Existing axes | `dendrite_tip_velocity_plot(ax=...)` ignored the supplied axis. | One external axis creates a twin velocity axis; a two-axis sequence is also accepted. |
| Battery grouping | Several electrochemical plots exposed `groups`, `direction`, `scan_rate` and `constraints` without using them. | Group curves, charge/discharge styles, scan-rate labels and operating constraints are implemented. |
| Materials metadata | Band occupations, EOS pressure and grouped stress–strain curves were ignored. | Occupation encoding/colorbar, pressure twin axis and group-wise metrics are implemented. |
| Field metadata | Crack displacement, network edge values, embedding labels and animation trails were ignored. | Quiver overlays, edge color/width mapping, annotations and trails are implemented. |
| Scalar uncertainty | Scalar uncertainty values failed when boolean masks were applied. | Scalar, vector and asymmetric uncertainty inputs are broadcast/validated. |
| Masked statistics | Percentiles and error metrics could include masked backing values. | Statistics use compressed finite values. |
| Contour legend | New Matplotlib versions do not reliably expose `QuadContourSet.collections`. | Constraint contours use compatible proxy legend artists. |
| Documentation | The catalog stated 110 plots and omitted the battery/phase-field/multiscale families. | The catalog is generated from the live registry and now contains 162 plots. |

## New reusable plot families

- **Battery (20):** electrode profiles, concentration/current distributions,
  degradation maps, DRT, plating-risk and fast-charge maps, tortuosity,
  thermal/current-collector maps, voltage-loss breakdown, operando maps and
  stack-pressure optimization.
- **Phase field (17):** dendrite morphology, phase boundaries, free-energy and
  chemical-potential plots, interfacial anisotropy, tip velocity, nucleation,
  stress–concentration coupling, grain orientation, curvature, convergence,
  energy evolution and branching metrics.
- **Multiscale (13):** scale bridges, model hierarchy, homogenization and RVE
  convergence, mesh convergence, Sobol/tornado sensitivity, uncertainty fans,
  field errors, transfer matrices, computational scaling and
  experiment–simulation validation.
- **Composition (2):** registry-driven multi-panel composition and shared
  de-duplicated legends.

## Validation

- Live registry import: **162** plots.
- Themes exercised: `battery`, `cell`, `colorblind`, `monochrome`, `nature`,
  `nature-dark`, `phase-field`, `poster`, `science`, `thermal`.
- Representative runtime smoke tests: **71** static calls, plus animation and
  multi-panel composition tests.
- User-provided parity, residual, ROC and precision–recall examples were run
  against the revised API.
- Full source tree compiles with `python -m compileall`.
