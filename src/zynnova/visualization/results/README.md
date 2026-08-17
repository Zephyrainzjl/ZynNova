# ZynNova reusable result visualization

`zynnova.visualization.results` is a lazy-loaded, publication-oriented plotting
layer for materials science, atomistic simulation, machine learning,
electrochemistry, battery multiscale modelling, phase-field simulation,
continuum fields, biology and scientific networks.

The live registry contains **162** reusable plots. Every static plot returns a
`PlotResult` containing `figure`, `axes`, named `artists`, processed `data`,
computed `metrics` and `metadata`.

## Minimal use

```python
from zynnova.visualization import results as zv

result = zv.parity_plot(
    reference,
    prediction,
    groups=species,
    mode="hexbin",
    config=zv.PlotConfig(
        title="Force parity",
        xlabel="DFT force (eV Å$^{-1}$)",
        ylabel="MLIP force (eV Å$^{-1}$)",
        legend_loc="upper left",
        legend_ncol=2,
        line_width=1.4,
        marker_size=4,
    ),
    theme="nature",
)
result.save("force_parity.pdf")
```

## Universal style control

`PlotConfig` supports reusable control of figure size, titles, labels, axis
scales/limits, tick rotation, grid style, axes position, background, spines,
line color/width/style, marker/size, alpha, palettes, per-artist styles, legend
location/title/columns/order/labels, subplot spacing and metadata.

```python
config = zv.PlotConfig(
    figsize=(6.8, 4.2),
    palette=("#264653", "#2A9D8F", "#E76F51"),
    line_width=1.8,
    line_style="-",
    marker="o",
    marker_size=3.5,
    legend_loc="outside upper right",  # or any Matplotlib location
    legend_bbox_to_anchor=(1.02, 1.0),
    legend_frameon=False,
    legend_ncol=1,
    grid=True,
    grid_kwargs={"alpha": 0.2, "linewidth": 0.5},
    subplot_adjust={"right": 0.78},
)
```

A returned result can be changed without regenerating scientific data. Only the
explicitly supplied fields are modified:

```python
result.restyle(
    line_width=2.0,
    line_style="--",
    legend_loc="upper left",
    artist_styles={"identity": {"color": "black", "linewidth": 1.0}},
)
```

## Battery and phase-field examples

```python
risk = zv.lithium_plating_risk_map(
    soc,
    temperature,
    plating_indicator,
    threshold=0.5,
    config=zv.PlotConfig(title="Fast-charge plating boundary"),
)

morphology = zv.dendrite_morphology_plot(
    phase_field,
    concentration=electrolyte_concentration,
    potential=electric_potential,
)

bridge = zv.scale_bridge_plot(
    ["DFT", "MLMD", "phase field", "electrode", "cell"],
    length_ranges=length_ranges,
    time_ranges=time_ranges,
    methods=methods,
    couplings=couplings,
)
```

## Multi-panel composition

Every plot accepts an external `ax`. For registry-driven composition:

```python
panels = [
    zv.PanelSpec(
        "voltage-capacity",
        args=(capacity, voltage),
        kwargs={"cycles": cycles},
        title="Voltage profiles",
    ),
    zv.PanelSpec(
        "differential-capacity-map",
        args=(voltage_grid, cycle_grid, dqdv_map),
        title="Degradation evolution",
    ),
]

figure = zv.compose_plots(
    panels,
    nrows=1,
    ncols=2,
    figsize=(7.2, 3.2),
    shared_legend_enabled=True,
    panel_label_kwargs={"x": -0.10, "y": 1.04},
    theme="battery",
)
figure.save("battery_summary.pdf")
```

`FigureComposer` additionally supports shared axes, width/height ratios,
custom panel labels, a shared de-duplicated legend, subplot spacing and a
suptitle.

## Registry and discovery

```python
catalog = zv.plot_catalog()
print(len(catalog))
print(zv.available_plots(category="phase-field"))

result = zv.plot("interface-curvature", phase_field)
```

## Themes

Included themes are `nature`, `science`, `cell`, `colorblind`, `monochrome`,
`poster`, `nature-dark`, `battery`, `phase-field` and `thermal`. PDF/SVG text
remains editable by default. Dense marks can be rasterized while axes and text
stay vector.

## Optional dependencies

Core static plots use NumPy and Matplotlib. Selected functions use SciPy,
scikit-learn, NetworkX, Plotly or Pillow only when called. Missing optional
packages do not prevent the remaining modules from loading.
