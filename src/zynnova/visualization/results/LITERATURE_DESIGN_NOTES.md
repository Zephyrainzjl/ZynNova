# Literature-informed battery, multiscale and phase-field visual grammars

The implementation does **not** copy any published figure. It identifies
recurring scientific visual grammars in high-impact literature and turns them
into configurable, data-agnostic APIs. Colors, line styles, labels, legends,
axes positions and multi-panel layouts remain controlled by the caller.

## 1. Three-dimensional electrode microstructure and transport

Recurring visual elements include segmented 3-D/2-D morphology, through-plane
profiles, tortuosity–porosity relations, phase fractions, spatial reaction
current and resolved-versus-homogenized comparisons.

Relevant APIs:

- `electrode_profile_plot`, `concentration_profile_plot`
- `reaction_current_profile_plot`, `electrode_utilization_plot`
- `tortuosity_porosity_plot`, `through_plane_transport_plot`
- `electrode_phase_fraction_plot`, `homogenization_comparison_plot`
- `representative_volume_convergence_plot`, `spatial_error_map`

Representative studies:

1. Lu et al., *3D microstructure design of lithium-ion battery electrodes
   assisted by X-ray nano-computed tomography and modelling*, **Nature
   Communications** 11, 2079 (2020), DOI: 10.1038/s41467-020-15811-x.
2. Zhang et al., *Coupling of multiscale imaging analysis and computational
   modeling for understanding thick cathode degradation mechanisms*, **Joule**
   7, 201–220 (2023), DOI: 10.1016/j.joule.2022.12.001.
3. Müller et al., *Deep learning-based segmentation of lithium-ion battery
   electrodes*, **Nature Communications** (2021), DOI:
   10.1038/s41467-021-26480-9.
4. Finegan et al., *In-operando high-speed tomography of lithium-ion batteries
   during thermal runaway*, **Nature Communications** (2015), DOI:
   10.1038/ncomms7924.
5. Lee et al., *Advanced parametrization for the production of high-energy
   lithium-ion cells*, **Nature Communications** (2024), DOI:
   10.1038/s41467-024-50075-9.

## 2. Fast charging, plating and cell-scale heterogeneity

High-information battery figures repeatedly combine voltage/capacity curves,
differential-capacity maps, SOC–temperature operating windows, plating-risk
boundaries, thermal maps, operando space–time maps and degradation attribution.

Relevant APIs:

- `voltage_capacity_plot`, `differential_capacity_map`
- `lithium_plating_risk_map`, `fast_charge_window_plot`
- `soc_temperature_map`, `current_collector_temperature_plot`
- `operando_map_plot`, `capacity_fade_components_plot`
- `cell_voltage_breakdown_plot`, `cycle_waterfall_plot`

Representative studies:

6. Lu et al., *Multiscale dynamics of charging and plating in graphite
   electrodes*, **Nature Communications** (2023), DOI:
   10.1038/s41467-023-40574-6.
7. Lin et al., *Multiscale coupling of surface temperature with solid-phase
   reactions in lithium-ion batteries*, **Communications Engineering** (2022),
   DOI: 10.1038/s44172-022-00005-8.
8. Song et al., *A microstructural electrochemo-mechanical model of high-energy
   lithium-ion batteries*, **Energy & Environmental Science** (2025), DOI:
   10.1039/D4EE04856C.
9. Konz et al., lithium-plating onset and fast-charge monitoring with voltage
   relaxation, **ACS Energy Letters** 5, 1750–1757 (2020).
10. Wang et al., underpotential lithium plating caused by temperature
    heterogeneity, **PNAS** 117, 29453 (2020).

## 3. Phase-field morphology, interfaces and dendrite growth

Phase-field papers commonly present order-parameter snapshots, interface
contours, concentration/potential overlays, free-energy curves, chemical
potential profiles, anisotropic interfacial energy, tip velocity, curvature,
stress coupling, branch counts and energy convergence.

Relevant APIs:

- `dendrite_morphology_plot`, `phase_boundary_plot`
- `free_energy_landscape_plot`, `chemical_potential_profile_plot`
- `interfacial_energy_polar_plot`, `interface_curvature_plot`
- `dendrite_tip_velocity_plot`, `dendrite_branching_plot`
- `stress_concentration_coupling_plot`, `phase_field_energy_plot`
- `phase_field_convergence_plot`, `morphology_comparison_grid`

Representative studies:

11. Wang et al., *Application of phase-field method in rechargeable batteries*,
    **npj Computational Materials** 6 (2020), DOI:
    10.1038/s41524-020-00445-w.
12. Kamikawa et al., *Chemo-electro-mechanical phase-field simulation of
    lithium dendrite penetration*, **Communications Materials** (2024), DOI:
    10.1038/s43246-024-00600-6.
13. Yildirim et al., *Understanding the origin of lithium dendrite branching in
    Li solid electrolytes*, **Nature Communications** (2024), DOI:
    10.1038/s41467-024-52412-4.
14. Li et al., *Proactive lithium dendrite regulation enabled by manipulating
    separator microstructure using high-fidelity phase-field simulation*,
    **Advanced Energy Materials** (2025), DOI: 10.1002/aenm.202500503.
15. Han et al., *Manipulation of lithium dendrites based on electric field
    regulation*, **Nature Communications** (2025), DOI:
    10.1038/s41467-025-58818-y.
16. Pokharel et al., diffusion-barrier control at the lithium metal/SEI
    interface, **Nature Communications** (2024), DOI:
    10.1038/s41467-024-47521-z.

## 4. Chemo-mechanics, grains and coupled fields

Correlative studies often use orientation maps, phase boundaries, line
profiles, density-colored correlation plots, stress fields and grain-resolved
comparisons.

Relevant APIs:

- `grain_orientation_map`, `stress_concentration_coupling_plot`
- `field_profile_comparison`, `spatial_error_map`
- `crack_path_plot`, `vector_field_plot`
- `morphology_metrics_plot`, `model_hierarchy_plot`

Representative studies:

17. Liu et al., *Role of grain-level chemo-mechanics in composite cathodes*,
    **Nature Communications** (2024), DOI: 10.1038/s41467-024-52123-w.
18. Li et al., *Mutual modulation between surface chemistry and bulk
    microstructure in nickel-rich cathodes*, **Nature Communications** (2020),
    DOI: 10.1038/s41467-020-18278-y.
19. Chang et al., *Evolving contact mechanics and microstructure formation in
    solid-state lithium batteries*, **Nature Communications** (2021), DOI:
    10.1038/s41467-021-26632-x.

## 5. Multiscale coupling, uncertainty and computational design

Multiscale papers increasingly display explicit scale bridges, model
hierarchies, transfer matrices, RVE/mesh convergence, computational scaling,
Sobol sensitivity, uncertainty fans and experiment–simulation overlays.

Relevant APIs:

- `scale_bridge_plot`, `model_hierarchy_plot`
- `scale_transfer_matrix_plot`, `mesh_convergence_plot`
- `computational_cost_scaling_plot`, `representative_volume_convergence_plot`
- `parameter_sensitivity_tornado`, `sobol_indices_plot`
- `uncertainty_fan_plot`, `experiment_simulation_overlay`

Representative studies:

20. *Knowledge-driven design of solid-electrolyte interphases on lithium metal
    via multiscale modelling*, **Nature Communications** 14, 6823 (2023), DOI:
    10.1038/s41467-023-42212-7.
21. *Navigating chemical design spaces for metal-ion batteries via
    machine-learning-guided phase-field simulations*, **npj Computational
    Materials** 11, 243 (2025), DOI: 10.1038/s41524-025-01735-x.
22. *Predicting dendrite growth in lithium metal batteries through iterative
    neural networks and voltage embedding*, **npj Computational Materials** 11,
    337 (2025), DOI: 10.1038/s41524-025-01824-x.
23. *Machine-learning-accelerated mechanistic exploration of interface
    modification in lithium metal anodes*, **npj Computational Materials**
    (2025), DOI: 10.1038/s41524-025-01747-7.

## Publication implementation principles

- Preserve vector text and axes in PDF/SVG; rasterize only dense marks.
- Return processed data, metrics and named artists with every static plot.
- Use redundant encodings (color plus line style/marker) where possible.
- Keep scientific layers separable: field, contour, annotation, threshold,
  uncertainty and legend proxy are independent named artists.
- Support caller-owned axes and registry-driven multi-panel composition.
- Avoid hidden global style mutation by applying themes within context managers.
- Show uncertainty and constraints explicitly rather than encoding them only in
  color.
