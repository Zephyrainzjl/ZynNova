# Plot catalog

Total registered plot functions: **162**.

The table is generated from the live registry. Registry names and aliases can be passed to `zv.plot(...)` or used in `PanelSpec`.

## animation

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `animate-embedding` | `animate_embedding` | `animate-embedding` | Animate evolving embeddings, latent trajectories or particle projections. |
| `animate-field` | `animate_field` | `animate-field`, `field-animation` | Animate 2-D scalar fields, phase fields, microscopy or spatial maps. |
| `animate-series` | `animate_series` | `animate-line`, `trajectory-animation` | Animate one or multiple evolving 1-D result series. |

## atomistic

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `angular-distribution-plot` | `angular_distribution_plot` | `angular-distribution` | Bond-angle or orientation distribution in Cartesian or polar form. |
| `contact-map-plot` | `contact_map_plot` | `contact-map`, `distance-map` | Residue/atom contact or pair-distance map. |
| `energy-conservation-plot` | `energy_conservation_plot` | `energy-drift`, `md-conservation` | MD energy conservation and drift diagnostics with optional smoothing. |
| `force-error-by-species` | `force_error_by_species` | `species-force-error` | Force-error distributions separated by chemical species. |
| `free-energy-surface` | `free_energy_surface` | `fes`, `free-energy-landscape` | 2-D potential of mean force from collective-variable samples. |
| `mean-squared-displacement-plot` | `mean_squared_displacement_plot` | `msd`, `mean-square-displacement` | MSD curves with Einstein diffusion fits and diffusion coefficients. |
| `mlip-parity-dashboard` | `mlip_parity_dashboard` | `mlip-dashboard`, `energy-force-stress-parity` | Multi-panel machine-learning potential validation dashboard. |
| `potential-of-mean-force` | `potential_of_mean_force` | `pmf` | One-dimensional PMF from sampled coordinates. |
| `radial-distribution-plot` | `radial_distribution_plot` | `rdf`, `pair-correlation` | Radial distribution functions with optional coordination-number axis. |
| `state-population-plot` | `state_population_plot` | `state-population` | Time-resolved conformational or phase-state populations. |
| `thermodynamic-trace` | `thermodynamic_trace` | `thermodynamic-trace` | Stacked MD traces for temperature, pressure, density and volume. |
| `velocity-autocorrelation-plot` | `velocity_autocorrelation_plot` | `vacf` | Velocity autocorrelation with optional vibrational density spectrum. |

## battery

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `capacity-fade-components-plot` | `capacity_fade_components_plot` | `capacity-fade-components`, `aging-breakdown` | Capacity-loss attribution to SEI, plating, LAM and transport losses. |
| `cell-voltage-breakdown-plot` | `cell_voltage_breakdown_plot` | `voltage-breakdown`, `cell-polarization-breakdown` | Cell-voltage decomposition into OCV and polarization losses. |
| `concentration-profile-plot` | `concentration_profile_plot` | `electrolyte-concentration-profile`, `li-concentration-profile` | Electrolyte or solid-phase concentration profiles across the cell. |
| `current-collector-temperature-plot` | `current_collector_temperature_plot` | `cell-temperature-map`, `collector-temperature` | Cell or current-collector temperature field with current-flow overlay. |
| `cycle-waterfall-plot` | `cycle_waterfall_plot` | `cycling-waterfall`, `voltage-waterfall` | Vertically offset voltage profiles that reveal cycle evolution. |
| `degradation-mode-map` | `degradation_mode_map` | `degradation-mode-map`, `aging-heatmap` | Cycle–position degradation heatmap for full-cell aging simulations. |
| `differential-capacity-map` | `differential_capacity_map` | `dqdv-map`, `incremental-capacity-map` | Cycle-resolved dQ/dV heatmap with optional tracked peak positions. |
| `electrode-phase-fraction-plot` | `electrode_phase_fraction_plot` | `electrode-phase-fraction`, `lithiation-phase-fraction` | Spatial or temporal phase fractions in phase-separating electrodes. |
| `electrode-profile-plot` | `electrode_profile_plot` | `electrode-profile`, `through-thickness-profile` | Through-thickness electrode profiles with region shading. |
| `electrode-utilization-plot` | `electrode_utilization_plot` | `electrode-utilization`, `particle-utilization` | Spatial active-material utilization with optional groups and target. |
| `fast-charge-window-plot` | `fast_charge_window_plot` | `fast-charge-window`, `charging-operating-window` | Fast-charge objective surface with electrochemical and thermal constraints. |
| `impedance-drt-plot` | `impedance_drt_plot` | `drt`, `impedance-drt` | Distribution-of-relaxation-times spectra for impedance analysis. |
| `lithium-plating-risk-map` | `lithium_plating_risk_map` | `plating-risk-map`, `lithium-plating-map` | SOC–temperature lithium-plating risk surface and safe boundary. |
| `operando-map-plot` | `operando_map_plot` | `operando-map`, `spatiotemporal-electrode-map` | Operando spatial signal map with optional synchronized voltage trace. |
| `overpotential-decomposition-plot` | `overpotential_decomposition_plot` | `overpotential-stack`, `voltage-loss-breakdown` | Activation, ohmic, diffusion and other overpotential contributions. |
| `particle-utilization-histogram` | `particle_utilization_histogram` | `particle-utilization-histogram`, `particle-soc-distribution` | Distribution of particle SOC or utilization by electrode region. |
| `reaction-current-profile-plot` | `reaction_current_profile_plot` | `reaction-current-profile`, `local-current-density` | Local interfacial reaction-current distribution through an electrode. |
| `stack-pressure-performance-plot` | `stack_pressure_performance_plot` | `stack-pressure-performance`, `pressure-cell-map` | Stack-pressure trade-off curve for solid-state battery simulations. |
| `through-plane-transport-plot` | `through_plane_transport_plot` | `transport-profile`, `effective-transport` | Ionic, electronic and diffusive transport coefficients through thickness. |
| `tortuosity-porosity-plot` | `tortuosity_porosity_plot` | `tortuosity-porosity`, `bruggeman-map` | Porosity–tortuosity design map with optional Bruggeman reference. |

## biology

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `enrichment-bubble-plot` | `enrichment_bubble_plot` | `enrichment-bubble`, `pathway-enrichment` | Pathway/ontology enrichment bubble chart. |
| `expression-dot-plot` | `expression_dot_plot` | `single-cell-dotplot`, `marker-dotplot` | Single-cell marker dot plot: color=mean, size=fraction expressing. |
| `expression-heatmap` | `expression_heatmap` | `expression-heatmap`, `clustered-heatmap` | Clustered gene/protein/feature expression heatmap without seaborn. |
| `ma-plot` | `ma_plot` | `ma` | MA plot for abundance-dependent differential effects. |
| `manhattan-plot` | `manhattan_plot` | `manhattan` | GWAS Manhattan plot with alternating chromosomes and hit labels. |
| `spatial-expression-plot` | `spatial_expression_plot` | `spatial-expression`, `spatial-omics` | Spatial transcriptomic/proteomic values over tissue or microscopy image. |
| `survival-curve-plot` | `survival_curve_plot` | `kaplan-meier`, `survival` | Kaplan–Meier survival curves with Greenwood confidence bands. |
| `volcano-plot` | `volcano_plot` | `volcano` | Differential-expression volcano plot with highlighting and q-value labels. |

## distributions

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `empirical-cdf` | `empirical_cdf` | `ecdf` | Empirical cumulative or survival distribution for robust comparisons. |
| `histogram-density` | `histogram_density` | `hist-kde` | Histogram and optional KDE overlay with consistent normalization. |
| `mirrored-histogram` | `mirrored_histogram` | `mirror-histogram` | Mirrored histograms for comparing uncertainty and true error distributions. |
| `raincloud-plot` | `raincloud_plot` | `raincloud` | Raincloud plot combining half-violin, box summary and raw observations. |
| `ridge-plot` | `ridge_plot` | `ridge`, `joyplot` | Ridgeline density plot for temperatures, compositions or cohorts. |
| `violin-box-scatter` | `violin_box_scatter` | `violin-box`, `distribution-summary` | Layer violin, quartile box and jittered data without seaborn. |

## electrochemistry

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `bode-plot` | `bode_plot` | `bode`, `eis-bode` | Two-panel Bode magnitude and phase plot. |
| `cyclic-voltammetry-plot` | `cyclic_voltammetry_plot` | `cyclic-voltammetry`, `cv` | Cyclic voltammogram with cycle- and scan-rate-separated traces. |
| `cycling-performance-plot` | `cycling_performance_plot` | `cycling-retention` | Capacity retention and coulombic efficiency versus cycle number. |
| `differential-capacity-plot` | `differential_capacity_plot` | `dqdv`, `differential-capacity` | Differential capacity dQ/dV or differential voltage dV/dQ. |
| `nyquist-plot` | `nyquist_plot` | `nyquist`, `eis-nyquist` | Electrochemical impedance Nyquist plot with frequency annotations. |
| `ragone-plot` | `ragone_plot` | `ragone` | Ragone energy–power trade-off map. |
| `rate-capability-plot` | `rate_capability_plot` | `rate-capability` | Rate capability across C-rates or current densities. |
| `soc-temperature-map` | `soc_temperature_map` | `soc-temperature-map`, `battery-operating-map` | SOC–temperature performance, degradation or risk surface. |
| `voltage-capacity-plot` | `voltage_capacity_plot` | `voltage-capacity`, `charge-discharge` | Charge/discharge voltage profiles colored by cycle or rate. |

## embeddings

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `cluster-hulls` | `cluster_hulls` | `cluster-hulls` | Cluster scatter with convex hulls for phases, cell types or regimes. |
| `embedding-density` | `embedding_density` | `embedding-density`, `latent-density` | Hexbin or smooth density map for very large embeddings. |
| `embedding-scatter` | `embedding_scatter` | `latent-scatter`, `umap-plot`, `tsne-plot` | General UMAP/t-SNE/PCA/latent-space scatter for materials and biology. |
| `embedding-stability-plot` | `embedding_stability_plot` | `embedding-stability` | Neighborhood-overlap profile for comparing latent projections. |
| `latent-property-surface` | `latent_property_surface` | `latent-property-map` | Interpolate a property over a learned latent manifold. |
| `trajectory-embedding` | `trajectory_embedding` | `pseudotime-trajectory`, `latent-trajectory` | Latent trajectory or pseudotime path with directional arrows. |

## explainability

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `attention-map` | `attention_map` | `attention` | Attention or attribution matrix for tokens, atoms, residues or graph nodes. |
| `attribution-waterfall` | `attribution_waterfall` | `attribution-waterfall` | Local additive-explanation waterfall from baseline to prediction. |
| `feature-importance-plot` | `feature_importance_plot` | `importance` | Horizontal feature-importance ranking with optional confidence bars. |
| `interaction-heatmap` | `interaction_heatmap` | `interaction-map` | Symmetric interaction-strength heatmap with optional clustering. |
| `partial-dependence-plot` | `partial_dependence_plot` | `partial-dependence`, `ice` | Partial-dependence curve with ICE trajectories and uncertainty band. |
| `shap-beeswarm` | `shap_beeswarm` | `shap-summary`, `attribution-beeswarm` | Dependency-free SHAP-style beeswarm for local attribution matrices. |

## fields

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `crack-path-plot` | `crack_path_plot` | `crack-path`, `damage-map` | Phase-field fracture damage map, crack path and displacement vectors. |
| `kymograph-plot` | `kymograph_plot` | `kymograph`, `space-time-map` | Space–time kymograph for fronts, waves and concentration profiles. |
| `mesh-quality-plot` | `mesh_quality_plot` | `mesh-quality` | Finite-element mesh-quality distribution with failure threshold. |
| `orthogonal-slices-plot` | `orthogonal_slices_plot` | `orthogonal-slices`, `volume-slices` | Three orthogonal slices through a 3-D scalar volume. |
| `phase-field-montage` | `phase_field_montage` | `phase-montage`, `field-montage` | Snapshot montage for phase-field, damage or concentration evolution. |
| `scalar-field-plot` | `scalar_field_plot` | `scalar-map`, `heatmap-field` | Scalar simulation field with optional contours and physical coordinates. |
| `vector-field-plot` | `vector_field_plot` | `quiver`, `vector-map` | 2-D vector field with magnitude backdrop and quiver arrows. |

## interactive

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `interactive-embedding-plot` | `interactive_embedding_plot` | `interactive-embedding` | Interactive 2-D or 3-D embedding with rich hover information. |
| `interactive-pareto-plot` | `interactive_pareto_plot` | `interactive-pareto` | Interactive 2-D/3-D Pareto front for candidate exploration. |
| `interactive-parity-plot` | `interactive_parity_plot` | `interactive-parity` | Interactive Plotly parity plot with hover labels and regression metrics. |
| `interactive-sankey-plot` | `interactive_sankey_plot` | `interactive-sankey` | Interactive Sankey flow diagram for mechanisms, states or data pipelines. |
| `interactive-volume-plot` | `interactive_volume_plot` | `interactive-volume` | Interactive Plotly volume rendering for fields and tomography. |

## materials

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `band-dos-plot` | `band_dos_plot` | `band-dos` | Journal-style combined band-structure and DOS panel. |
| `band-structure-plot` | `band_structure_plot` | `bands` | Electronic band structure with symmetry path, spin and occupations. |
| `convex-hull-plot` | `convex_hull_plot` | `convex-hull`, `formation-energy-hull` | Binary composition–formation-energy convex hull with stability labels. |
| `density-of-states-plot` | `density_of_states_plot` | `dos`, `density-of-states` | Total and projected density of states with spin mirroring. |
| `diffraction-pattern-plot` | `diffraction_pattern_plot` | `xrd`, `diffraction-pattern` | XRD/neutron diffraction pattern with reference sticks and peak labels. |
| `elastic-polar-plot` | `elastic_polar_plot` | `elastic-polar`, `directional-property` | Polar anisotropy plot for modulus, conductivity or surface energy. |
| `equation-of-state-plot` | `equation_of_state_plot` | `eos`, `equation-of-state` | Energy–volume equation of state with optional pressure overlay. |
| `phase-diagram-2d` | `phase_diagram_2d` | `phase-map`, `phase-diagram` | Categorical 2-D phase diagram from grids or scattered samples. |
| `phonon-dispersion-plot` | `phonon_dispersion_plot` | `phonons` | Phonon dispersion with imaginary-mode highlighting. |
| `stacked-spectra` | `stacked_spectra` | `spectra-stack`, `waterfall-spectra` | Stack Raman/IR/XPS/NMR/XAS spectra with controlled vertical offsets. |
| `stress-strain-plot` | `stress_strain_plot` | `stress-strain` | Stress–strain curves with group-wise modulus, strength and toughness. |

## model-evaluation

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `confusion-matrix-plot` | `confusion_matrix_plot` | `confusion` | Confusion matrix from raw labels or a precomputed square matrix. |
| `parity-plot` | `parity_plot` | `prediction-parity`, `predicted-vs-true` | Publication parity plot with density, hexbin or grouped scatter modes. |
| `performance-profile` | `performance_profile` | `performance-profile` | Dolan–Moré style performance profile across benchmark tasks. |
| `precision-recall-plot` | `precision_recall_plot` | `pr-curve`, `precision-recall` | Precision–recall curve with average precision. |
| `radar-plot` | `radar_plot` | `benchmark-radar` | Radar chart for compact multi-metric benchmark comparisons. |
| `residual-diagnostics` | `residual_diagnostics` | `residual-dashboard` | Four-panel residual diagnostics: parity, residual, QQ and distribution. |
| `roc-curve-plot` | `roc_curve_plot` | `roc` | ROC curve with AUC and optional bootstrap confidence interval. |

## multiscale

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `computational-cost-scaling-plot` | `computational_cost_scaling_plot` | `cost-scaling`, `runtime-scaling` | Runtime and memory scaling for atomistic-to-continuum workflows. |
| `experiment-simulation-overlay` | `experiment_simulation_overlay` | `experiment-simulation-overlay`, `validation-overlay` | Experiment/simulation overlay with uncertainty and goodness-of-fit. |
| `field-profile-comparison` | `field_profile_comparison` | `field-profile-comparison`, `profile-validation` | Overlay field profiles from multiple scales, solvers or experiments. |
| `homogenization-comparison-plot` | `homogenization_comparison_plot` | `homogenization-comparison`, `resolved-vs-homogenized` | Compare microstructure-resolved and homogenized model predictions. |
| `mesh-convergence-plot` | `mesh_convergence_plot` | `mesh-convergence`, `discretization-convergence` | Finite-element/finite-volume convergence against mesh size or DOFs. |
| `model-hierarchy-plot` | `model_hierarchy_plot` | `model-hierarchy`, `fidelity-hierarchy` | Accuracy–cost hierarchy from DFT/MD to continuum and system models. |
| `parameter-sensitivity-tornado` | `parameter_sensitivity_tornado` | `sensitivity-tornado`, `parameter-tornado` | Tornado chart for one-at-a-time multiscale parameter sensitivity. |
| `representative-volume-convergence-plot` | `representative_volume_convergence_plot` | `rve-convergence`, `representative-volume-convergence` | Representative-volume convergence of effective properties. |
| `scale-bridge-plot` | `scale_bridge_plot` | `scale-bridge`, `multiscale-bridge` | Visual map from electronic/atomistic scales to electrode, cell and pack. |
| `scale-transfer-matrix-plot` | `scale_transfer_matrix_plot` | `scale-transfer-matrix`, `coupling-matrix` | Information/parameter transfer strength between model scales. |
| `sobol-indices-plot` | `sobol_indices_plot` | `sobol-indices`, `global-sensitivity` | First-order and total Sobol sensitivity indices. |
| `spatial-error-map` | `spatial_error_map` | `spatial-error-map`, `field-error-map` | Spatial absolute or relative error between resolved simulation fields. |
| `uncertainty-fan-plot` | `uncertainty_fan_plot` | `uncertainty-fan`, `ensemble-fan` | Nested ensemble-quantile fan for propagated multiscale uncertainty. |

## networks

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `adjacency-matrix-plot` | `adjacency_matrix_plot` | `adjacency` | Adjacency or contact matrix with community-aware reordering. |
| `alluvial-plot` | `alluvial_plot` | `alluvial`, `sankey` | Two-stage alluvial/Sankey diagram implemented with Matplotlib ribbons. |
| `chord-diagram` | `chord_diagram` | `chord` | Circular chord diagram for transitions, interactions or flows. |
| `network-plot` | `network_plot` | `graph`, `network` | General graph visualization with reusable node and edge styling. |
| `reaction-network-plot` | `reaction_network_plot` | `reaction-network`, `pathway-network` | Directed chemical or biochemical reaction network with rate-weighted edges. |

## optimization

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `acquisition-landscape` | `acquisition_landscape` | `acquisition-map` | Two-dimensional Bayesian-optimization acquisition landscape. |
| `active-learning-progress` | `active_learning_progress` | `active-learning-progress`, `discovery-progress` | Active-learning discovery progress with best-so-far and uncertainty. |
| `hypervolume-curve` | `hypervolume_curve` | `hypervolume` | Hypervolume versus query budget for multi-objective optimization. |
| `parallel-coordinates-plot` | `parallel_coordinates_plot` | `parallel-coordinates` | Parallel-coordinates plot for composition–process–property spaces. |
| `pareto-front-plot` | `pareto_front_plot` | `pareto`, `pareto-scatter` | Two- or three-objective Pareto front with highlighted non-dominated set. |
| `ternary-composition-plot` | `ternary_composition_plot` | `ternary`, `composition-ternary` | Dependency-free ternary composition–property map. |

## panels

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `add-panel-labels` | `add_panel_labels` | `panel-labels` | Add journal-style a, b, c panel labels to one or many axes. |
| `broken-axis-plot` | `broken_axis_plot` | `broken-axis` | Two-panel broken y-axis for separated magnitude ranges. |
| `compose-plots` | `compose_plots` | `compose-figure`, `multi-panel-figure` | Functional API for composing registered plots into one figure. |
| `inset-zoom` | `inset_zoom` | `inset-zoom` | Create a reusable zoomed inset and mark the selected region. |
| `shared-legend` | `shared_legend` | `shared-legend`, `figure-legend` | Collect handles from multiple axes into one figure-level legend. |

## phase-field

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `chemical-potential-profile-plot` | `chemical_potential_profile_plot` | `chemical-potential-profile`, `mu-profile` | Chemical-potential profile with diffuse-interface locations. |
| `dendrite-branching-plot` | `dendrite_branching_plot` | `dendrite-branching`, `branch-statistics` | Dendrite branch and tip statistics through phase-field evolution. |
| `dendrite-morphology-plot` | `dendrite_morphology_plot` | `dendrite-morphology`, `phase-dendrite` | Dendrite phase morphology with concentration or potential backdrop. |
| `dendrite-tip-velocity-plot` | `dendrite_tip_velocity_plot` | `dendrite-tip-velocity`, `front-velocity` | Dendrite-tip position and derived velocity versus time. |
| `free-energy-landscape-plot` | `free_energy_landscape_plot` | `phase-free-energy-landscape`, `double-well` | Bulk free-energy density, common tangent and chemical potential. |
| `front-position-plot` | `front_position_plot` | `front-position`, `interface-position` | Interface/front position and optional power-law scaling reference. |
| `grain-orientation-map` | `grain_orientation_map` | `grain-orientation-map`, `orientation-field` | Periodic grain-orientation field with phase/interface overlay. |
| `interface-curvature-plot` | `interface_curvature_plot` | `interface-curvature`, `curvature-map` | Approximate diffuse-interface curvature derived from the phase gradient. |
| `interfacial-energy-polar-plot` | `interfacial_energy_polar_plot` | `interfacial-energy-polar`, `surface-energy-anisotropy` | Polar interfacial-energy anisotropy and optional surface stiffness. |
| `morphology-comparison-grid` | `morphology_comparison_grid` | `morphology-comparison`, `phase-field-comparison-grid` | Journal-style comparison grid for phase-field morphologies. |
| `morphology-metrics-plot` | `morphology_metrics_plot` | `morphology-metrics`, `dendrite-metrics` | Small-multiple morphology metrics such as roughness and branch count. |
| `nucleation-growth-map` | `nucleation_growth_map` | `nucleation-growth-map`, `nucleation-regime` | Nucleation/growth response over driving-force and interface-energy space. |
| `order-parameter-distribution-plot` | `order_parameter_distribution_plot` | `order-parameter-distribution`, `phase-histogram` | Evolution of order-parameter distributions during phase separation. |
| `phase-boundary-plot` | `phase_boundary_plot` | `phase-boundary`, `interface-map` | Diffuse-interface contours over a phase or coupled-field background. |
| `phase-field-convergence-plot` | `phase_field_convergence_plot` | `phase-field-convergence`, `solver-convergence` | Nonlinear/linear solver residual histories for phase-field systems. |
| `phase-field-energy-plot` | `phase_field_energy_plot` | `phase-field-energy`, `energy-dissipation` | Phase-field energy components and total free-energy dissipation. |
| `stress-concentration-coupling-plot` | `stress_concentration_coupling_plot` | `stress-concentration-coupling`, `chemo-mechanical-map` | Coupled concentration–stress relation from chemo-mechanical simulations. |

## statistics

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `agreement-plot` | `agreement_plot` | `bland-altman` | Bland–Altman agreement plot with limits of agreement and bias trend. |
| `binned-trend` | `binned_trend` | `binned-relation`, `conditional-trend` | Show raw observations and a robust binned conditional trend. |
| `correlation-matrix` | `correlation_matrix` | `correlation-heatmap` | Correlation heatmap with optional hierarchical reordering and masking. |
| `effect-size-forest` | `effect_size_forest` | `forest`, `effect-forest` | Forest plot for effects, confidence intervals and optional sample weights. |
| `line-with-uncertainty` | `line_with_uncertainty` | `trend-band`, `uncertainty-band` | Plot a mean trajectory with asymmetric uncertainty or symmetric error. |
| `quantile-dotplot` | `quantile_dotplot` | `quantile-dot` | Compact quantile glyphs for comparing many distributions. |

## training

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `benchmark-heatmap` | `benchmark_heatmap` | `benchmark-heatmap` | Model × task benchmark matrix with raw scores or within-task ranks. |
| `dataset-cartography-plot` | `dataset_cartography_plot` | `dataset-cartography` | Dataset cartography: confidence versus variability and correctness. |
| `gradient-flow-plot` | `gradient_flow_plot` | `gradient-flow` | Layerwise gradient magnitude diagnostic for vanishing or exploding gradients. |
| `learning-rate-schedule-plot` | `learning_rate_schedule_plot` | `lr-schedule`, `learning-rate` | Learning-rate schedule with optional synchronized loss. |
| `training-history-plot` | `training_history_plot` | `learning-curves`, `history` | Flexible training-history panels for losses, metrics and learning rates. |

## uncertainty

| Registry name | Python function | Aliases | Description |
|---|---|---|---|
| `calibration-plot` | `calibration_plot` | `reliability`, `calibration` | Reliability diagram with expected calibration error and confidence histogram. |
| `conformal-coverage-plot` | `conformal_coverage_plot` | `conformal-coverage` | Nominal versus empirical coverage for conformal or Bayesian intervals. |
| `error-vs-uncertainty` | `error_vs_uncertainty` | `error-uncertainty`, `uq-hexbin` | Relation between predicted uncertainty and realized error. |
| `prediction-interval-plot` | `prediction_interval_plot` | `prediction-band`, `interval-coverage` | Ordered prediction intervals with observed values and empirical coverage. |
| `uncertainty-decomposition` | `uncertainty_decomposition` | `variance-decomposition` | Visualize aleatoric, epistemic and misspecification contributions. |
