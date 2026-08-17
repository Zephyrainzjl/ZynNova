"""Reusable publication-quality result visualizations.

The namespace is lazy-loaded so importing :mod:`zynnova.visualization` does not
require Matplotlib, SciPy, Plotly, NetworkX or scikit-learn.  Access a plotting
function directly or use the registry-driven :func:`plot` interface.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._core import PlotConfig, PlotResult, PlotTheme, journal_size, theme_context
from ._registry import PlotDescriptor, available_plots, get_plot, plot, plot_catalog, register_plot

_LAZY: dict[str, str] = {
    # themes
    "THEMES": "themes",
    "available_themes": "themes",
    "register_theme": "themes",
    # statistics and distributions
    "line_with_uncertainty": "statistics",
    "binned_trend": "statistics",
    "agreement_plot": "statistics",
    "effect_size_forest": "statistics",
    "correlation_matrix": "statistics",
    "quantile_dotplot": "statistics",
    "raincloud_plot": "distributions",
    "ridge_plot": "distributions",
    "violin_box_scatter": "distributions",
    "empirical_cdf": "distributions",
    "histogram_density": "distributions",
    "mirrored_histogram": "distributions",
    # model evaluation
    "parity_plot": "model_evaluation",
    "residual_diagnostics": "model_evaluation",
    "confusion_matrix_plot": "model_evaluation",
    "roc_curve_plot": "model_evaluation",
    "precision_recall_plot": "model_evaluation",
    "radar_plot": "model_evaluation",
    "performance_profile": "model_evaluation",
    # uncertainty
    "calibration_plot": "uncertainty",
    "error_vs_uncertainty": "uncertainty",
    "prediction_interval_plot": "uncertainty",
    "conformal_coverage_plot": "uncertainty",
    "uncertainty_decomposition": "uncertainty",
    # explainability
    "feature_importance_plot": "explainability",
    "shap_beeswarm": "explainability",
    "attribution_waterfall": "explainability",
    "partial_dependence_plot": "explainability",
    "interaction_heatmap": "explainability",
    "attention_map": "explainability",
    # embeddings
    "embedding_scatter": "embeddings",
    "embedding_density": "embeddings",
    "trajectory_embedding": "embeddings",
    "latent_property_surface": "embeddings",
    "cluster_hulls": "embeddings",
    "embedding_stability_plot": "embeddings",
    # optimization
    "pareto_front_plot": "optimization",
    "parallel_coordinates_plot": "optimization",
    "ternary_composition_plot": "optimization",
    "active_learning_progress": "optimization",
    "acquisition_landscape": "optimization",
    "hypervolume_curve": "optimization",
    # materials
    "phase_diagram_2d": "materials",
    "convex_hull_plot": "materials",
    "band_structure_plot": "materials",
    "density_of_states_plot": "materials",
    "band_dos_plot": "materials",
    "phonon_dispersion_plot": "materials",
    "equation_of_state_plot": "materials",
    "stress_strain_plot": "materials",
    "diffraction_pattern_plot": "materials",
    "stacked_spectra": "materials",
    "elastic_polar_plot": "materials",
    # atomistic and ML potential
    "radial_distribution_plot": "atomistic",
    "mean_squared_displacement_plot": "atomistic",
    "velocity_autocorrelation_plot": "atomistic",
    "free_energy_surface": "atomistic",
    "potential_of_mean_force": "atomistic",
    "energy_conservation_plot": "atomistic",
    "thermodynamic_trace": "atomistic",
    "angular_distribution_plot": "atomistic",
    "contact_map_plot": "atomistic",
    "state_population_plot": "atomistic",
    "force_error_by_species": "atomistic",
    "mlip_parity_dashboard": "atomistic",
    # electrochemistry
    "voltage_capacity_plot": "electrochemistry",
    "differential_capacity_plot": "electrochemistry",
    "nyquist_plot": "electrochemistry",
    "bode_plot": "electrochemistry",
    "cyclic_voltammetry_plot": "electrochemistry",
    "ragone_plot": "electrochemistry",
    "cycling_performance_plot": "electrochemistry",
    "rate_capability_plot": "electrochemistry",
    "soc_temperature_map": "electrochemistry",
    # battery-specific continuum and operando plots
    "electrode_profile_plot": "battery",
    "concentration_profile_plot": "battery",
    "reaction_current_profile_plot": "battery",
    "overpotential_decomposition_plot": "battery",
    "electrode_utilization_plot": "battery",
    "capacity_fade_components_plot": "battery",
    "degradation_mode_map": "battery",
    "differential_capacity_map": "battery",
    "impedance_drt_plot": "battery",
    "lithium_plating_risk_map": "battery",
    "fast_charge_window_plot": "battery",
    "particle_utilization_histogram": "battery",
    "tortuosity_porosity_plot": "battery",
    "through_plane_transport_plot": "battery",
    "current_collector_temperature_plot": "battery",
    "cell_voltage_breakdown_plot": "battery",
    "cycle_waterfall_plot": "battery",
    "operando_map_plot": "battery",
    "electrode_phase_fraction_plot": "battery",
    "stack_pressure_performance_plot": "battery",
    # phase-field and chemo-mechanical plots
    "dendrite_morphology_plot": "phase_field",
    "phase_boundary_plot": "phase_field",
    "free_energy_landscape_plot": "phase_field",
    "chemical_potential_profile_plot": "phase_field",
    "interfacial_energy_polar_plot": "phase_field",
    "dendrite_tip_velocity_plot": "phase_field",
    "morphology_metrics_plot": "phase_field",
    "nucleation_growth_map": "phase_field",
    "stress_concentration_coupling_plot": "phase_field",
    "grain_orientation_map": "phase_field",
    "interface_curvature_plot": "phase_field",
    "order_parameter_distribution_plot": "phase_field",
    "phase_field_convergence_plot": "phase_field",
    "morphology_comparison_grid": "phase_field",
    "front_position_plot": "phase_field",
    "phase_field_energy_plot": "phase_field",
    "dendrite_branching_plot": "phase_field",
    # multiscale/full-scale simulation plots
    "scale_bridge_plot": "multiscale",
    "model_hierarchy_plot": "multiscale",
    "homogenization_comparison_plot": "multiscale",
    "representative_volume_convergence_plot": "multiscale",
    "mesh_convergence_plot": "multiscale",
    "parameter_sensitivity_tornado": "multiscale",
    "sobol_indices_plot": "multiscale",
    "uncertainty_fan_plot": "multiscale",
    "spatial_error_map": "multiscale",
    "field_profile_comparison": "multiscale",
    "scale_transfer_matrix_plot": "multiscale",
    "computational_cost_scaling_plot": "multiscale",
    "experiment_simulation_overlay": "multiscale",
    # biology
    "volcano_plot": "biology",
    "ma_plot": "biology",
    "expression_heatmap": "biology",
    "expression_dot_plot": "biology",
    "spatial_expression_plot": "biology",
    "survival_curve_plot": "biology",
    "enrichment_bubble_plot": "biology",
    "manhattan_plot": "biology",
    # fields
    "scalar_field_plot": "fields",
    "vector_field_plot": "fields",
    "orthogonal_slices_plot": "fields",
    "phase_field_montage": "fields",
    "kymograph_plot": "fields",
    "mesh_quality_plot": "fields",
    "crack_path_plot": "fields",
    # networks
    "network_plot": "networks",
    "reaction_network_plot": "networks",
    "adjacency_matrix_plot": "networks",
    "chord_diagram": "networks",
    "alluvial_plot": "networks",
    # training
    "training_history_plot": "training",
    "learning_rate_schedule_plot": "training",
    "gradient_flow_plot": "training",
    "dataset_cartography_plot": "training",
    "benchmark_heatmap": "training",
    # panels
    "PanelSpec": "panels",
    "FigureComposer": "panels",
    "compose_plots": "panels",
    "shared_legend": "panels",
    "add_panel_labels": "panels",
    "inset_zoom": "panels",
    "broken_axis_plot": "panels",
    # animation
    "AnimationResult": "animation",
    "animate_series": "animation",
    "animate_field": "animation",
    "animate_embedding": "animation",
    # interactive
    "interactive_parity_plot": "interactive",
    "interactive_embedding_plot": "interactive",
    "interactive_pareto_plot": "interactive",
    "interactive_volume_plot": "interactive",
    "interactive_sankey_plot": "interactive",
    # presets and gallery
    "mlip_validation_figure": "presets",
    "atomistic_dynamics_figure": "presets",
    "materials_discovery_figure": "presets",
    "synthetic_gallery": "gallery",
    "save_gallery": "gallery",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "PlotConfig",
    "PlotResult",
    "PlotTheme",
    "PlotDescriptor",
    "journal_size",
    "theme_context",
    "register_plot",
    "get_plot",
    "plot",
    "available_plots",
    "plot_catalog",
    *_LAZY.keys(),
]
