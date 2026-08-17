# ZynNova public notebooks

These notebooks document only supported public imports. Internal source paths, 
private helpers, and implementation-only contracts are deliberately excluded.

- Public API notebooks: **344**
- End-to-end workflow notebooks: **12**

## End-to-end workflows

- [00 Installation And Discovery](workflows/00_installation_and_discovery.ipynb)
- [01 Structure Graph Roundtrip](workflows/01_structure_graph_roundtrip.ipynb)
- [02 Polymer Record And Views](workflows/02_polymer_record_and_views.ipynb)
- [03 Data Pipeline](workflows/03_data_pipeline.ipynb)
- [04 Visualization](workflows/04_visualization.ipynb)
- [05 Dynamics](workflows/05_dynamics.ipynb)
- [06 Crystal Gnn Training](workflows/06_crystal_gnn_training.ipynb)
- [07 Poly Prediction Training And Screening](workflows/07_poly_prediction_training_and_screening.ipynb)
- [08 Qm9 Flow Training And Sampling](workflows/08_qm9_flow_training_and_sampling.ipynb)
- [09 Qm9 Conditional Generation](workflows/09_qm9_conditional_generation.ipynb)
- [10 Polygen Training And Generation](workflows/10_polygen_training_and_generation.ipynb)
- [11 Znnp Training And Md](workflows/11_znnp_training_and_md.ipynb)

## API reference notebooks

### `zynnova`

- [`GraphData`](api/zynnova/GraphData.ipynb)
- [`StructureData`](api/zynnova/StructureData.ipynb)
- [`__version__`](api/zynnova/version.ipynb)

### `zynnova.data`

- [`MaterialSample`](api/zynnova/data/MaterialSample.ipynb)
- [`MaterialType`](api/zynnova/data/MaterialType.ipynb)
- [`FieldRole`](api/zynnova/data/FieldRole.ipynb)
- [`FieldLevel`](api/zynnova/data/FieldLevel.ipynb)
- [`FieldSpec`](api/zynnova/data/FieldSpec.ipynb)
- [`MissingPolicy`](api/zynnova/data/MissingPolicy.ipynb)
- [`TaskKind`](api/zynnova/data/TaskKind.ipynb)
- [`TaskSpec`](api/zynnova/data/TaskSpec.ipynb)
- [`StructureEncodingSpec`](api/zynnova/data/StructureEncodingSpec.ipynb)
- [`DatasetSource`](api/zynnova/data/DatasetSource.ipynb)
- [`FieldStatistics`](api/zynnova/data/FieldStatistics.ipynb)
- [`fit_field_statistics`](api/zynnova/data/fit_field_statistics.ipynb)
- [`standardization_pipeline`](api/zynnova/data/standardization_pipeline.ipynb)
- [`DatasetIssue`](api/zynnova/data/DatasetIssue.ipynb)
- [`DatasetReport`](api/zynnova/data/DatasetReport.ipynb)
- [`validate_dataset`](api/zynnova/data/validate_dataset.ipynb)
- [`SequenceSource`](api/zynnova/data/SequenceSource.ipynb)
- [`DatasetConfig`](api/zynnova/data/DatasetConfig.ipynb)
- [`LoaderConfig`](api/zynnova/data/LoaderConfig.ipynb)
- [`DownloadSpec`](api/zynnova/data/DownloadSpec.ipynb)
- [`DownloadManager`](api/zynnova/data/DownloadManager.ipynb)
- [`LocalDatasetInput`](api/zynnova/data/LocalDatasetInput.ipynb)
- [`CompiledSample`](api/zynnova/data/CompiledSample.ipynb)
- [`TaskCompiler`](api/zynnova/data/TaskCompiler.ipynb)
- [`compile_sample`](api/zynnova/data/compile_sample.ipynb)
- [`encode_structure`](api/zynnova/data/encode_structure.ipynb)
- [`MaterialDataset`](api/zynnova/data/MaterialDataset.ipynb)
- [`StreamingMaterialDataset`](api/zynnova/data/StreamingMaterialDataset.ipynb)
- [`MaterialDataModule`](api/zynnova/data/MaterialDataModule.ipynb)
- [`material_collate`](api/zynnova/data/material_collate.ipynb)
- [`random_split_indices`](api/zynnova/data/random_split_indices.ipynb)
- [`DataPipeline`](api/zynnova/data/DataPipeline.ipynb)
- [`PreparedDataset`](api/zynnova/data/PreparedDataset.ipynb)
- [`save_dataset`](api/zynnova/data/save_dataset.ipynb)
- [`load_dataset`](api/zynnova/data/load_dataset.ipynb)
- [`create_dataset`](api/zynnova/data/create_dataset.ipynb)
- [`dataset_class`](api/zynnova/data/dataset_class.ipynb)
- [`list_datasets`](api/zynnova/data/list_datasets.ipynb)
- [`dataset_catalog`](api/zynnova/data/dataset_catalog.ipynb)
- [`DatasetInfo`](api/zynnova/data/DatasetInfo.ipynb)
- [`make_torch_dataset`](api/zynnova/data/make_torch_dataset.ipynb)
- [`make_datamodule`](api/zynnova/data/make_datamodule.ipynb)
- [`make_dataloader`](api/zynnova/data/make_dataloader.ipynb)
- [`pipeline`](api/zynnova/data/pipeline.ipynb)
- [`Registry`](api/zynnova/data/Registry.ipynb)
- [`DATASETS`](api/zynnova/data/DATASETS.ipynb)
- [`TRANSFORMS`](api/zynnova/data/TRANSFORMS.ipynb)
- [`ENCODERS`](api/zynnova/data/ENCODERS.ipynb)
- [`STORAGE_FORMATS`](api/zynnova/data/STORAGE_FORMATS.ipynb)
- [`AddDerivedFields`](api/zynnova/data/AddDerivedFields.ipynb)
- [`CenterStructure`](api/zynnova/data/CenterStructure.ipynb)
- [`ClipField`](api/zynnova/data/ClipField.ipynb)
- [`Compose`](api/zynnova/data/Compose.ipynb)
- [`ConvertStructure`](api/zynnova/data/ConvertStructure.ipynb)
- [`DropMissing`](api/zynnova/data/DropMissing.ipynb)
- [`Filter`](api/zynnova/data/Filter.ipynb)
- [`MapField`](api/zynnova/data/MapField.ipynb)
- [`RenameFields`](api/zynnova/data/RenameFields.ipynb)
- [`SampleTransform`](api/zynnova/data/SampleTransform.ipynb)
- [`SelectFields`](api/zynnova/data/SelectFields.ipynb)
- [`StandardizeField`](api/zynnova/data/StandardizeField.ipynb)

### `zynnova.dynamics`

- [`CellMode`](api/zynnova/dynamics/CellMode.ipynb)
- [`DynamicsSession`](api/zynnova/dynamics/DynamicsSession.ipynb)
- [`Ensemble`](api/zynnova/dynamics/Ensemble.ipynb)
- [`MDConfig`](api/zynnova/dynamics/MDConfig.ipynb)
- [`OptimizerMethod`](api/zynnova/dynamics/OptimizerMethod.ipynb)
- [`OutputConfig`](api/zynnova/dynamics/OutputConfig.ipynb)
- [`RelaxationConfig`](api/zynnova/dynamics/RelaxationConfig.ipynb)
- [`RunConfig`](api/zynnova/dynamics/RunConfig.ipynb)
- [`SafetyConfig`](api/zynnova/dynamics/SafetyConfig.ipynb)
- [`TemperatureStage`](api/zynnova/dynamics/TemperatureStage.ipynb)
- [`TorchPotentialCalculator`](api/zynnova/dynamics/TorchPotentialCalculator.ipynb)
- [`VelocityConfig`](api/zynnova/dynamics/VelocityConfig.ipynb)
- [`VelocityMode`](api/zynnova/dynamics/VelocityMode.ipynb)
- [`WorkflowResult`](api/zynnova/dynamics/WorkflowResult.ipynb)
- [`anneal`](api/zynnova/dynamics/anneal.ipynb)
- [`available_ensembles`](api/zynnova/dynamics/available_ensembles.ipynb)
- [`available_optimizers`](api/zynnova/dynamics/available_optimizers.ipynb)
- [`calculator_capabilities`](api/zynnova/dynamics/calculator_capabilities.ipynb)
- [`create_classical_calculator`](api/zynnova/dynamics/create_classical_calculator.ipynb)
- [`equilibrate`](api/zynnova/dynamics/equilibrate.ipynb)
- [`fix_atoms`](api/zynnova/dynamics/fix_atoms.ipynb)
- [`fix_bonds`](api/zynnova/dynamics/fix_bonds.ipynb)
- [`iter_trajectory`](api/zynnova/dynamics/iter_trajectory.ipynb)
- [`load_trajectory`](api/zynnova/dynamics/load_trajectory.ipynb)
- [`relax`](api/zynnova/dynamics/relax.ipynb)
- [`run_md`](api/zynnova/dynamics/run_md.ipynb)
- [`staged_relax`](api/zynnova/dynamics/staged_relax.ipynb)
- [`summarize_thermo`](api/zynnova/dynamics/summarize_thermo.ipynb)
- [`write_trajectory`](api/zynnova/dynamics/write_trajectory.ipynb)
- [`ConfigurationError`](api/zynnova/dynamics/ConfigurationError.ipynb)
- [`DynamicsError`](api/zynnova/dynamics/DynamicsError.ipynb)
- [`LAMMPSLibConfig`](api/zynnova/dynamics/LAMMPSLibConfig.ipynb)
- [`MissingBackendError`](api/zynnova/dynamics/MissingBackendError.ipynb)
- [`PotentialError`](api/zynnova/dynamics/PotentialError.ipynb)
- [`RelaxationResult`](api/zynnova/dynamics/RelaxationResult.ipynb)
- [`RestartError`](api/zynnova/dynamics/RestartError.ipynb)
- [`SimulationDivergedError`](api/zynnova/dynamics/SimulationDivergedError.ipynb)
- [`SimulationResult`](api/zynnova/dynamics/SimulationResult.ipynb)
- [`ThermoSeries`](api/zynnova/dynamics/ThermoSeries.ipynb)

### `zynnova.ml`

- [`MLWorkspace`](api/zynnova/ml/MLWorkspace.ipynb)
- [`MODELS`](api/zynnova/ml/MODELS.ipynb)
- [`ModelEntry`](api/zynnova/ml/ModelEntry.ipynb)
- [`ModelRegistry`](api/zynnova/ml/ModelRegistry.ipynb)
- [`RunPaths`](api/zynnova/ml/RunPaths.ipynb)
- [`TrainingResult`](api/zynnova/ml/TrainingResult.ipynb)
- [`create_model`](api/zynnova/ml/create_model.ipynb)
- [`list_models`](api/zynnova/ml/list_models.ipynb)
- [`znnp`](api/zynnova/ml/znnp.ipynb)
- [`LAMMPSRunConfig`](api/zynnova/ml/LAMMPSRunConfig.ipynb)
- [`ZNNP`](api/zynnova/ml/ZNNP.ipynb)
- [`ZNNPConfig`](api/zynnova/ml/ZNNPConfig.ipynb)
- [`ZNNPDataConfig`](api/zynnova/ml/ZNNPDataConfig.ipynb)
- [`ZNNPLAMMPSBridge`](api/zynnova/ml/ZNNPLAMMPSBridge.ipynb)
- [`ZNNPModelConfig`](api/zynnova/ml/ZNNPModelConfig.ipynb)
- [`ZNNPTrainConfig`](api/zynnova/ml/ZNNPTrainConfig.ipynb)
- [`build_radius_graph`](api/zynnova/ml/build_radius_graph.ipynb)
- [`create_znnp`](api/zynnova/ml/create_znnp.ipynb)
- [`fit_energy_normalization`](api/zynnova/ml/fit_energy_normalization.ipynb)
- [`load_znnp`](api/zynnova/ml/load_znnp.ipynb)
- [`load_znnp_calculator`](api/zynnova/ml/load_znnp_calculator.ipynb)
- [`prepare_rmd17_datamodule`](api/zynnova/ml/prepare_rmd17_datamodule.ipynb)
- [`run_znnp_lammps`](api/zynnova/ml/run_znnp_lammps.ipynb)
- [`train_znnp`](api/zynnova/ml/train_znnp.ipynb)
- [`write_znnp_lammps_data`](api/zynnova/ml/write_znnp_lammps_data.ipynb)
- [`znnp_calculator`](api/zynnova/ml/znnp_calculator.ipynb)
- [`PolyGen`](api/zynnova/ml/PolyGen.ipynb)
- [`qm9_flow`](api/zynnova/ml/qm9_flow.ipynb)
- [`qm9_generator`](api/zynnova/ml/qm9_generator.ipynb)
- [`GeneratedPolymer`](api/zynnova/ml/GeneratedPolymer.ipynb)
- [`GenerationRepresentation`](api/zynnova/ml/GenerationRepresentation.ipynb)
- [`LoadedPolyGenerator`](api/zynnova/ml/LoadedPolyGenerator.ipynb)
- [`PAPER_IRRADIATION_MECHANISMS`](api/zynnova/ml/PAPER_IRRADIATION_MECHANISMS.ipynb)
- [`PolyGenConfig`](api/zynnova/ml/PolyGenConfig.ipynb)
- [`PolyGenDataConfig`](api/zynnova/ml/PolyGenDataConfig.ipynb)
- [`PolyGenDataModule`](api/zynnova/ml/PolyGenDataModule.ipynb)
- [`PolyGenModelConfig`](api/zynnova/ml/PolyGenModelConfig.ipynb)
- [`PolyGenSamplingConfig`](api/zynnova/ml/PolyGenSamplingConfig.ipynb)
- [`PolyGenTrainConfig`](api/zynnova/ml/PolyGenTrainConfig.ipynb)
- [`PolyGenerationResult`](api/zynnova/ml/PolyGenerationResult.ipynb)
- [`PolymerFlowDataset`](api/zynnova/ml/PolymerFlowDataset.ipynb)
- [`PolymerMaskedFlow`](api/zynnova/ml/PolymerMaskedFlow.ipynb)
- [`PolymerValidityReport`](api/zynnova/ml/PolymerValidityReport.ipynb)
- [`IrradiationMechanism`](api/zynnova/ml/IrradiationMechanism.ipynb)
- [`assess_paper_mechanisms`](api/zynnova/ml/assess_paper_mechanisms.ipynb)
- [`corrupt_discrete_flow`](api/zynnova/ml/corrupt_discrete_flow.ipynb)
- [`create_poly_gen`](api/zynnova/ml/create_poly_gen.ipynb)
- [`decode_polymer_generation_sequence`](api/zynnova/ml/decode_polymer_generation_sequence.ipynb)
- [`encode_polymer_generation_sequence`](api/zynnova/ml/encode_polymer_generation_sequence.ipynb)
- [`generate_polymers`](api/zynnova/ml/generate_polymers.ipynb)
- [`load_poly_generator`](api/zynnova/ml/load_poly_generator.ipynb)
- [`polymer_flow_collate`](api/zynnova/ml/polymer_flow_collate.ipynb)
- [`prepare_poly_gen_data`](api/zynnova/ml/prepare_poly_gen_data.ipynb)
- [`polymer_selfies_to_psmiles`](api/zynnova/ml/polymer_selfies_to_psmiles.ipynb)
- [`psmiles_to_polymer_selfies`](api/zynnova/ml/psmiles_to_polymer_selfies.ipynb)
- [`save_generation`](api/zynnova/ml/save_generation.ipynb)
- [`train_poly_gen`](api/zynnova/ml/train_poly_gen.ipynb)
- [`validate_generated_polymer`](api/zynnova/ml/validate_generated_polymer.ipynb)
- [`QM9CoordinateDataset`](api/zynnova/ml/QM9CoordinateDataset.ipynb)
- [`QM9EquivariantFlow`](api/zynnova/ml/QM9EquivariantFlow.ipynb)
- [`QM9FlowConfig`](api/zynnova/ml/QM9FlowConfig.ipynb)
- [`QM9FlowDataConfig`](api/zynnova/ml/QM9FlowDataConfig.ipynb)
- [`QM9FlowModelConfig`](api/zynnova/ml/QM9FlowModelConfig.ipynb)
- [`QM9FlowTrainConfig`](api/zynnova/ml/QM9FlowTrainConfig.ipynb)
- [`center_coordinates`](api/zynnova/ml/center_coordinates.ipynb)
- [`create_qm9_flow`](api/zynnova/ml/create_qm9_flow.ipynb)
- [`load_qm9_flow`](api/zynnova/ml/load_qm9_flow.ipynb)
- [`prepare_qm9_flow_data`](api/zynnova/ml/prepare_qm9_flow_data.ipynb)
- [`sample_qm9_coordinates`](api/zynnova/ml/sample_qm9_coordinates.ipynb)
- [`save_generated_structures`](api/zynnova/ml/save_generated_structures.ipynb)
- [`train_qm9_flow`](api/zynnova/ml/train_qm9_flow.ipynb)
- [`GeneratedMolecule`](api/zynnova/ml/GeneratedMolecule.ipynb)
- [`GeometryReport`](api/zynnova/ml/GeometryReport.ipynb)
- [`QM9ConditionalGenerator`](api/zynnova/ml/QM9ConditionalGenerator.ipynb)
- [`QM9GenerationResult`](api/zynnova/ml/QM9GenerationResult.ipynb)
- [`QM9GeneratorConfig`](api/zynnova/ml/QM9GeneratorConfig.ipynb)
- [`QM9GeneratorDataConfig`](api/zynnova/ml/QM9GeneratorDataConfig.ipynb)
- [`QM9GeneratorDataModule`](api/zynnova/ml/QM9GeneratorDataModule.ipynb)
- [`QM9GeneratorModelConfig`](api/zynnova/ml/QM9GeneratorModelConfig.ipynb)
- [`QM9GeneratorSamplingConfig`](api/zynnova/ml/QM9GeneratorSamplingConfig.ipynb)
- [`QM9GeneratorTrainConfig`](api/zynnova/ml/QM9GeneratorTrainConfig.ipynb)
- [`QM9PropertyDataset`](api/zynnova/ml/QM9PropertyDataset.ipynb)
- [`QM9PropertyNormalizer`](api/zynnova/ml/QM9PropertyNormalizer.ipynb)
- [`QM9_PROPERTY_UNITS`](api/zynnova/ml/QM9_PROPERTY_UNITS.ipynb)
- [`analyze_generated_structure`](api/zynnova/ml/analyze_generated_structure.ipynb)
- [`composition_to_atomic_numbers`](api/zynnova/ml/composition_to_atomic_numbers.ipynb)
- [`create_qm9_generator`](api/zynnova/ml/create_qm9_generator.ipynb)
- [`generate_qm9_candidates`](api/zynnova/ml/generate_qm9_candidates.ipynb)
- [`generate_qm9_molecule`](api/zynnova/ml/generate_qm9_molecule.ipynb)
- [`infer_bonds`](api/zynnova/ml/infer_bonds.ipynb)
- [`load_qm9_generator`](api/zynnova/ml/load_qm9_generator.ipynb)
- [`prepare_qm9_generator_data`](api/zynnova/ml/prepare_qm9_generator_data.ipynb)
- [`save_qm9_generation`](api/zynnova/ml/save_qm9_generation.ipynb)
- [`train_qm9_generator`](api/zynnova/ml/train_qm9_generator.ipynb)
- [`PolyPrediction`](api/zynnova/ml/PolyPrediction.ipynb)
- [`crystal_gnn`](api/zynnova/ml/crystal_gnn.ipynb)
- [`CrystalGNN`](api/zynnova/ml/CrystalGNN.ipynb)
- [`CrystalGNNConfig`](api/zynnova/ml/CrystalGNNConfig.ipynb)
- [`CrystalGNNDataConfig`](api/zynnova/ml/CrystalGNNDataConfig.ipynb)
- [`CrystalGNNModelConfig`](api/zynnova/ml/CrystalGNNModelConfig.ipynb)
- [`CrystalGNNTrainConfig`](api/zynnova/ml/CrystalGNNTrainConfig.ipynb)
- [`CrystalGraphDataset`](api/zynnova/ml/CrystalGraphDataset.ipynb)
- [`create_crystal_gnn`](api/zynnova/ml/create_crystal_gnn.ipynb)
- [`crystal_graph_collate`](api/zynnova/ml/crystal_graph_collate.ipynb)
- [`fit_target_normalization`](api/zynnova/ml/fit_target_normalization.ipynb)
- [`load_crystal_gnn`](api/zynnova/ml/load_crystal_gnn.ipynb)
- [`predict_crystal_property`](api/zynnova/ml/predict_crystal_property.ipynb)
- [`prepare_matbench_data`](api/zynnova/ml/prepare_matbench_data.ipynb)
- [`train_crystal_gnn`](api/zynnova/ml/train_crystal_gnn.ipynb)
- [`ConformalCalibrator`](api/zynnova/ml/ConformalCalibrator.ipynb)
- [`ENERGY_STORAGE_CONDITIONS`](api/zynnova/ml/ENERGY_STORAGE_CONDITIONS.ipynb)
- [`ENERGY_STORAGE_PROPERTIES`](api/zynnova/ml/ENERGY_STORAGE_PROPERTIES.ipynb)
- [`LoadedPolyPredictor`](api/zynnova/ml/LoadedPolyPredictor.ipynb)
- [`PolyPredictionConfig`](api/zynnova/ml/PolyPredictionConfig.ipynb)
- [`PolyPredictionDataConfig`](api/zynnova/ml/PolyPredictionDataConfig.ipynb)
- [`PolyPredictionDataModule`](api/zynnova/ml/PolyPredictionDataModule.ipynb)
- [`PolyPredictionModelConfig`](api/zynnova/ml/PolyPredictionModelConfig.ipynb)
- [`PolyPredictionNetwork`](api/zynnova/ml/PolyPredictionNetwork.ipynb)
- [`PolyPredictionTrainConfig`](api/zynnova/ml/PolyPredictionTrainConfig.ipynb)
- [`PolymerPrediction`](api/zynnova/ml/PolymerPrediction.ipynb)
- [`PolymerPropertyDataset`](api/zynnova/ml/PolymerPropertyDataset.ipynb)
- [`PropertyConstraint`](api/zynnova/ml/PropertyConstraint.ipynb)
- [`PropertySpec`](api/zynnova/ml/PropertySpec.ipynb)
- [`ScreenedPolymer`](api/zynnova/ml/ScreenedPolymer.ipynb)
- [`create_poly_prediction`](api/zynnova/ml/create_poly_prediction.ipynb)
- [`high_entropy_report`](api/zynnova/ml/high_entropy_report.ipynb)
- [`load_poly_predictor`](api/zynnova/ml/load_poly_predictor.ipynb)
- [`physics_consistency_loss`](api/zynnova/ml/physics_consistency_loss.ipynb)
- [`polymer_property_collate`](api/zynnova/ml/polymer_property_collate.ipynb)
- [`predict_polymer`](api/zynnova/ml/predict_polymer.ipynb)
- [`predict_polymers`](api/zynnova/ml/predict_polymers.ipynb)
- [`prepare_poly_prediction_data`](api/zynnova/ml/prepare_poly_prediction_data.ipynb)
- [`screen_predictions`](api/zynnova/ml/screen_predictions.ipynb)
- [`train_poly_prediction`](api/zynnova/ml/train_poly_prediction.ipynb)

### `zynnova.structure`

- [`GraphData`](api/zynnova/structure/GraphData.ipynb)
- [`StructureData`](api/zynnova/structure/StructureData.ipynb)

### `zynnova.structure.common`

- [`BackendName`](api/zynnova/structure/common/BackendName.ipynb)
- [`FeatureConfig`](api/zynnova/structure/common/FeatureConfig.ipynb)
- [`GraphData`](api/zynnova/structure/common/GraphData.ipynb)
- [`StructureData`](api/zynnova/structure/common/StructureData.ipynb)
- [`native_available`](api/zynnova/structure/common/native_available.ipynb)
- [`resolve_backend`](api/zynnova/structure/common/resolve_backend.ipynb)

### `zynnova.structure.crystal`

- [`graph2stru`](api/zynnova/structure/crystal/graph2stru.ipynb)
- [`simple2stru`](api/zynnova/structure/crystal/simple2stru.ipynb)
- [`stru2graph`](api/zynnova/structure/crystal/stru2graph.ipynb)
- [`stru2pyg`](api/zynnova/structure/crystal/stru2pyg.ipynb)
- [`stru2simple`](api/zynnova/structure/crystal/stru2simple.ipynb)
- [`to_graph`](api/zynnova/structure/crystal/to_graph.ipynb)

### `zynnova.structure.molecular`

- [`graph2stru`](api/zynnova/structure/molecular/graph2stru.ipynb)
- [`stru2graph`](api/zynnova/structure/molecular/stru2graph.ipynb)
- [`stru2pyg`](api/zynnova/structure/molecular/stru2pyg.ipynb)
- [`to_graph`](api/zynnova/structure/molecular/to_graph.ipynb)

### `zynnova.structure.polymer`

- [`Atom`](api/zynnova/structure/polymer/Atom.ipynb)
- [`Bond`](api/zynnova/structure/polymer/Bond.ipynb)
- [`ConnectionPort`](api/zynnova/structure/polymer/ConnectionPort.ipynb)
- [`MolecularGraph`](api/zynnova/structure/polymer/MolecularGraph.ipynb)
- [`ArchitectureType`](api/zynnova/structure/polymer/ArchitectureType.ipynb)
- [`DistributionKind`](api/zynnova/structure/polymer/DistributionKind.ipynb)
- [`EdgeKind`](api/zynnova/structure/polymer/EdgeKind.ipynb)
- [`Resolution`](api/zynnova/structure/polymer/Resolution.ipynb)
- [`UnitRole`](api/zynnova/structure/polymer/UnitRole.ipynb)
- [`ArchitectureEdge`](api/zynnova/structure/polymer/ArchitectureEdge.ipynb)
- [`ArchitectureNode`](api/zynnova/structure/polymer/ArchitectureNode.ipynb)
- [`PolymerArchitecture`](api/zynnova/structure/polymer/PolymerArchitecture.ipynb)
- [`PolymerRecord`](api/zynnova/structure/polymer/PolymerRecord.ipynb)
- [`PolymerUnit`](api/zynnova/structure/polymer/PolymerUnit.ipynb)
- [`PropertyValue`](api/zynnova/structure/polymer/PropertyValue.ipynb)
- [`Provenance`](api/zynnova/structure/polymer/Provenance.ipynb)
- [`ProcessHistory`](api/zynnova/structure/polymer/ProcessHistory.ipynb)
- [`ProcessStep`](api/zynnova/structure/polymer/ProcessStep.ipynb)
- [`PeriodicBox`](api/zynnova/structure/polymer/PeriodicBox.ipynb)
- [`SpatialFrame`](api/zynnova/structure/polymer/SpatialFrame.ipynb)
- [`SpatialState`](api/zynnova/structure/polymer/SpatialState.ipynb)
- [`Distribution`](api/zynnova/structure/polymer/Distribution.ipynb)
- [`EnsembleStatistics`](api/zynnova/structure/polymer/EnsembleStatistics.ipynb)
- [`PolymerCodec`](api/zynnova/structure/polymer/PolymerCodec.ipynb)
- [`RepresentationSchema`](api/zynnova/structure/polymer/RepresentationSchema.ipynb)
- [`ViewKind`](api/zynnova/structure/polymer/ViewKind.ipynb)
- [`make_view`](api/zynnova/structure/polymer/make_view.ipynb)
- [`record2view`](api/zynnova/structure/polymer/record2view.ipynb)
- [`stru2record`](api/zynnova/structure/polymer/stru2record.ipynb)
- [`record2stru`](api/zynnova/structure/polymer/record2stru.ipynb)
- [`stru2simple`](api/zynnova/structure/polymer/stru2simple.ipynb)
- [`simple2stru`](api/zynnova/structure/polymer/simple2stru.ipynb)
- [`stru2graph`](api/zynnova/structure/polymer/stru2graph.ipynb)
- [`stru2pyg`](api/zynnova/structure/polymer/stru2pyg.ipynb)
- [`graph2stru`](api/zynnova/structure/polymer/graph2stru.ipynb)
- [`to_graph`](api/zynnova/structure/polymer/to_graph.ipynb)
- [`view2record`](api/zynnova/structure/polymer/view2record.ipynb)
- [`view2stru`](api/zynnova/structure/polymer/view2stru.ipynb)
- [`chemical2record`](api/zynnova/structure/polymer/chemical2record.ipynb)
- [`single_chain2record`](api/zynnova/structure/polymer/single_chain2record.ipynb)
- [`multiscale2record`](api/zynnova/structure/polymer/multiscale2record.ipynb)
- [`transformer2record`](api/zynnova/structure/polymer/transformer2record.ipynb)
- [`generative2record`](api/zynnova/structure/polymer/generative2record.ipynb)
- [`ChemicalStructureView`](api/zynnova/structure/polymer/ChemicalStructureView.ipynb)
- [`SingleChainView`](api/zynnova/structure/polymer/SingleChainView.ipynb)
- [`MultiScaleView`](api/zynnova/structure/polymer/MultiScaleView.ipynb)
- [`TransformerInputView`](api/zynnova/structure/polymer/TransformerInputView.ipynb)
- [`GenerativeTensorView`](api/zynnova/structure/polymer/GenerativeTensorView.ipynb)
- [`generative_view_from_logits`](api/zynnova/structure/polymer/generative_view_from_logits.ipynb)
- [`GraphTensorView`](api/zynnova/structure/polymer/GraphTensorView.ipynb)
- [`RelationTable`](api/zynnova/structure/polymer/RelationTable.ipynb)
- [`to_chemical_structure_view`](api/zynnova/structure/polymer/to_chemical_structure_view.ipynb)
- [`to_single_chain_view`](api/zynnova/structure/polymer/to_single_chain_view.ipynb)
- [`to_multiscale_view`](api/zynnova/structure/polymer/to_multiscale_view.ipynb)
- [`to_transformer_view`](api/zynnova/structure/polymer/to_transformer_view.ipynb)
- [`to_generative_view`](api/zynnova/structure/polymer/to_generative_view.ipynb)
- [`chemical_view_to_pyg`](api/zynnova/structure/polymer/chemical_view_to_pyg.ipynb)
- [`single_chain_view_to_pyg`](api/zynnova/structure/polymer/single_chain_view_to_pyg.ipynb)
- [`single_chain_view_from_pyg`](api/zynnova/structure/polymer/single_chain_view_from_pyg.ipynb)
- [`multiscale_view_to_pyg`](api/zynnova/structure/polymer/multiscale_view_to_pyg.ipynb)
- [`generative_view_to_pyg`](api/zynnova/structure/polymer/generative_view_to_pyg.ipynb)
- [`generative_view_from_pyg`](api/zynnova/structure/polymer/generative_view_from_pyg.ipynb)
- [`view_to_pyg`](api/zynnova/structure/polymer/view_to_pyg.ipynb)
- [`pyg_to_record`](api/zynnova/structure/polymer/pyg_to_record.ipynb)
- [`collate_generative_views`](api/zynnova/structure/polymer/collate_generative_views.ipynb)
- [`collate_transformer_views`](api/zynnova/structure/polymer/collate_transformer_views.ipynb)
- [`molecular_graph_from_smiles`](api/zynnova/structure/polymer/molecular_graph_from_smiles.ipynb)
- [`architecture_to_networkx`](api/zynnova/structure/polymer/architecture_to_networkx.ipynb)
- [`DatasetAdapter`](api/zynnova/structure/polymer/DatasetAdapter.ipynb)
- [`FunctionalDatasetAdapter`](api/zynnova/structure/polymer/FunctionalDatasetAdapter.ipynb)
- [`AdapterRegistry`](api/zynnova/structure/polymer/AdapterRegistry.ipynb)
- [`DEFAULT_ADAPTER_REGISTRY`](api/zynnova/structure/polymer/DEFAULT_ADAPTER_REGISTRY.ipynb)
- [`record_to_dict`](api/zynnova/structure/polymer/record_to_dict.ipynb)
- [`record_from_dict`](api/zynnova/structure/polymer/record_from_dict.ipynb)
- [`save_json`](api/zynnova/structure/polymer/save_json.ipynb)
- [`load_json`](api/zynnova/structure/polymer/load_json.ipynb)
- [`save_zpoly`](api/zynnova/structure/polymer/save_zpoly.ipynb)
- [`load_zpoly`](api/zynnova/structure/polymer/load_zpoly.ipynb)

### `zynnova.visualization`

- [`results`](api/zynnova/visualization/results.ipynb)
- [`structure`](api/zynnova/visualization/structure.ipynb)
- [`ViewerConfig`](api/zynnova/visualization/ViewerConfig.ipynb)
- [`available_backends`](api/zynnova/visualization/available_backends.ipynb)
- [`visualize`](api/zynnova/visualization/visualize.ipynb)
- [`view`](api/zynnova/visualization/view.ipynb)
- [`visualize_structure`](api/zynnova/visualization/visualize_structure.ipynb)
- [`visualize_molecule`](api/zynnova/visualization/visualize_molecule.ipynb)
- [`visualize_polymer`](api/zynnova/visualization/visualize_polymer.ipynb)
- [`visualize_crystal`](api/zynnova/visualization/visualize_crystal.ipynb)
- [`visualize_trajectory`](api/zynnova/visualization/visualize_trajectory.ipynb)

```{toctree}
:maxdepth: 1
:hidden:

workflows/00_installation_and_discovery
workflows/01_structure_graph_roundtrip
workflows/02_polymer_record_and_views
workflows/03_data_pipeline
workflows/04_visualization
workflows/05_dynamics
workflows/06_crystal_gnn_training
workflows/07_poly_prediction_training_and_screening
workflows/08_qm9_flow_training_and_sampling
workflows/09_qm9_conditional_generation
workflows/10_polygen_training_and_generation
workflows/11_znnp_training_and_md
api/zynnova/GraphData
api/zynnova/StructureData
api/zynnova/version
api/zynnova/data/MaterialSample
api/zynnova/data/MaterialType
api/zynnova/data/FieldRole
api/zynnova/data/FieldLevel
api/zynnova/data/FieldSpec
api/zynnova/data/MissingPolicy
api/zynnova/data/TaskKind
api/zynnova/data/TaskSpec
api/zynnova/data/StructureEncodingSpec
api/zynnova/data/DatasetSource
api/zynnova/data/FieldStatistics
api/zynnova/data/fit_field_statistics
api/zynnova/data/standardization_pipeline
api/zynnova/data/DatasetIssue
api/zynnova/data/DatasetReport
api/zynnova/data/validate_dataset
api/zynnova/data/SequenceSource
api/zynnova/data/DatasetConfig
api/zynnova/data/LoaderConfig
api/zynnova/data/DownloadSpec
api/zynnova/data/DownloadManager
api/zynnova/data/LocalDatasetInput
api/zynnova/data/CompiledSample
api/zynnova/data/TaskCompiler
api/zynnova/data/compile_sample
api/zynnova/data/encode_structure
api/zynnova/data/MaterialDataset
api/zynnova/data/StreamingMaterialDataset
api/zynnova/data/MaterialDataModule
api/zynnova/data/material_collate
api/zynnova/data/random_split_indices
api/zynnova/data/DataPipeline
api/zynnova/data/PreparedDataset
api/zynnova/data/save_dataset
api/zynnova/data/load_dataset
api/zynnova/data/create_dataset
api/zynnova/data/dataset_class
api/zynnova/data/list_datasets
api/zynnova/data/dataset_catalog
api/zynnova/data/DatasetInfo
api/zynnova/data/make_torch_dataset
api/zynnova/data/make_datamodule
api/zynnova/data/make_dataloader
api/zynnova/data/pipeline
api/zynnova/data/Registry
api/zynnova/data/DATASETS
api/zynnova/data/TRANSFORMS
api/zynnova/data/ENCODERS
api/zynnova/data/STORAGE_FORMATS
api/zynnova/data/AddDerivedFields
api/zynnova/data/CenterStructure
api/zynnova/data/ClipField
api/zynnova/data/Compose
api/zynnova/data/ConvertStructure
api/zynnova/data/DropMissing
api/zynnova/data/Filter
api/zynnova/data/MapField
api/zynnova/data/RenameFields
api/zynnova/data/SampleTransform
api/zynnova/data/SelectFields
api/zynnova/data/StandardizeField
api/zynnova/dynamics/CellMode
api/zynnova/dynamics/DynamicsSession
api/zynnova/dynamics/Ensemble
api/zynnova/dynamics/MDConfig
api/zynnova/dynamics/OptimizerMethod
api/zynnova/dynamics/OutputConfig
api/zynnova/dynamics/RelaxationConfig
api/zynnova/dynamics/RunConfig
api/zynnova/dynamics/SafetyConfig
api/zynnova/dynamics/TemperatureStage
api/zynnova/dynamics/TorchPotentialCalculator
api/zynnova/dynamics/VelocityConfig
api/zynnova/dynamics/VelocityMode
api/zynnova/dynamics/WorkflowResult
api/zynnova/dynamics/anneal
api/zynnova/dynamics/available_ensembles
api/zynnova/dynamics/available_optimizers
api/zynnova/dynamics/calculator_capabilities
api/zynnova/dynamics/create_classical_calculator
api/zynnova/dynamics/equilibrate
api/zynnova/dynamics/fix_atoms
api/zynnova/dynamics/fix_bonds
api/zynnova/dynamics/iter_trajectory
api/zynnova/dynamics/load_trajectory
api/zynnova/dynamics/relax
api/zynnova/dynamics/run_md
api/zynnova/dynamics/staged_relax
api/zynnova/dynamics/summarize_thermo
api/zynnova/dynamics/write_trajectory
api/zynnova/dynamics/ConfigurationError
api/zynnova/dynamics/DynamicsError
api/zynnova/dynamics/LAMMPSLibConfig
api/zynnova/dynamics/MissingBackendError
api/zynnova/dynamics/PotentialError
api/zynnova/dynamics/RelaxationResult
api/zynnova/dynamics/RestartError
api/zynnova/dynamics/SimulationDivergedError
api/zynnova/dynamics/SimulationResult
api/zynnova/dynamics/ThermoSeries
api/zynnova/ml/MLWorkspace
api/zynnova/ml/MODELS
api/zynnova/ml/ModelEntry
api/zynnova/ml/ModelRegistry
api/zynnova/ml/RunPaths
api/zynnova/ml/TrainingResult
api/zynnova/ml/create_model
api/zynnova/ml/list_models
api/zynnova/ml/znnp
api/zynnova/ml/LAMMPSRunConfig
api/zynnova/ml/ZNNP
api/zynnova/ml/ZNNPConfig
api/zynnova/ml/ZNNPDataConfig
api/zynnova/ml/ZNNPLAMMPSBridge
api/zynnova/ml/ZNNPModelConfig
api/zynnova/ml/ZNNPTrainConfig
api/zynnova/ml/build_radius_graph
api/zynnova/ml/create_znnp
api/zynnova/ml/fit_energy_normalization
api/zynnova/ml/load_znnp
api/zynnova/ml/load_znnp_calculator
api/zynnova/ml/prepare_rmd17_datamodule
api/zynnova/ml/run_znnp_lammps
api/zynnova/ml/train_znnp
api/zynnova/ml/write_znnp_lammps_data
api/zynnova/ml/znnp_calculator
api/zynnova/ml/PolyGen
api/zynnova/ml/qm9_flow
api/zynnova/ml/qm9_generator
api/zynnova/ml/GeneratedPolymer
api/zynnova/ml/GenerationRepresentation
api/zynnova/ml/LoadedPolyGenerator
api/zynnova/ml/PAPER_IRRADIATION_MECHANISMS
api/zynnova/ml/PolyGenConfig
api/zynnova/ml/PolyGenDataConfig
api/zynnova/ml/PolyGenDataModule
api/zynnova/ml/PolyGenModelConfig
api/zynnova/ml/PolyGenSamplingConfig
api/zynnova/ml/PolyGenTrainConfig
api/zynnova/ml/PolyGenerationResult
api/zynnova/ml/PolymerFlowDataset
api/zynnova/ml/PolymerMaskedFlow
api/zynnova/ml/PolymerValidityReport
api/zynnova/ml/IrradiationMechanism
api/zynnova/ml/assess_paper_mechanisms
api/zynnova/ml/corrupt_discrete_flow
api/zynnova/ml/create_poly_gen
api/zynnova/ml/decode_polymer_generation_sequence
api/zynnova/ml/encode_polymer_generation_sequence
api/zynnova/ml/generate_polymers
api/zynnova/ml/load_poly_generator
api/zynnova/ml/polymer_flow_collate
api/zynnova/ml/prepare_poly_gen_data
api/zynnova/ml/polymer_selfies_to_psmiles
api/zynnova/ml/psmiles_to_polymer_selfies
api/zynnova/ml/save_generation
api/zynnova/ml/train_poly_gen
api/zynnova/ml/validate_generated_polymer
api/zynnova/ml/QM9CoordinateDataset
api/zynnova/ml/QM9EquivariantFlow
api/zynnova/ml/QM9FlowConfig
api/zynnova/ml/QM9FlowDataConfig
api/zynnova/ml/QM9FlowModelConfig
api/zynnova/ml/QM9FlowTrainConfig
api/zynnova/ml/center_coordinates
api/zynnova/ml/create_qm9_flow
api/zynnova/ml/load_qm9_flow
api/zynnova/ml/prepare_qm9_flow_data
api/zynnova/ml/sample_qm9_coordinates
api/zynnova/ml/save_generated_structures
api/zynnova/ml/train_qm9_flow
api/zynnova/ml/GeneratedMolecule
api/zynnova/ml/GeometryReport
api/zynnova/ml/QM9ConditionalGenerator
api/zynnova/ml/QM9GenerationResult
api/zynnova/ml/QM9GeneratorConfig
api/zynnova/ml/QM9GeneratorDataConfig
api/zynnova/ml/QM9GeneratorDataModule
api/zynnova/ml/QM9GeneratorModelConfig
api/zynnova/ml/QM9GeneratorSamplingConfig
api/zynnova/ml/QM9GeneratorTrainConfig
api/zynnova/ml/QM9PropertyDataset
api/zynnova/ml/QM9PropertyNormalizer
api/zynnova/ml/QM9_PROPERTY_UNITS
api/zynnova/ml/analyze_generated_structure
api/zynnova/ml/composition_to_atomic_numbers
api/zynnova/ml/create_qm9_generator
api/zynnova/ml/generate_qm9_candidates
api/zynnova/ml/generate_qm9_molecule
api/zynnova/ml/infer_bonds
api/zynnova/ml/load_qm9_generator
api/zynnova/ml/prepare_qm9_generator_data
api/zynnova/ml/save_qm9_generation
api/zynnova/ml/train_qm9_generator
api/zynnova/ml/PolyPrediction
api/zynnova/ml/crystal_gnn
api/zynnova/ml/CrystalGNN
api/zynnova/ml/CrystalGNNConfig
api/zynnova/ml/CrystalGNNDataConfig
api/zynnova/ml/CrystalGNNModelConfig
api/zynnova/ml/CrystalGNNTrainConfig
api/zynnova/ml/CrystalGraphDataset
api/zynnova/ml/create_crystal_gnn
api/zynnova/ml/crystal_graph_collate
api/zynnova/ml/fit_target_normalization
api/zynnova/ml/load_crystal_gnn
api/zynnova/ml/predict_crystal_property
api/zynnova/ml/prepare_matbench_data
api/zynnova/ml/train_crystal_gnn
api/zynnova/ml/ConformalCalibrator
api/zynnova/ml/ENERGY_STORAGE_CONDITIONS
api/zynnova/ml/ENERGY_STORAGE_PROPERTIES
api/zynnova/ml/LoadedPolyPredictor
api/zynnova/ml/PolyPredictionConfig
api/zynnova/ml/PolyPredictionDataConfig
api/zynnova/ml/PolyPredictionDataModule
api/zynnova/ml/PolyPredictionModelConfig
api/zynnova/ml/PolyPredictionNetwork
api/zynnova/ml/PolyPredictionTrainConfig
api/zynnova/ml/PolymerPrediction
api/zynnova/ml/PolymerPropertyDataset
api/zynnova/ml/PropertyConstraint
api/zynnova/ml/PropertySpec
api/zynnova/ml/ScreenedPolymer
api/zynnova/ml/create_poly_prediction
api/zynnova/ml/high_entropy_report
api/zynnova/ml/load_poly_predictor
api/zynnova/ml/physics_consistency_loss
api/zynnova/ml/polymer_property_collate
api/zynnova/ml/predict_polymer
api/zynnova/ml/predict_polymers
api/zynnova/ml/prepare_poly_prediction_data
api/zynnova/ml/screen_predictions
api/zynnova/ml/train_poly_prediction
api/zynnova/structure/GraphData
api/zynnova/structure/StructureData
api/zynnova/structure/common/BackendName
api/zynnova/structure/common/FeatureConfig
api/zynnova/structure/common/GraphData
api/zynnova/structure/common/StructureData
api/zynnova/structure/common/native_available
api/zynnova/structure/common/resolve_backend
api/zynnova/structure/crystal/graph2stru
api/zynnova/structure/crystal/simple2stru
api/zynnova/structure/crystal/stru2graph
api/zynnova/structure/crystal/stru2pyg
api/zynnova/structure/crystal/stru2simple
api/zynnova/structure/crystal/to_graph
api/zynnova/structure/molecular/graph2stru
api/zynnova/structure/molecular/stru2graph
api/zynnova/structure/molecular/stru2pyg
api/zynnova/structure/molecular/to_graph
api/zynnova/structure/polymer/Atom
api/zynnova/structure/polymer/Bond
api/zynnova/structure/polymer/ConnectionPort
api/zynnova/structure/polymer/MolecularGraph
api/zynnova/structure/polymer/ArchitectureType
api/zynnova/structure/polymer/DistributionKind
api/zynnova/structure/polymer/EdgeKind
api/zynnova/structure/polymer/Resolution
api/zynnova/structure/polymer/UnitRole
api/zynnova/structure/polymer/ArchitectureEdge
api/zynnova/structure/polymer/ArchitectureNode
api/zynnova/structure/polymer/PolymerArchitecture
api/zynnova/structure/polymer/PolymerRecord
api/zynnova/structure/polymer/PolymerUnit
api/zynnova/structure/polymer/PropertyValue
api/zynnova/structure/polymer/Provenance
api/zynnova/structure/polymer/ProcessHistory
api/zynnova/structure/polymer/ProcessStep
api/zynnova/structure/polymer/PeriodicBox
api/zynnova/structure/polymer/SpatialFrame
api/zynnova/structure/polymer/SpatialState
api/zynnova/structure/polymer/Distribution
api/zynnova/structure/polymer/EnsembleStatistics
api/zynnova/structure/polymer/PolymerCodec
api/zynnova/structure/polymer/RepresentationSchema
api/zynnova/structure/polymer/ViewKind
api/zynnova/structure/polymer/make_view
api/zynnova/structure/polymer/record2view
api/zynnova/structure/polymer/stru2record
api/zynnova/structure/polymer/record2stru
api/zynnova/structure/polymer/stru2simple
api/zynnova/structure/polymer/simple2stru
api/zynnova/structure/polymer/stru2graph
api/zynnova/structure/polymer/stru2pyg
api/zynnova/structure/polymer/graph2stru
api/zynnova/structure/polymer/to_graph
api/zynnova/structure/polymer/view2record
api/zynnova/structure/polymer/view2stru
api/zynnova/structure/polymer/chemical2record
api/zynnova/structure/polymer/single_chain2record
api/zynnova/structure/polymer/multiscale2record
api/zynnova/structure/polymer/transformer2record
api/zynnova/structure/polymer/generative2record
api/zynnova/structure/polymer/ChemicalStructureView
api/zynnova/structure/polymer/SingleChainView
api/zynnova/structure/polymer/MultiScaleView
api/zynnova/structure/polymer/TransformerInputView
api/zynnova/structure/polymer/GenerativeTensorView
api/zynnova/structure/polymer/generative_view_from_logits
api/zynnova/structure/polymer/GraphTensorView
api/zynnova/structure/polymer/RelationTable
api/zynnova/structure/polymer/to_chemical_structure_view
api/zynnova/structure/polymer/to_single_chain_view
api/zynnova/structure/polymer/to_multiscale_view
api/zynnova/structure/polymer/to_transformer_view
api/zynnova/structure/polymer/to_generative_view
api/zynnova/structure/polymer/chemical_view_to_pyg
api/zynnova/structure/polymer/single_chain_view_to_pyg
api/zynnova/structure/polymer/single_chain_view_from_pyg
api/zynnova/structure/polymer/multiscale_view_to_pyg
api/zynnova/structure/polymer/generative_view_to_pyg
api/zynnova/structure/polymer/generative_view_from_pyg
api/zynnova/structure/polymer/view_to_pyg
api/zynnova/structure/polymer/pyg_to_record
api/zynnova/structure/polymer/collate_generative_views
api/zynnova/structure/polymer/collate_transformer_views
api/zynnova/structure/polymer/molecular_graph_from_smiles
api/zynnova/structure/polymer/architecture_to_networkx
api/zynnova/structure/polymer/DatasetAdapter
api/zynnova/structure/polymer/FunctionalDatasetAdapter
api/zynnova/structure/polymer/AdapterRegistry
api/zynnova/structure/polymer/DEFAULT_ADAPTER_REGISTRY
api/zynnova/structure/polymer/record_to_dict
api/zynnova/structure/polymer/record_from_dict
api/zynnova/structure/polymer/save_json
api/zynnova/structure/polymer/load_json
api/zynnova/structure/polymer/save_zpoly
api/zynnova/structure/polymer/load_zpoly
api/zynnova/visualization/results
api/zynnova/visualization/structure
api/zynnova/visualization/ViewerConfig
api/zynnova/visualization/available_backends
api/zynnova/visualization/visualize
api/zynnova/visualization/view
api/zynnova/visualization/visualize_structure
api/zynnova/visualization/visualize_molecule
api/zynnova/visualization/visualize_polymer
api/zynnova/visualization/visualize_crystal
api/zynnova/visualization/visualize_trajectory
```
