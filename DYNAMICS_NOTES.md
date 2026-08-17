# Dynamics integration notes

## Location

The molecular-dynamics package is installed under:

```text
src/zynnova/dynamics/
```

The stable public entry point is:

```python
import zynnova.dynamics as zd
```

## Included subsystems

- ASE-compatible classical calculator factory.
- In-process and subprocess LAMMPS adapters through ASE.
- Generic PyTorch energy/force/stress calculator.
- NVE, multiple NVT thermostats, and multiple NPT barostats.
- Atomic and variable-cell relaxation, including staged relaxation.
- Trajectory, thermodynamic CSV, metadata, atomic checkpoints, and restart.
- Divergence detection and emergency checkpointing.
- Annealing and multi-stage equilibration workflows.
- Adapters for `StructureData`, ASE `Atoms`, `PolymerRecord`, and structure files.

## Validation performed in the build workspace

- Python byte-code compilation completed successfully.
- Project test suite result: `16 passed, 3 skipped`.
- PyTorch autograd force test completed successfully.
- Ruff-compatible line-length and import organization were checked for new files.

The three skipped tests require optional runtime components that were not installed in the
build workspace: PyTorch Geometric, py3Dmol, and the compiled native structure extension.
ASE was also not installed in the build workspace, so end-to-end ASE dynamics runs were not
executed there. The implementation targets the current documented ASE APIs, and the user's
reported environment already contains ASE 3.29.0.

## Restart semantics

Atomic checkpoints preserve positions, cell, momenta, atom arrays, constraints serialized by
ASE, and the completed-step count. ASE thermostat and barostat classes do not expose one common
portable serialization contract for every extended variable, so NVT/NPT restarts are physical
continuations from the checkpointed atomic state, not guaranteed bitwise-identical continuations.
