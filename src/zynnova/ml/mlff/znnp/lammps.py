from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...common import resolve_device
from ...workspace import MLWorkspace
from .model import ZNNP


@dataclass(slots=True)
class LAMMPSRunConfig:
    steps: int = 1000
    timestep_ps: float = 0.001
    ensemble: str = "nve"
    boundary: str = "f f f"
    temperature_K: float = 300.0
    initialize_velocities: bool = True
    velocity_seed: int = 42
    damping_ps: float = 0.1
    thermo_interval: int = 10
    dump_interval: int = 10
    neighbor_skin_A: float = 2.0
    workspace_root: str | Path | None = None
    run_name: str | None = None


class ZNNPLAMMPSBridge:
    """Serial LAMMPS ``fix external`` bridge for a ZNNP model.

    The exact reference bridge uses LAMMPS' library callback and supports NVE/NVT
    with ``units metal``. It intentionally rejects multi-rank execution because a
    raw Python callback receives processor-local atoms, while this reference model
    requires a complete global neighborhood. A future native pair style or ML-IAP
    adapter can replace this bridge without changing ZNNP checkpoints.
    """

    def __init__(
        self,
        model: ZNNP,
        *,
        atom_type_to_atomic_number: Mapping[int, int],
        device: str = "auto",
        dtype: str = "float32",
        fix_id: str = "zynnp",
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self.dtype = getattr(torch, dtype)
        self.model = model.to(self.device).eval()
        self.type_map = {int(key): int(value) for key, value in atom_type_to_atomic_number.items()}
        self.fix_id = fix_id
        self.lmp: Any | None = None

    def attach(self, lmp: Any) -> None:
        world_size = int(lmp.extract_setting("world_size"))
        if world_size != 1:
            raise RuntimeError(
                "ZNNPLAMMPSBridge is an exact serial reference bridge. Run LAMMPS "
                "with one MPI rank or implement a native neighbor-list pair style."
            )
        if not lmp.has_style("fix", "external"):
            raise RuntimeError("LAMMPS was built without fix external support")
        self.lmp = lmp
        lmp.command(f"fix {self.fix_id} all external pf/callback 1 1")
        lmp.command(f"fix_modify {self.fix_id} energy yes")
        lmp.set_fix_external_callback(self.fix_id, self._callback, self)

    @staticmethod
    def _callback(
        bridge: "ZNNPLAMMPSBridge",
        ntimestep: int,
        nlocal: int,
        tags: Any,
        positions: Any,
        external_forces: Any,
    ) -> None:
        bridge._compute_callback(ntimestep, nlocal, tags, positions, external_forces)

    def _compute_callback(
        self,
        ntimestep: int,
        nlocal: int,
        tags: Any,
        positions: Any,
        external_forces: Any,
    ) -> None:
        del ntimestep
        if self.lmp is None:
            raise RuntimeError("bridge is not attached")
        import torch

        tags_array = np.asarray(tags[:nlocal], dtype=np.int64)
        local_positions = np.asarray(positions[:nlocal], dtype=np.float64)
        local_types = np.asarray(self.lmp.numpy.extract_atom("type")[:nlocal], dtype=np.int64)
        order = np.argsort(tags_array)
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        z = np.asarray([self.type_map[int(value)] for value in local_types[order]], dtype=np.int64)
        sorted_positions = local_positions[order]
        cell, pbc = _extract_lammps_cell(self.lmp)
        position_tensor = torch.as_tensor(
            sorted_positions,
            device=self.device,
            dtype=self.dtype,
        ).requires_grad_(True)
        inputs = {
            "z": torch.as_tensor(z, device=self.device, dtype=torch.long),
            "pos": position_tensor,
            "batch": torch.zeros(len(z), device=self.device, dtype=torch.long),
            "cell": torch.as_tensor(cell[None, :, :], device=self.device, dtype=self.dtype),
            "pbc": torch.as_tensor(pbc[None, :], device=self.device, dtype=torch.bool),
        }
        with torch.enable_grad():
            output = self.model(inputs)
            energy = output["energy"].sum()
            force = -torch.autograd.grad(energy, position_tensor)[0]
        force_local = force.detach().cpu().numpy()[inverse_order]
        np.asarray(external_forces[:nlocal])[:] = force_local
        self.lmp.fix_external_set_energy_global(self.fix_id, float(energy.detach().cpu()))


def _extract_lammps_cell(lmp: Any) -> tuple[np.ndarray, np.ndarray]:
    boxlo, boxhi, xy, yz, xz, periodicity, _box_change = lmp.extract_box()
    lx, ly, lz = np.asarray(boxhi, dtype=float) - np.asarray(boxlo, dtype=float)
    cell = np.asarray(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=np.float64,
    )
    return cell, np.asarray(periodicity, dtype=bool)


def _set_lammps_masses(lmp: Any, mapping: Mapping[int, int]) -> None:
    try:
        from ase.data import atomic_masses
    except ImportError as exc:
        raise ImportError("ASE is required to assign LAMMPS atomic masses") from exc
    for atom_type, atomic_number in sorted(mapping.items()):
        z = int(atomic_number)
        if z <= 0 or z >= len(atomic_masses):
            raise ValueError(f"invalid atomic number for LAMMPS type {atom_type}: {z}")
        lmp.command(f"mass {int(atom_type)} {float(atomic_masses[z]):.12g}")


def write_znnp_lammps_data(
    structure: Any,
    path: str | Path,
    *,
    species_order: list[str] | tuple[str, ...] | None = None,
) -> dict[int, int]:
    """Write an ASE-compatible atomic LAMMPS data file and return type→Z mapping.

    ``species_order`` fixes the LAMMPS atom-type order. When omitted, chemical
    symbols are sorted by atomic number, making the mapping deterministic.
    """

    try:
        from ase.data import atomic_numbers
        from ase.io import write
    except ImportError as exc:
        raise ImportError("ASE is required; install zynnova[mlff]") from exc
    from ....dynamics.adapters import to_ase_atoms

    atoms = to_ase_atoms(structure)
    if species_order is None:
        species_order = tuple(
            sorted(set(atoms.get_chemical_symbols()), key=lambda symbol: atomic_numbers[symbol])
        )
    else:
        species_order = tuple(species_order)
    missing = set(atoms.get_chemical_symbols()) - set(species_order)
    if missing:
        raise ValueError(f"species_order is missing symbols: {sorted(missing)}")
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, atoms, format="lammps-data", atom_style="atomic", specorder=list(species_order))
    return {index + 1: int(atomic_numbers[symbol]) for index, symbol in enumerate(species_order)}


def run_znnp_lammps(
    model: ZNNP,
    data_file: str | Path,
    *,
    atom_type_to_atomic_number: Mapping[int, int],
    config: LAMMPSRunConfig | None = None,
    device: str = "auto",
):
    """Run an external ZNNP force field through the LAMMPS shared library."""

    config = config or LAMMPSRunConfig()
    try:
        from lammps import lammps
    except ImportError as exc:
        raise ImportError("LAMMPS Python bindings are required; install zynnova[lammps]") from exc
    workspace = MLWorkspace(config.workspace_root)
    run = workspace.create_run("mlff", "znnp-lammps", name=config.run_name, config=config)
    lmp = lammps()
    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("dimension 3")
    if len(config.boundary.split()) != 3:
        raise ValueError("boundary must contain three LAMMPS boundary flags")
    lmp.command(f"boundary {config.boundary}")
    lmp.command("atom_style atomic")
    lmp.command(f"read_data {Path(data_file).resolve()}")
    _set_lammps_masses(lmp, atom_type_to_atomic_number)
    lmp.command(f"pair_style zero {model.config.cutoff_A}")
    lmp.command("pair_coeff * *")
    lmp.command(f"neighbor {config.neighbor_skin_A} bin")
    lmp.command("neigh_modify every 1 delay 0 check yes")
    bridge = ZNNPLAMMPSBridge(
        model,
        atom_type_to_atomic_number=atom_type_to_atomic_number,
        device=device,
    )
    bridge.attach(lmp)
    lmp.command(f"timestep {config.timestep_ps}")
    if config.initialize_velocities:
        lmp.command(
            f"velocity all create {config.temperature_K} {config.velocity_seed} "
            "mom yes rot yes dist gaussian"
        )
    if config.ensemble.lower() == "nve":
        lmp.command("fix zyn_integrator all nve")
    elif config.ensemble.lower() == "nvt":
        lmp.command(
            "fix zyn_integrator all nvt temp "
            f"{config.temperature_K} {config.temperature_K} {config.damping_ps}"
        )
    else:
        raise ValueError("the reference LAMMPS bridge supports ensemble='nve' or 'nvt'")
    dump_path = run.root / "trajectory.lammpstrj"
    lmp.command(
        f"dump zyn_dump all custom {config.dump_interval} {dump_path} id type x y z vx vy vz"
    )
    lmp.command(f"thermo {config.thermo_interval}")
    # ZNNP does not currently provide a virial; pressure is therefore omitted.
    lmp.command("thermo_style custom step temp pe ke etotal")
    lmp.command(f"run {config.steps}")
    return {"lammps": lmp, "bridge": bridge, "run_dir": run.root, "trajectory": dump_path}


__all__ = [
    "LAMMPSRunConfig",
    "ZNNPLAMMPSBridge",
    "run_znnp_lammps",
    "write_znnp_lammps_data",
]
