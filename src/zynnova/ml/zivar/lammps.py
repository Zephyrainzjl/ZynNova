"""MPI-correct *reference/development* LAMMPS coupling for ZIVAR.

The optional long-range electronic model is intrinsically global and is therefore
served through LAMMPS's public ``fix external pf/callback`` interface. Native
MLIAP export is deliberately limited to the local upstream backbone.  This module
does not claim that the Python callback is a native CUDA/Kokkos implementation:
it is the correctness reference against which such an implementation is tested.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._deps import require_torch
from .calculator import zivar_calculator
from .checkpoint import load_zivar

torch = require_torch()


DEFAULT_LAMMPS_BINARY = str(
    Path("~/software/lammps-mliap-gpu-nompi/bin/lmp").expanduser()
)
REFERENCE_BACKEND = "python_fix_external_reference_development"


@dataclass(frozen=True, slots=True)
class LAMMPSConfig:
    fix_id: str = "zivar"
    device: str = "auto"
    dtype: str = "float32"
    boundary: str = "p p p"
    total_charge: float = 0.0
    closed_region_charge: float | None = None
    electrode_potential: float | None = None
    electrode_potential_by_type: tuple[float, ...] | None = None
    reservoir_atom_types: tuple[int, ...] = ()
    external_electric_field: tuple[float, float, float] | None = None
    external_magnetic_field: tuple[float, float, float] | None = None
    write_lammps_charges: bool = False
    compute_virial: bool = True
    require_electronic_validity: bool = True
    require_release_evidence: bool = True
    evaluation_mode: str = "replicated"
    spin_input_mode: str = "lammps"
    spin_vectors_by_type: tuple[tuple[float, float, float], ...] | None = None
    spin_evolution: str = "frozen"
    lattice_integrator: str = "nve"
    # An executable, rather than the separately installed ``lammps`` Python
    # package, is the auditable identity of a LAMMPS deployment.  ``None`` is
    # permitted for an in-process development callback, but fail-closed export
    # with release evidence requires an executable.
    lammps_binary: str | None = DEFAULT_LAMMPS_BINARY

    def __post_init__(self) -> None:
        if not self.fix_id or any(value.isspace() for value in self.fix_id):
            raise ValueError("fix_id must be one token")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        scalars = (self.total_charge, self.closed_region_charge, self.electrode_potential)
        if any(value is not None and not np.isfinite(value) for value in scalars):
            raise ValueError("charge and electrode scalars must be finite")
        for name, value in (
            ("external_electric_field", self.external_electric_field),
            ("external_magnetic_field", self.external_magnetic_field),
        ):
            if value is not None and (
                len(value) != 3 or not bool(np.isfinite(np.asarray(value)).all())
            ):
                raise ValueError(f"{name} requires a finite xyz triple")
        if any(int(value) < 1 for value in self.reservoir_atom_types):
            raise ValueError("reservoir_atom_types must be positive LAMMPS types")
        if self.electrode_potential_by_type is not None and not all(
            np.isfinite(value) for value in self.electrode_potential_by_type
        ):
            raise ValueError("electrode_potential_by_type must be finite")
        if self.evaluation_mode not in {"root", "replicated"}:
            raise ValueError("evaluation_mode must be root or replicated")
        if self.spin_input_mode not in {"lammps", "by_type"}:
            raise ValueError("spin_input_mode must be lammps or by_type")
        if self.spin_evolution != "frozen":
            raise ValueError(
                "fix external cannot supply LAMMPS magnetic force fm; native coupled "
                "spin evolution is therefore rejected rather than silently incorrect"
            )
        if self.lattice_integrator not in {"nve", "none"}:
            raise ValueError("lattice_integrator must be nve or none")
        if self.spin_vectors_by_type is not None:
            if any(
                len(value) != 3 or not bool(np.isfinite(value).all())
                for value in map(np.asarray, self.spin_vectors_by_type)
            ):
                raise ValueError("spin_vectors_by_type requires finite xyz triples")
        if self.spin_input_mode == "by_type" and self.spin_vectors_by_type is None:
            raise ValueError("by_type spin input requires spin_vectors_by_type")
        if self.spin_input_mode == "lammps" and self.spin_vectors_by_type is not None:
            raise ValueError(
                "spin_vectors_by_type is ambiguous when spin_input_mode='lammps'"
            )
        tokens = self.boundary.split()
        if len(tokens) != 3 or any(token not in {"p", "f", "s", "m"} for token in tokens):
            raise ValueError("boundary requires three p/f/s/m tokens")
        if self.lammps_binary is not None:
            if not str(self.lammps_binary).strip():
                raise ValueError("lammps_binary must be a nonempty path or None")
            resolved = Path(self.lammps_binary).expanduser().resolve()
            object.__setattr__(self, "lammps_binary", str(resolved))


@dataclass(frozen=True, slots=True)
class LAMMPSResult:
    timestep: int
    tags: np.ndarray
    forces: np.ndarray
    atomic_energies: np.ndarray
    charges: np.ndarray
    magmom_vectors: np.ndarray
    energy: float
    virial: np.ndarray
    electronic_residual: float
    spin_constraint_residual: float
    electronic_converged: bool
    electronic_method: str
    coulomb_energy: float
    effective_fields_T: np.ndarray
    magnetic_torques_eV: np.ndarray


def _device(value: str) -> Any:
    if value == "cuda:local":
        local_rank = int(
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
        )
        if not torch.cuda.is_available():
            raise RuntimeError("cuda:local requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        return torch.device("cuda", torch.cuda.current_device())
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def lammps_cell(
    boxlo: Sequence[float], boxhi: Sequence[float], xy: float, yz: float, xz: float
) -> np.ndarray:
    lengths = np.asarray(boxhi, dtype=float) - np.asarray(boxlo, dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0):
        raise ValueError("invalid LAMMPS bounds")
    return np.asarray(
        ((lengths[0], 0.0, 0.0), (xy, lengths[1], 0.0), (xz, yz, lengths[2])),
        dtype=float,
    )


def lammps_spin_vectors(sp: Any, nlocal: int) -> np.ndarray:
    """Convert LAMMPS ``sp=(direction_x,direction_y,direction_z,moment)``.

    ``atom_style spin`` normalizes the first three values and stores the moment
    norm separately.  Revalidating this public state keeps malformed data from
    entering the non-collinear Hamiltonian unnoticed.
    """

    values = np.asarray(sp[:nlocal], dtype=float)
    if values.ndim != 2 or values.shape != (nlocal, 4):
        raise RuntimeError("LAMMPS atom_style spin must expose an (nlocal,4) sp array")
    if not bool(np.isfinite(values).all()):
        raise RuntimeError("LAMMPS spin state contains non-finite values")
    direction_norm = np.linalg.norm(values[:, :3], axis=1)
    if np.any(direction_norm <= 1.0e-14):
        raise RuntimeError("LAMMPS spin directions must be nonzero")
    if np.any(values[:, 3] < 0.0):
        raise RuntimeError("LAMMPS magnetic moment norms must be nonnegative")
    if not np.allclose(direction_norm, 1.0, rtol=1.0e-7, atol=1.0e-10):
        raise RuntimeError("LAMMPS spin directions must be normalized")
    return values[:, :3] * values[:, 3:4]


def _virial6(gradient: Any) -> np.ndarray:
    # LAMMPS's virial/pressure sign is opposite to the tensile ASE stress sign.
    value = -gradient.detach().cpu().to(torch.float64).numpy()
    value = 0.5 * (value + value.T)
    return np.asarray(
        (value[0, 0], value[1, 1], value[2, 2], value[0, 1], value[0, 2], value[1, 2])
    )


def _dispatch(caller: Any, timestep: int, nlocal: int, tag: Any, x: Any, forces: Any) -> None:
    caller(timestep, nlocal, tag, x, forces)


class ZIVARLAMMPSCallback:
    """MPI-safe reference/development callback keyed by stable atom tags.

    The callback evaluates the complete Python model, but is not a native
    CUDA/Kokkos LAMMPS pair style and must not be benchmarked or advertised as
    one.
    """

    vector_names = (
        "total_charge", "electronic_residual", "spin_constraint_residual",
        "coulomb_energy",
    )

    def __init__(
        self,
        lmp: Any,
        model: Any,
        type_to_atomic_number: Sequence[int] | Mapping[int, int],
        *,
        config: LAMMPSConfig | None = None,
        communicator: Any | None = None,
    ) -> None:
        self.lmp, self.config = lmp, config or LAMMPSConfig()
        values = _type_map(type_to_atomic_number)
        unavailable = sorted(set(values) - set(model.atomic_numbers))
        if unavailable:
            raise ValueError(f"checkpoint lacks atomic numbers {unavailable}")
        self.type_to_atomic_number = values
        self.device, self.dtype = _device(self.config.device), getattr(torch, self.config.dtype)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        if self.model.config.spin.mode == "spin_lattice":
            if (
                self.config.spin_input_mode == "by_type"
                and len(self.config.spin_vectors_by_type or ())
                != len(self.type_to_atomic_number)
            ):
                raise ValueError("spin_vectors_by_type must match the LAMMPS type map")
        self.calculator = zivar_calculator(
            self.model,
            device=self.device,
            dtype=self.config.dtype,
            require_electronic_validity=self.config.require_electronic_validity,
        )
        self.communicator = communicator
        if self.communicator is None and callable(getattr(lmp, "get_mpi_comm", None)):
            self.communicator = lmp.get_mpi_comm()
        self.last_result: LAMMPSResult | None = None
        self._installed = False

    @property
    def rank(self) -> int:
        return 0 if self.communicator is None else int(self.communicator.Get_rank())

    @property
    def size(self) -> int:
        return 1 if self.communicator is None else int(self.communicator.Get_size())

    def install(self, *, create_fix: bool = True) -> ZIVARLAMMPSCallback:
        if create_fix:
            self.lmp.command(f"fix {self.config.fix_id} all external pf/callback 1 1")
            virial = "yes" if self.config.compute_virial else "no"
            self.lmp.command(f"fix_modify {self.config.fix_id} energy yes virial {virial}")
        self.lmp.fix_external_set_vector_length(self.config.fix_id, len(self.vector_names))
        self.lmp.set_fix_external_callback(self.config.fix_id, _dispatch, self)
        self._installed = True
        return self

    def _box(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxlo, boxhi, xy, yz, xz, periodicity, _ = self.lmp.extract_box()
        return (
            np.asarray(boxlo, dtype=float),
            lammps_cell(boxlo, boxhi, xy, yz, xz),
            np.asarray(periodicity, dtype=bool),
        )

    def _evaluate(
        self,
        timestep: int,
        gathered: list[dict[str, np.ndarray]],
        cell: np.ndarray,
        pbc: np.ndarray,
    ) -> LAMMPSResult:
        from ase import Atoms

        tags = np.concatenate([part["tags"] for part in gathered])
        atom_types = np.concatenate([part["types"] for part in gathered])
        positions = np.concatenate([part["positions"] for part in gathered])
        spins = (
            np.concatenate([part["spins"] for part in gathered])
            if all("spins" in part for part in gathered) else None
        )
        order = np.argsort(tags, kind="stable")
        tags, atom_types, positions = tags[order], atom_types[order], positions[order]
        if spins is not None:
            spins = spins[order]
        if tags.size == 0 or np.unique(tags).size != tags.size:
            raise RuntimeError("LAMMPS owned atom tags are empty or non-unique")
        if atom_types.min() < 1 or atom_types.max() > len(self.type_to_atomic_number):
            raise ValueError("LAMMPS atom type lies outside the element map")
        numbers = np.asarray(self.type_to_atomic_number)[atom_types - 1]
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        if spins is not None:
            atoms.arrays["spin_vectors"] = spins
        atoms.info["total_charge"] = self.config.total_charge
        if self.config.closed_region_charge is not None:
            atoms.info["closed_region_charge"] = self.config.closed_region_charge
        if self.config.reservoir_atom_types:
            atoms.arrays["reservoir_mask"] = np.isin(
                atom_types, np.asarray(self.config.reservoir_atom_types)
            ).astype(float)
        if (
            self.model.config.electronic.boundary_mode == "mixed"
            and not self.config.reservoir_atom_types
        ):
            raise ValueError("mixed-boundary LAMMPS deployment requires reservoir_atom_types")
        if self.config.electrode_potential_by_type is not None:
            values = np.asarray(self.config.electrode_potential_by_type, dtype=float)
            if values.shape != (len(self.type_to_atomic_number),):
                raise ValueError(
                    "electrode_potential_by_type must match the LAMMPS type map"
                )
            atoms.arrays["electrode_potential"] = values[atom_types - 1]
        elif self.config.electrode_potential is not None:
            atoms.info["electrode_potential"] = self.config.electrode_potential
        if self.config.external_electric_field is not None:
            atoms.info["external_electric_field"] = self.config.external_electric_field
        if self.config.external_magnetic_field is not None:
            atoms.info["external_magnetic_field"] = self.config.external_magnetic_field
        atoms.calc = self.calculator
        properties = ["energy", "forces", "charges", "magmom_vectors"]
        if self.model.config.spin.mode == "spin_lattice":
            properties.extend(("effective_field_T", "magnetic_torque_eV"))
        if self.config.compute_virial:
            properties.append("stress")
        self.calculator.calculate(atoms, properties=properties)
        result = self.calculator.results
        virial = (
            -np.asarray(result["stress"])[[0, 1, 2, 5, 4, 3]] * atoms.get_volume()
            if self.config.compute_virial else np.zeros(6)
        )
        # ASE order: xx yy zz yz xz xy; LAMMPS order: xx yy zz xy xz yz.
        return LAMMPSResult(
            int(timestep), tags, np.asarray(result["forces"]),
            np.asarray(result["atomic_energy"]),
            np.asarray(result["charges"]), np.asarray(result["magmom_vectors"]),
            float(result["energy"]), virial,
            float(np.max(np.asarray(result["electronic_residual"]))),
            float(np.max(np.abs(np.asarray(result["spin_constraint_residual"])))),
            bool(result["electronic_converged"]),
            str(result["electronic_method"]),
            float(np.asarray(result["coulomb_energy"]).sum()),
            np.asarray(result.get("effective_field_T", np.zeros_like(positions))),
            np.asarray(result.get("magnetic_torque_eV", np.zeros_like(positions))),
        )

    def __call__(self, timestep: int, nlocal: int, tag: Any, x: Any, fexternal: Any) -> None:
        if not self._installed:
            raise RuntimeError("callback must be installed before use")
        origin, cell, pbc = self._box()
        local_tags = np.asarray(tag[:nlocal], dtype=np.int64).copy()
        types = np.asarray(self.lmp.numpy.extract_atom("type")[:nlocal], dtype=np.int64).copy()
        payload = {
            "tags": local_tags,
            "types": types,
            "positions": np.asarray(x[:nlocal], dtype=float).copy() - origin,
        }
        if (
            self.model.config.spin.mode == "spin_lattice"
            and self.config.spin_input_mode == "lammps"
        ):
            sp = self.lmp.numpy.extract_atom("sp")
            if sp is None:
                raise RuntimeError(
                    "spin_input_mode='lammps' requires atom_style spin (or hybrid spin ...)"
                )
            payload["spins"] = lammps_spin_vectors(sp, nlocal)
        elif self.config.spin_vectors_by_type is not None:
            spin_table = np.asarray(self.config.spin_vectors_by_type, dtype=float)
            payload["spins"] = spin_table[types - 1]
        if self.communicator is None:
            gathered = [payload]
            result = self._evaluate(timestep, gathered, cell, pbc)
        elif self.config.evaluation_mode == "replicated":
            gathered = self.communicator.allgather(payload)
            result = self._evaluate(timestep, gathered, cell, pbc)
        else:
            gathered = self.communicator.gather(payload, root=0)
            result = self._evaluate(timestep, gathered, cell, pbc) if self.rank == 0 else None
            result = self.communicator.bcast(result, root=0)
        if not isinstance(result, LAMMPSResult):
            raise RuntimeError("MPI result broadcast failed")
        indices = np.searchsorted(result.tags, local_tags)
        if np.any(indices >= result.tags.size) or not np.array_equal(
            result.tags[indices], local_tags
        ):
            raise RuntimeError("atom-tag force scatter failed")
        np.asarray(fexternal[:nlocal])[:] = result.forces[indices]
        self.lmp.fix_external_set_energy_global(self.config.fix_id, result.energy)
        self.lmp.fix_external_set_virial_global(self.config.fix_id, result.virial.tolist())
        if callable(getattr(self.lmp, "fix_external_set_energy_peratom", None)):
            self.lmp.fix_external_set_energy_peratom(
                self.config.fix_id, result.atomic_energies[indices].tolist()
            )
        values = (
            float(result.charges.sum()),
            result.electronic_residual,
            result.spin_constraint_residual,
            result.coulomb_energy,
        )
        for index, value in enumerate(values, start=1):
            self.lmp.fix_external_set_vector(self.config.fix_id, index, value)
        if self.config.write_lammps_charges:
            charge = self.lmp.numpy.extract_atom("q")
            if charge is None:
                raise RuntimeError("charge writing requires a charge-capable atom_style")
            charge[:nlocal] = result.charges[indices]
        self.last_result = result


@dataclass(frozen=True, slots=True)
class LAMMPSBundle:
    directory: Path
    checkpoint: Path
    manifest: Path
    driver: Path
    input_script: Path


def _type_map(value: Sequence[int] | Mapping[int, int]) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        keys = sorted(int(key) for key in value)
        if keys != list(range(1, len(keys) + 1)):
            raise ValueError("type map keys must be consecutive from one")
        result = tuple(int(value[key]) for key in keys)
    else:
        result = tuple(int(item) for item in value)
    if not result or any(item < 1 for item in result):
        raise ValueError("type map requires positive atomic numbers")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _help_value(help_text: str, label: str) -> str | None:
    for line in help_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            _, _, value = stripped.partition(":")
            return value.strip() or None
    return None


def _installed_packages(help_text: str) -> tuple[str, ...]:
    marker = "Installed packages:"
    if marker not in help_text:
        return ()
    package_block = help_text.split(marker, 1)[1]
    package_block = package_block.split("List of individual style options", 1)[0]
    return tuple(package_block.split())


def _inspect_lammps_binary(
    binary: str | Path | None,
    *,
    required: bool,
) -> dict[str, Any]:
    """Hash and interrogate the exact LAMMPS executable used by a bundle.

    ``lmp -h`` is read-only and reports the compiled packages and accelerator
    configuration.  It is intentionally executed as a subprocess: importing
    ``lammps`` would inspect whichever shared library happens to be visible to
    Python, not necessarily the executable named by the deployment contract.
    """

    if binary is None:
        if required:
            raise RuntimeError(
                "production LAMMPS export requires LAMMPSConfig.lammps_binary; "
                "the Python fix_external reference must be bound to an audited executable"
            )
        return {
            "configured": False,
            "binary": None,
            "binary_sha256": None,
            "version": None,
            "packages": [],
            "kokkos": {"compiled": False},
        }

    path = Path(binary).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured LAMMPS executable does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"configured LAMMPS executable is not executable: {path}")

    environment = os.environ.copy()
    conda_prefix = environment.get("CONDA_PREFIX")
    if conda_prefix:
        conda_library = str(Path(conda_prefix) / "lib")
        current = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            conda_library if not current else f"{conda_library}{os.pathsep}{current}"
        )
    try:
        completed = subprocess.run(
            [str(path), "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to interrogate LAMMPS executable {path}: {error}") from error
    help_text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        diagnostic = help_text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"LAMMPS executable {path} rejected '-h' with exit code "
            f"{completed.returncode}: {diagnostic}"
        )
    banner = next(
        (
            line.strip()
            for line in help_text.splitlines()
            if line.strip().startswith("Large-scale Atomic/Molecular Massively Parallel Simulator")
        ),
        None,
    )
    if banner is None:
        raise RuntimeError(f"{path} did not emit a recognizable LAMMPS help banner")
    banner_prefix = "Large-scale Atomic/Molecular Massively Parallel Simulator - "
    version = banner[len(banner_prefix):] if banner.startswith(banner_prefix) else banner
    packages = _installed_packages(help_text)
    kokkos_api = _help_value(help_text, "KOKKOS package API")
    kokkos = {
        "compiled": "KOKKOS" in packages,
        "api": None if kokkos_api is None else kokkos_api.split(),
        "precision": _help_value(help_text, "KOKKOS package precision"),
        "view_layout": _help_value(help_text, "KOKKOS package view layout"),
        "library_version": _help_value(help_text, "Kokkos library version"),
    }
    return {
        "configured": True,
        "binary": str(path),
        "binary_sha256": _sha256(path),
        "banner": banner,
        "version": version,
        "git_info": next(
            (line.strip() for line in help_text.splitlines() if line.startswith("Git info")),
            None,
        ),
        "compiler": _help_value(help_text, "Compiler"),
        "mpi": next(
            (line.strip() for line in help_text.splitlines() if line.startswith("MPI ")),
            None,
        ),
        "packages": list(packages),
        "kokkos": kokkos,
        "fix_external_compiled": "external" in help_text.split(),
    }


def _driver_source() -> str:
    return '''"""Explicit host binding for the ZIVAR reference/development callback.

This module intentionally does not import ``lammps`` or construct a LAMMPS
instance.  A Python package named ``lammps`` may load a different shared library
than the executable hashed in ``manifest.json``.
"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_reference_callback(lmp, *, launched_binary):
    """Bind to an already-created LAMMPS object after verifying its declared binary.

    The host application is responsible for proving that ``lmp`` was created by
    the declared executable/build.  The byte hash check prevents accidentally
    selecting a different path; this callback cannot turn Python into native
    Kokkos execution.
    """
    from zynnova.ml.zivar.checkpoint import load_zivar
    from zynnova.ml.zivar.lammps import LAMMPSConfig, ZIVARLAMMPSCallback

    binary = Path(launched_binary).expanduser().resolve()
    expected = manifest["lammps"]["binary_sha256"]
    if expected is None:
        raise RuntimeError(
            "this development bundle has no bound LAMMPS executable; re-export "
            "with LAMMPSConfig.lammps_binary before installing the callback"
        )
    if not binary.is_file() or _sha256(binary) != expected:
        raise RuntimeError("launched_binary does not match manifest LAMMPS SHA-256")
    runtime = LAMMPSConfig(**manifest["runtime"])
    model = load_zivar(
        root / manifest["checkpoint"], device=runtime.device, dtype=runtime.dtype
    )
    lmp.file(str(root / "in.zivar"))
    return ZIVARLAMMPSCallback(
        lmp, model, manifest["type_to_atomic_number"], config=runtime
    ).install()


if __name__ == "__main__":
    raise SystemExit(
        "This reference/development callback binding module is not a LAMMPS launcher. "
        "It deliberately refuses to import an implicit Python LAMMPS "
        "library. Create LAMMPS from the exact manifest executable/build in a "
        "verified host and call install_reference_callback(lmp, launched_binary=...). "
        "A native full-ZIVAR Kokkos implementation is not provided by this bundle."
    )
'''


def export_zivar_lammps_bundle(
    checkpoint: str | Path,
    directory: str | Path,
    type_to_atomic_number: Sequence[int] | Mapping[int, int],
    *,
    data_file: str = "system.data",
    config: LAMMPSConfig | None = None,
    run_steps: int = 0,
) -> LAMMPSBundle:
    """Export an auditable reference bundle bound to an exact LAMMPS binary.

    Release evidence makes the checkpoint auditable; it does not promote this
    Python callback to a native production backend.  That distinction is
    machine-readable in the manifest and fail-closed in the generated driver.
    """

    runtime = config or LAMMPSConfig()
    if run_steps < 0:
        raise ValueError("run_steps must be nonnegative")
    lammps_identity = _inspect_lammps_binary(
        runtime.lammps_binary,
        required=runtime.require_release_evidence,
    )
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    model = load_zivar(source, device="cpu", dtype=runtime.dtype)
    capabilities = getattr(model, "_checkpoint_capabilities", {})
    if runtime.require_release_evidence and not capabilities.get("production_validated", False):
        raise RuntimeError(
            "production LAMMPS export requires hash-bound release evidence in the "
            "checkpoint; use require_release_evidence=False only for explicit development tests"
        )
    if runtime.require_release_evidence:
        from .maturity import source_tree_hash

        validation = getattr(model, "_release_validation", None) or {}
        current_hash = source_tree_hash(Path(__file__).resolve().parent)
        if validation.get("source_hash") != current_hash:
            raise RuntimeError(
                "checkpoint release evidence was produced by a different source tree"
            )
    type_map = _type_map(type_to_atomic_number)
    if not set(type_map).issubset(set(model.atomic_numbers)):
        raise ValueError("type map contains an element absent from the checkpoint")
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target / "model.pt"
    if source != checkpoint_path:
        shutil.copy2(source, checkpoint_path)
    manifest = {
        "schema": "zivar-lammps-reference-bundle-0.2.0",
        "deployment_class": "reference_development",
        "backend": REFERENCE_BACKEND,
        "callback_interface": "fix_external_global_electrospin",
        "native": False,
        "production_native_ready": False,
        "lammps": lammps_identity,
        "backbone": model.backbone_manifest,
        "checkpoint_capabilities": capabilities,
        "release_validation": getattr(model, "_release_validation", None),
        "release_evidence": getattr(model, "_release_evidence", None),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "type_to_atomic_number": list(type_map),
        "runtime": asdict(runtime),
        "run_steps": int(run_steps),
        "limitations": {
            "units": "metal only",
            "purpose": (
                "reference/development full-system callback for correctness and "
                "Python/LAMMPS consistency testing"
            ),
            "mpi": (
                "deterministic allgather/replicated GPU evaluation"
                if runtime.evaluation_mode == "replicated"
                else "global gather/evaluate/broadcast"
            ),
            "full_zivar_native_kokkos": False,
            "local_backbone_mliap_only": (
                "the separately exported local MLIAP artifact excludes all ZIVAR "
                "q/p/Q/m heads, SCF, and long-range electrostatics"
            ),
            "spin_integration": (
                "frozen spin state only: fix external supplies Cartesian lattice forces "
                "but has no public magnetic-force callback; coupled spin-lattice "
                "trajectories must use zivar.spin_dynamics"
            ),
        },
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    driver = target / "run_zivar_lammps.py"
    driver.write_text(_driver_source())
    input_script = target / "in.zivar"
    spin_lattice = model.config.spin.mode == "spin_lattice"
    if spin_lattice and runtime.spin_input_mode == "lammps":
        atom_style = "hybrid spin charge" if runtime.write_lammps_charges else "spin"
    else:
        atom_style = "charge" if runtime.write_lammps_charges else "atomic"
    atom_map = "atom_modify map array\n" if spin_lattice else ""
    lattice_fix = (
        "fix zivar_lattice all nve\n" if runtime.lattice_integrator == "nve" else ""
    )
    input_script.write_text(
        "# Reference/development input only.  It contains no run command and must\n"
        "# be loaded by an explicitly verified host before installing the callback.\n"
        "# It is not a native ZIVAR Kokkos input deck.\n"
        f"units metal\natom_style {atom_style}\n{atom_map}boundary {runtime.boundary}\n"
        f"read_data {data_file}\npair_style zero 1.0\npair_coeff * *\n"
        f"{lattice_fix}"
        "thermo 1\nthermo_style custom step atoms pe press vol\n"
    )
    return LAMMPSBundle(target, checkpoint_path, manifest_path, driver, input_script)


def export_local_backbone_mliap(model: Any, directory: str | Path) -> Path:
    """Call the official converter; output is explicitly local-backbone-only.

    MACE requires ML-IAP conversion on a GPU, preferably the same GPU
    architecture used for inference. The upstream model is saved on its current
    device and this function never moves or mutates the caller's model.
    """

    if not model.backbone.capabilities.local_mliap:
        raise NotImplementedError(
            f"backbone {model.backbone.kind!r} has no native local ML-IAP "
            "converter; export_zivar_lammps_bundle remains fully supported"
        )
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    correlations = model.config.backbone.correlations
    if len(set(correlations)) != 1 or correlations[0] not in {2, 3}:
        raise ValueError(
            "MACE 0.3.16 MLIAP conversion requires a uniform correlation of 2 or 3"
        )
    source = target / "local_backbone.pt"
    torch.save(model.backbone.model, source)
    parameter_dtype = next(model.backbone.model.parameters()).dtype
    if parameter_dtype == torch.float32:
        dtype_name = "float32"
    elif parameter_dtype == torch.float64:
        dtype_name = "float64"
    else:
        raise ValueError("official MLIAP conversion requires a float32 or float64 backbone")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mace.cli.create_lammps_model",
            str(source),
            "--format=mliap",
            f"--dtype={dtype_name}",
        ],
        cwd=target,
        check=True,
    )
    converted = Path(f"{source}-mliap_lammps.pt")
    if not converted.is_file():
        raise RuntimeError(f"official MLIAP converter did not create {converted.name}")
    (target / "LOCAL_BACKBONE_ONLY.txt").write_text(
        "This artifact excludes ZIVAR charge/spin heads and long-range electrostatics.\n"
    )
    return converted


__all__ = [
    "DEFAULT_LAMMPS_BINARY", "LAMMPSBundle", "LAMMPSConfig", "LAMMPSResult",
    "REFERENCE_BACKEND", "ZIVARLAMMPSCallback",
    "export_local_backbone_mliap", "export_zivar_lammps_bundle", "lammps_cell",
    "lammps_spin_vectors",
]
