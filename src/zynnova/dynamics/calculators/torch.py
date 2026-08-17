from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..exceptions import MissingBackendError, PotentialError

try:  # Importable without ASE so model tooling and documentation still work.
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:  # pragma: no cover - exercised only in minimal installations
    all_changes = ["positions", "numbers", "cell", "pbc", "initial_charges"]

    class Calculator:  # type: ignore[no-redef]
        implemented_properties: list[str] = []

        def __init__(self, **kwargs: Any) -> None:
            self.results: dict[str, Any] = {}
            self.atoms = kwargs.get("atoms")

        def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
            self.atoms = atoms
            self.results = {}

        def get_property(self, name, atoms=None, allow_calculation=True):
            if allow_calculation:
                self.calculate(atoms, [name], all_changes)
            return self.results[name]

        def get_potential_energy(self, atoms=None, force_consistent=False):
            return self.get_property("energy", atoms)

        def get_forces(self, atoms=None):
            return self.get_property("forces", atoms)

        def get_stress(self, atoms=None):
            return self.get_property("stress", atoms)


InputAdapter = Callable[..., Any]
OutputAdapter = Callable[[Any], Mapping[str, Any]]


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise MissingBackendError(
            "PyTorch is required for neural-network potentials; install zynnova[ml-potential]"
        ) from exc
    return torch


def default_torch_input_adapter(
    atoms: Any,
    *,
    positions: Any,
    cell: Any,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    torch = _require_torch()
    numbers = torch.as_tensor(atoms.get_atomic_numbers(), device=device, dtype=torch.long)
    pbc = torch.as_tensor(np.asarray(atoms.pbc), device=device, dtype=torch.bool)
    batch = torch.zeros(len(numbers), device=device, dtype=torch.long)
    inputs = {
        "z": numbers,
        "atomic_numbers": numbers,
        "pos": positions,
        "positions": positions,
        "cell": cell,
        "pbc": pbc,
        "batch": batch,
    }
    charges = np.asarray(atoms.get_initial_charges(), dtype=float)
    if np.any(charges):
        inputs["charges"] = torch.as_tensor(charges, device=device, dtype=dtype)
    return inputs


def default_torch_output_adapter(output: Any) -> Mapping[str, Any]:
    if isinstance(output, Mapping):
        return output
    if hasattr(output, "energy"):
        result = {"energy": output.energy}
        for name in ("forces", "stress", "atomic_energies","magmoms", "charges","dipole",):
            if hasattr(output, name):
                result[name] = getattr(output, name)
        return result
    return {"energy": output}


class TorchPotentialCalculator(Calculator):
    """ASE-compatible calculator for arbitrary PyTorch energy models.

    The default model contract is ``model(inputs: dict)``. The input dictionary
    contains aliases ``z``/``atomic_numbers`` and ``pos``/``positions`` together
    with ``cell``, ``pbc``, and ``batch``. Models may return a scalar energy or a
    mapping containing ``energy`` and optionally ``forces`` and ``stress``.
    Missing forces are computed as ``-dE/dR`` using :mod:`torch.autograd`.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: Any,
        *,
        device: str | Any = "cpu",
        dtype: str | Any = "float32",
        input_adapter: InputAdapter | None = None,
        output_adapter: OutputAdapter | None = None,
        call_style: Literal["dict", "kwargs"] = "dict",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        atomic_energies_key: str = "atomic_energies",
        extra_properties: Mapping[str, str] | None = None,
        energy_scale: float = 1.0,
        force_scale: float | None = None,
        stress_scale: float = 1.0,
        compute_forces: bool = True,
        stress_mode: Literal["model", "finite_difference", "none"] = "model",
        strain_delta: float = 1.0e-5,
        model_eval: bool = True,
        compile_model: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        torch = _require_torch()
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if not getattr(self.dtype, "is_floating_point", False):
            raise ValueError("dtype must be a floating-point torch dtype")
        if call_style not in {"dict", "kwargs"}:
            raise ValueError("call_style must be 'dict' or 'kwargs'")
        self.model = (
            model.to(device=self.device, dtype=self.dtype)
            if hasattr(model, "to")
            else model
        )
        if model_eval and hasattr(self.model, "eval"):
            self.model.eval()
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.input_adapter = input_adapter or default_torch_input_adapter
        self.output_adapter = output_adapter or default_torch_output_adapter
        self.call_style = call_style
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.atomic_energies_key = atomic_energies_key
        self.extra_properties = dict(extra_properties or {})
        self.energy_scale = float(energy_scale)
        self.force_scale = float(force_scale if force_scale is not None else energy_scale)
        self.stress_scale = float(stress_scale)
        if not np.isfinite(
            [self.energy_scale, self.force_scale, self.stress_scale]
        ).all():
            raise ValueError("energy, force and stress scales must be finite")
        self.compute_forces = bool(compute_forces)
        if stress_mode not in {"model", "finite_difference", "none"}:
            raise ValueError("stress_mode must be 'model', 'finite_difference', or 'none'")
        self.stress_mode = stress_mode
        self.implemented_properties = ["energy", "free_energy", "forces"]
        if stress_mode != "none":
            self.implemented_properties.append("stress")
        for property_name in self.extra_properties:
            if property_name not in self.implemented_properties:
                self.implemented_properties.append(property_name)

        self.strain_delta = float(strain_delta)
        if self.strain_delta <= 0:
            raise ValueError("strain_delta must be positive")

    @classmethod
    def from_torchscript(cls, path: str | Path, **kwargs: Any) -> "TorchPotentialCalculator":
        torch = _require_torch()
        device = kwargs.get("device", "cpu")
        model = torch.jit.load(str(path), map_location=device)
        return cls(model, **kwargs)

    def _call_model(self, inputs: Any) -> Mapping[str, Any]:
        if self.call_style == "dict":
            output = self.model(inputs)
        else:
            if not isinstance(inputs, Mapping):
                raise PotentialError("call_style='kwargs' requires a mapping input adapter")
            output = self.model(**inputs)
        return self.output_adapter(output)

    def _energy_tensor(self, output: Mapping[str, Any]):
        torch = _require_torch()
        if self.energy_key in output:
            energy = output[self.energy_key]
        elif self.atomic_energies_key in output:
            energy = output[self.atomic_energies_key].sum()
        else:
            raise PotentialError(
                f"Model output must contain {self.energy_key!r} or {self.atomic_energies_key!r}"
            )
        if not torch.is_tensor(energy):
            energy = torch.as_tensor(energy, device=self.device, dtype=self.dtype)
        return energy.sum()

    def _evaluate_tensors(self, atoms: Any, *, require_forces: bool) -> tuple[Any, Any, Mapping]:
        torch = _require_torch()
        positions = torch.as_tensor(
            atoms.get_positions(), device=self.device, dtype=self.dtype
        ).clone()
        positions.requires_grad_(require_forces)
        cell = torch.as_tensor(
            np.asarray(atoms.cell.array), device=self.device, dtype=self.dtype
        )
        inputs = self.input_adapter(
            atoms,
            positions=positions,
            cell=cell,
            device=self.device,
            dtype=self.dtype,
        )
        output = self._call_model(inputs)
        energy = self._energy_tensor(output)
        return positions, energy, output

    def _energy_only(self, atoms: Any) -> float:
        torch = _require_torch()
        with torch.no_grad():
            _, energy, _ = self._evaluate_tensors(atoms, require_forces=False)
        return float(energy.detach().cpu()) * self.energy_scale

    def _finite_difference_stress(self, atoms: Any) -> np.ndarray:
        if not np.any(atoms.pbc):
            raise PotentialError("finite-difference stress requires a periodic cell")
        volume = float(atoms.get_volume())
        if volume <= 0:
            raise PotentialError("finite-difference stress requires a positive cell volume")
        delta = self.strain_delta
        stress = np.zeros(6, dtype=np.float64)
        components = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        base_cell = np.asarray(atoms.cell.array, dtype=float)
        for index, (i, j) in enumerate(components):
            energies: list[float] = []
            for sign in (-1.0, 1.0):
                strain = np.zeros((3, 3), dtype=float)
                if i == j:
                    strain[i, j] = sign * delta
                else:
                    strain[i, j] = sign * delta / 2.0
                    strain[j, i] = sign * delta / 2.0
                trial = atoms.copy()
                trial.set_cell(base_cell @ (np.eye(3) + strain), scale_atoms=True)
                energies.append(self._energy_only(trial))
            stress[index] = (energies[1] - energies[0]) / (2.0 * delta * volume)
        return stress * self.stress_scale

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        torch = _require_torch()
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise PotentialError("TorchPotentialCalculator requires an atoms object")
        requested = set(properties or self.implemented_properties)
        require_forces = self.compute_forces or "forces" in requested
        with torch.enable_grad():
            positions, energy, output = self._evaluate_tensors(
                atoms, require_forces=require_forces
            )
            energy_value = float(energy.detach().cpu()) * self.energy_scale
            self.results["energy"] = energy_value
            self.results["free_energy"] = energy_value
            if self.forces_key in output:
                forces = output[self.forces_key]
            elif require_forces:
                forces = -torch.autograd.grad(
                    energy,
                    positions,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )[0]
            else:
                forces = None
            if forces is not None:
                if hasattr(forces, "detach"):
                    forces = forces.detach().cpu().numpy()
                self.results["forces"] = (
                    np.asarray(forces, dtype=np.float64) * self.force_scale
                )
            if self.stress_key in output:
                stress = output[self.stress_key]
                if hasattr(stress, "detach"):
                    stress = stress.detach().cpu().numpy()
                stress = np.asarray(stress, dtype=float)
                if stress.shape == (3, 3):
                    stress = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
                if stress.shape != (6,):
                    raise PotentialError("Model stress must have shape [6] or [3, 3]")
                self.results["stress"] = stress * self.stress_scale
            elif self.stress_mode == "finite_difference" and "stress" in requested:
                self.results["stress"] = self._finite_difference_stress(atoms)
            elif self.stress_mode == "model" and "stress" in requested:
                raise PotentialError(
                    "Stress was requested but the model did not return it; use "
                    "stress_mode='finite_difference' for a slow numerical fallback"
                )
            for property_name, output_key in self.extra_properties.items():
                if output_key not in output:
                    if property_name in requested:
                        raise PotentialError(
                            f"{property_name!r} was requested but the model output "
                            f"does not contain {output_key!r}"
                        )
                    continue

                value = output[output_key]

                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()

                self.results[property_name] = np.asarray(value)
