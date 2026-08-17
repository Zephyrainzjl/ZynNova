from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...common import require_torch
from .calculator import load_jouleweave
from .model import JouleWeave

torch = require_torch()

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified as _MLIAPUnified
except (ImportError, ModuleNotFoundError):

    class _MLIAPUnified:
        """Import-time fallback; execution still requires LAMMPS with ML-IAP."""


def _atomic_numbers(elements: Sequence[str], values: Sequence[int] | None) -> tuple[int, ...]:
    if values is not None:
        numbers = tuple(int(value) for value in values)
        if len(numbers) != len(elements):
            raise ValueError("atomic_numbers and elements must have the same length")
        if any(value < 1 for value in numbers):
            raise ValueError("atomic numbers must be positive")
        return numbers
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise ImportError(
            "ASE is required to translate element symbols; alternatively pass "
            "atomic_numbers explicitly"
        ) from exc
    try:
        return tuple(int(atomic_numbers[symbol]) for symbol in elements)
    except KeyError as exc:
        raise ValueError(f"unknown chemical element: {exc.args[0]}") from exc


class _ForwardExchange(torch.autograd.Function):
    """LAMMPS ghost exchange with the adjoint reverse communication."""

    @staticmethod
    def forward(ctx: Any, values: Any, data: Any) -> Any:
        output = torch.empty_like(values)
        data.forward_exchange(values, output, values.shape[1])
        ctx.data = data
        return output

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple[Any, None]:
        output = torch.empty_like(gradient)
        ctx.data.reverse_exchange(gradient, output, gradient.shape[1])
        return output, None


def _exchange_fields(
    scalar: Any,
    vector: Any,
    tensor: Any,
    *,
    data: Any,
) -> tuple[Any, Any, Any]:
    """Exchange all irreducible Cartesian fields in one LAMMPS communication."""

    atom_count, channels = scalar.shape
    packed = torch.cat(
        (
            scalar,
            vector.reshape(atom_count, 3 * channels),
            tensor.reshape(atom_count, 9 * channels),
        ),
        dim=-1,
    )
    exchanged = _ForwardExchange.apply(packed, data)
    scalar_end = channels
    vector_end = scalar_end + 3 * channels
    return (
        exchanged[:, :scalar_end],
        exchanged[:, scalar_end:vector_end].reshape(atom_count, channels, 3),
        exchanged[:, vector_end:].reshape(atom_count, channels, 3, 3),
    )


class JouleWeaveMLIAP(_MLIAPUnified):
    """Unified ML-IAP model with differentiable pair-force and ghost exchange.

    LAMMPS owns the neighbor list and domain decomposition. JouleWeave consumes
    its directed pair graph, synchronizes node fields after every interaction,
    and returns ``dE / d(r_j-r_i)`` so LAMMPS can tally forces and virials.
    """

    def __init__(
        self,
        model: JouleWeave,
        elements: Sequence[str],
        *,
        atomic_numbers: Sequence[int] | None = None,
        fidelity: int = 0,
        spin: float = 0.0,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        if model.config.use_qeq:
            raise ValueError(
                "distributed ML-IAP cannot use JouleWeave's global QEq solve; "
                "export a checkpoint with use_qeq=False"
            )
        if not elements:
            raise ValueError("elements cannot be empty")
        if len(set(elements)) != len(elements):
            raise ValueError("elements must be unique and follow pair_coeff order")
        try:
            self._dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown torch dtype: {dtype}") from exc
        if not self._dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")

        numbers = _atomic_numbers(elements, atomic_numbers)
        if max(numbers) > model.config.max_atomic_number:
            raise ValueError("model max_atomic_number is smaller than an exported element")
        self.model = model.eval()
        self.elements = tuple(str(symbol) for symbol in elements)
        self.atomic_numbers = numbers
        self.fidelity = int(fidelity)
        self.spin = float(spin)

        # Public attributes required by LAMMPS' MLIAPUnified protocol.
        self.ndescriptors = 1
        self.element_types = list(self.elements)
        self.nelements = len(self.elements)
        self.rcutfac = 0.5 * model.total_cutoff_A
        self.nparams = sum(parameter.numel() for parameter in model.parameters())

    def _device(self, pair_vectors: Any) -> Any:
        try:
            parameter = next(self.model.parameters())
        except StopIteration:
            return pair_vectors.device
        if parameter.device != pair_vectors.device or parameter.dtype != self._dtype:
            self.model.to(device=pair_vectors.device, dtype=self._dtype)
        return pair_vectors.device

    def compute_forces(self, data: Any) -> None:
        raw_pair_vectors = torch.as_tensor(data.rij)
        device = self._device(raw_pair_vectors)
        pair_vectors = (
            raw_pair_vectors.to(device=device, dtype=self._dtype).detach().requires_grad_(True)
        )
        receiver = torch.as_tensor(data.pair_i, device=device, dtype=torch.long)
        sender = torch.as_tensor(data.pair_j, device=device, dtype=torch.long)
        edge_index = torch.stack((receiver, sender), dim=0)

        # LAMMPS exposes element indices as 1..nelements in the unified interface.
        # LAMMPS pair_coeff 将元素映射为 0 ... nelements-1。
        element_index = torch.as_tensor(
            data.elems,
            device=device,
            dtype=torch.long,
        )

        if element_index.numel() and (
            int(element_index.min().item()) < 0
            or int(element_index.max().item()) >= self.nelements
        ):
            raise ValueError("LAMMPS element indices are outside the exported map")

        lookup = torch.as_tensor(
            self.atomic_numbers,
            device=device,
            dtype=torch.long,
        )

        atomic_number = lookup[element_index]
        batch = torch.zeros(
            int(data.ntotal),
            device=device,
            dtype=torch.long,
        )

        def exchange(scalar: Any, vector: Any, tensor: Any):
            return _exchange_fields(scalar, vector, tensor, data=data)

        output = self.model.forward_edges(
            atomic_number,
            edge_index,
            pair_vectors,
            batch=batch,
            spin=torch.as_tensor([self.spin], device=device, dtype=self._dtype),
            fidelity=torch.as_tensor([self.fidelity], device=device, dtype=torch.long),
            exchange=exchange,
            allow_qeq=False,
        )
        local_atoms = torch.as_tensor(data.iatoms, device=device, dtype=torch.long)
        energy = output["atomic_energies"][local_atoms].sum()
        pair_gradient = torch.autograd.grad(energy, pair_vectors)[0]
        data.energy = float(energy.detach().item())
        pair_gradient = pair_gradient.detach().to(raw_pair_vectors).contiguous()
        data.update_pair_forces_gpu(pair_gradient)

    def compute_descriptors(self, data: Any) -> None:
        """No-op required by ML-IAP unified for an end-to-end potential."""
        return None

    def compute_gradients(self, data: Any) -> None:
        """No-op because compute_forces evaluates energy and pair gradients."""
        return None


def export_jouleweave_mliap(
    model: JouleWeave,
    path: str | Path,
    elements: Sequence[str],
    *,
    atomic_numbers: Sequence[int] | None = None,
    fidelity: int = 0,
    spin: float = 0.0,
    dtype: str = "float32",
) -> Path:
    """Serialize a trusted unified model for ``pair_style mliap unified``."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    adapter = JouleWeaveMLIAP(
        model,
        elements,
        atomic_numbers=atomic_numbers,
        fidelity=fidelity,
        spin=spin,
        dtype=dtype,
    )
    if destination.suffix.lower() in {".pt", ".pth"}:
        torch.save(adapter, destination)
    else:
        with destination.open("wb") as handle:
            pickle.dump(adapter, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def export_jouleweave_checkpoint_mliap(
    checkpoint: str | Path,
    path: str | Path,
    elements: Sequence[str],
    *,
    atomic_numbers: Sequence[int] | None = None,
    fidelity: int = 0,
    spin: float = 0.0,
    dtype: str = "float32",
    use_ema: bool = True,
) -> Path:
    model = load_jouleweave(
        checkpoint,
        device="cpu",
        dtype=dtype,
        use_ema=use_ema,
    )
    return export_jouleweave_mliap(
        model,
        path,
        elements,
        atomic_numbers=atomic_numbers,
        fidelity=fidelity,
        spin=spin,
        dtype=dtype,
    )


def load_jouleweave_mliap(lmp: Any, adapter: JouleWeaveMLIAP) -> None:
    """Load an in-memory adapter into an active LAMMPS library instance."""

    try:
        from lammps.mliap import load_unified
        from lammps.mliappy import activate_mliappy
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "LAMMPS must include the ML-IAP and PYTHON packages for in-memory loading"
        ) from exc
    activate_mliappy(lmp)
    load_unified(adapter)


def jouleweave_mliap_commands(
    model_file: str | Path,
    elements: Sequence[str],
    *,
    ghost_neighbors: bool = False,
) -> tuple[str, str]:
    """Return the two LAMMPS commands needed to attach a serialized model."""

    if not elements:
        raise ValueError("elements cannot be empty")
    model_path = str(Path(model_file).expanduser())
    ghost_flag = 1 if ghost_neighbors else 0
    return (
        f"pair_style mliap unified {model_path} {ghost_flag}",
        f"pair_coeff * * {' '.join(elements)}",
    )


__all__ = [
    "JouleWeaveMLIAP",
    "export_jouleweave_checkpoint_mliap",
    "export_jouleweave_mliap",
    "jouleweave_mliap_commands",
    "load_jouleweave_mliap",
]