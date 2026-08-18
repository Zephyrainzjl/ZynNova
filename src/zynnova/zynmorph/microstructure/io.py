"""Portable microstructure I/O used by characterization and electrode workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _labels_and_metadata(data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if hasattr(data, "labels"):
        labels = np.asarray(data.labels)
        metadata = {
            "voxel_size_m": getattr(data, "voxel_size_m", None),
            "origin_m": getattr(data, "origin_m", None),
            "phase_names": getattr(data, "phase_names", None),
            "metadata": getattr(data, "metadata", None),
        }
        return labels, metadata
    return np.asarray(data), {}


def save_microstructure(
    path: str | Path,
    microstructure: Any,
    *,
    dataset: str = "labels",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save labels to NPY/NPZ/HDF5/TIFF/legacy VTK without pickle."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels, inferred = _labels_and_metadata(microstructure)
    payload = {**inferred, **dict(metadata or {})}
    suffix = target.suffix.lower()
    if suffix == ".npy":
        np.save(target, labels, allow_pickle=False)
    elif suffix == ".npz":
        np.savez_compressed(
            target,
            labels=labels,
            metadata_json=np.asarray(json.dumps(payload, default=str, sort_keys=True)),
        )
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 export requires h5py") from exc
        with h5py.File(target, "w") as handle:
            handle.create_dataset(dataset, data=labels, compression="gzip", shuffle=True)
            handle.attrs["zynnova_metadata_json"] = json.dumps(payload, default=str, sort_keys=True)
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("TIFF export requires tifffile") from exc
        tifffile.imwrite(target, labels)
    elif suffix == ".vtk":
        _write_legacy_vtk(target, labels, payload)
    else:
        raise ValueError("supported microstructure formats are .npy/.npz/.h5/.hdf5/.tif/.tiff/.vtk")
    return target


def load_microstructure(path: str | Path, *, dataset: str = "labels") -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.load(source, allow_pickle=False)
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            key = "labels" if "labels" in data.files else data.files[0]
            return np.asarray(data[key])
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 import requires h5py") from exc
        with h5py.File(source, "r") as handle:
            return np.asarray(handle[dataset])
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("TIFF import requires tifffile") from exc
        return np.asarray(tifffile.imread(source))
    raise ValueError("supported import formats are .npy/.npz/.h5/.hdf5/.tif/.tiff")


def _write_legacy_vtk(path: Path, labels: np.ndarray, metadata: Mapping[str, Any]) -> None:
    if labels.ndim != 3:
        raise ValueError("legacy VTK structured-points export requires a 3-D array")
    # labels are z,y,x while VTK dimensions are x,y,z.
    nz, ny, nx = labels.shape
    spacing = metadata.get("voxel_size_m") or (1.0, 1.0, 1.0)
    if np.isscalar(spacing):
        dz = dy = dx = float(spacing)
    else:
        dz, dy, dx = map(float, spacing)
    origin = metadata.get("origin_m") or (0.0, 0.0, 0.0)
    oz, oy, ox = map(float, origin)
    flattened = np.asarray(labels, dtype=np.int32).ravel(order="C")
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("ZynNova microstructure\nASCII\n")
        handle.write("DATASET STRUCTURED_POINTS\n")
        handle.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        handle.write(f"ORIGIN {ox:.17g} {oy:.17g} {oz:.17g}\n")
        handle.write(f"SPACING {dx:.17g} {dy:.17g} {dz:.17g}\n")
        handle.write(f"POINT_DATA {flattened.size}\n")
        handle.write("SCALARS phase int 1\nLOOKUP_TABLE default\n")
        for value in flattened:
            handle.write(f"{int(value)}\n")



def translate_microstructure(
    microstructure: Any,
    shifts: int | tuple[int, ...],
    *,
    axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Periodically translate a 2-D/3-D microstructure without changing phases.

    This is the ZynNova-native equivalent of the periodic translation utility
    used by descriptor/reconstruction workflows. ``shifts`` follows
    :func:`numpy.roll` semantics and therefore preserves shape, phase counts,
    and periodic topology exactly.
    """

    labels, _ = _labels_and_metadata(microstructure)
    if labels.ndim not in {2, 3}:
        raise ValueError("translation requires 2-D or 3-D labels")
    if axes is None:
        if isinstance(shifts, tuple):
            if len(shifts) != labels.ndim:
                raise ValueError("tuple shifts must have one entry per spatial axis")
            axes = tuple(range(labels.ndim))
        else:
            axes = (0,)
    axes = tuple(int(axis) for axis in axes)
    if any(axis < -labels.ndim or axis >= labels.ndim for axis in axes):
        raise ValueError("translation axis is outside the microstructure dimensions")
    if isinstance(shifts, tuple):
        if len(shifts) != len(axes):
            raise ValueError("shifts and axes must have the same length")
        normalized_shifts: int | tuple[int, ...] = tuple(int(v) for v in shifts)
    else:
        if len(axes) != 1:
            raise ValueError("scalar shift requires exactly one axis")
        normalized_shifts = int(shifts)
    return np.roll(labels, shift=normalized_shifts, axis=axes)

def smooth_microstructure(
    microstructure: Any,
    *,
    strength: float = 1.0,
    periodic: bool = True,
    phase_ids: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Gaussian indicator smoothing followed by an exact discrete phase projection."""

    from scipy.ndimage import gaussian_filter

    labels, _ = _labels_and_metadata(microstructure)
    if labels.ndim not in {2, 3}:
        raise ValueError("smoothing requires 2-D or 3-D labels")
    phases = tuple(map(int, np.unique(labels))) if phase_ids is None else tuple(map(int, phase_ids))
    mode = "wrap" if periodic else "nearest"
    indicators = np.stack(
        [gaussian_filter((labels == phase).astype(np.float64), strength, mode=mode) for phase in phases],
        axis=0,
    )
    return np.asarray(phases, dtype=labels.dtype)[np.argmax(indicators, axis=0)]


def view_microstructure(microstructure: Any, *, slice_index: int | None = None, axis: int = 0):
    import matplotlib.pyplot as plt

    labels, _ = _labels_and_metadata(microstructure)
    fig, ax = plt.subplots(figsize=(7, 6))
    if labels.ndim == 2:
        image = labels
    elif labels.ndim == 3:
        index = labels.shape[axis] // 2 if slice_index is None else int(slice_index)
        image = np.take(labels, index, axis=axis)
    else:
        raise ValueError("view requires 2-D or 3-D labels")
    ax.imshow(image, origin="lower", interpolation="nearest")
    ax.set_title("ZynNova microstructure")
    fig.tight_layout()
    return fig


__all__ = [
    "load_microstructure",
    "save_microstructure",
    "smooth_microstructure",
    "translate_microstructure",
    "view_microstructure",
]
