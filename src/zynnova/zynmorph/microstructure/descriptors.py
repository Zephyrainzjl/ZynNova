"""Descriptor plugins for 2-D/3-D, binary and multiphase microstructures.

The implementation deliberately keeps descriptor definitions independent of a
particular optimizer.  NumPy kernels are used for characterization and PyTorch
kernels are supplied for descriptors that can participate in gradient-based
reconstruction.  All kernels use periodic boundary conditions by default,
matching the statistically homogeneous RVE use case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Mapping

import numpy as np

from .registry import DESCRIPTORS, register_descriptor


@dataclass(frozen=True, slots=True)
class DescriptorDefinition:
    name: str
    numpy_fn: Callable[..., np.ndarray]
    torch_fn: Callable[..., Any] | None
    differentiable: bool
    aliases: tuple[str, ...] = ()
    default_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DescriptorResult:
    name: str
    values: np.ndarray
    differentiable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def flat(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64).reshape(-1)


def _register(definition: DescriptorDefinition) -> DescriptorDefinition:
    DESCRIPTORS.register(
        definition.name,
        definition,
        aliases=definition.aliases,
    )
    return definition


def phase_probabilities(
    data: Any,
    *,
    phase_ids: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Convert labels or phase probabilities to ``(phase, *spatial)`` form."""

    if hasattr(data, "labels"):
        data = data.labels
    array = np.asarray(data)
    if array.ndim not in {2, 3, 3, 4}:
        raise ValueError("microstructure must be a 2-D/3-D label or probability array")

    if np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        if array.ndim not in {2, 3}:
            raise ValueError("integer label fields must be 2-D or 3-D")
        phases = tuple(map(int, np.unique(array))) if phase_ids is None else tuple(map(int, phase_ids))
        missing = sorted(set(map(int, np.unique(array))) - set(phases))
        if missing:
            raise ValueError(f"phase_ids do not cover labels {missing}")
        prob = np.stack([(array == phase).astype(np.float64) for phase in phases], axis=0)
        return prob, phases

    array = np.asarray(array, dtype=np.float64)
    if array.ndim < 3:
        # Scalar grey-value microstructure -> binary indicator.
        array = np.stack((1.0 - array, array), axis=0)
    elif phase_ids is not None and array.shape[0] == len(phase_ids):
        pass
    elif array.shape[0] <= 16 and array.ndim in {3, 4}:
        # Explicit phase-first probability tensor.
        pass
    else:
        # Grey-value 2-D/3-D array.
        array = np.stack((1.0 - array, array), axis=0)
    if np.any(~np.isfinite(array)):
        raise ValueError("phase probabilities contain non-finite values")
    if np.min(array) < -1.0e-8:
        raise ValueError("phase probabilities must be non-negative")
    array = np.clip(array, 0.0, None)
    total = array.sum(axis=0, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("phase probability sum must be positive at every voxel")
    array = array / total
    phases = tuple(range(array.shape[0])) if phase_ids is None else tuple(map(int, phase_ids))
    if len(phases) != array.shape[0]:
        raise ValueError("phase_ids length does not match probability channels")
    return array, phases


def _center_crop_np(array: np.ndarray, limit_to: int | None) -> np.ndarray:
    if limit_to is None:
        return array
    slices = [slice(None)]
    for size in array.shape[1:]:
        half = min(int(limit_to), size // 2)
        center = size // 2
        slices.append(slice(center - half, center + half + 1))
    return array[tuple(slices)]


def _center_crop_torch(array: Any, limit_to: int | None) -> Any:
    if limit_to is None:
        return array
    slices = [slice(None)]
    for size in array.shape[1:]:
        half = min(int(limit_to), size // 2)
        center = size // 2
        slices.append(slice(center - half, center + half + 1))
    return array[tuple(slices)]


def _downsample_np(prob: np.ndarray) -> np.ndarray:
    spatial = prob.shape[1:]
    slices = tuple(slice(0, size - size % 2) for size in spatial)
    cropped = prob[(slice(None), *slices)]
    if any(size < 2 for size in cropped.shape[1:]):
        return prob
    shape: list[int] = [cropped.shape[0]]
    for size in cropped.shape[1:]:
        shape.extend([size // 2, 2])
    reshaped = cropped.reshape(shape)
    axes = tuple(range(2, reshaped.ndim, 2))
    return reshaped.mean(axis=axes)


def _downsample_torch(prob: Any) -> Any:
    import torch.nn.functional as F

    if prob.ndim == 3:
        return F.avg_pool2d(prob.unsqueeze(0), 2, 2).squeeze(0)
    if prob.ndim == 4:
        return F.avg_pool3d(prob.unsqueeze(0), 2, 2).squeeze(0)
    raise ValueError("probability tensor must have two or three spatial dimensions")


def _multigrid_levels_np(prob: np.ndarray, use: bool, max_levels: int | None) -> tuple[np.ndarray, ...]:
    levels = [prob]
    if not use:
        return tuple(levels)
    maximum = 16 if max_levels is None else int(max_levels)
    while len(levels) < maximum and min(levels[-1].shape[1:]) >= 8:
        next_level = _downsample_np(levels[-1])
        if next_level.shape == levels[-1].shape:
            break
        levels.append(next_level)
    return tuple(levels)


def _multigrid_levels_torch(prob: Any, use: bool, max_levels: int | None) -> tuple[Any, ...]:
    levels = [prob]
    if not use:
        return tuple(levels)
    maximum = 16 if max_levels is None else int(max_levels)
    while len(levels) < maximum and min(levels[-1].shape[1:]) >= 8:
        next_level = _downsample_torch(levels[-1])
        if tuple(next_level.shape) == tuple(levels[-1].shape):
            break
        levels.append(next_level)
    return tuple(levels)


def _pack_multigrid_np(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])


def _pack_multigrid_torch(values: list[Any]) -> Any:
    import torch

    return torch.cat([value.reshape(-1) for value in values])


def _volume_fraction_np(prob: np.ndarray, **_: Any) -> np.ndarray:
    return prob.mean(axis=tuple(range(1, prob.ndim)))


def _volume_fraction_torch(prob: Any, **_: Any) -> Any:
    return prob.mean(dim=tuple(range(1, prob.ndim)))


def _variation_np(prob: np.ndarray, periodic: bool = True, **_: Any) -> np.ndarray:
    outputs = []
    for axis in range(1, prob.ndim):
        if periodic:
            diff = prob - np.roll(prob, -1, axis=axis)
        else:
            diff = np.diff(prob, axis=axis)
        outputs.append(np.mean(np.abs(diff), axis=tuple(range(1, diff.ndim))))
    return np.stack(outputs, axis=-1)


def _variation_torch(prob: Any, periodic: bool = True, **_: Any) -> Any:
    import torch

    outputs = []
    for axis in range(1, prob.ndim):
        if periodic:
            diff = prob - torch.roll(prob, shifts=-1, dims=axis)
        else:
            left = [slice(None)] * prob.ndim
            right = [slice(None)] * prob.ndim
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            diff = prob[tuple(left)] - prob[tuple(right)]
        outputs.append(diff.abs().mean(dim=tuple(range(1, diff.ndim))))
    return torch.stack(outputs, dim=-1)


def _fft_auto_np(prob: np.ndarray, limit_to: int = 16, **_: Any) -> np.ndarray:
    spatial_axes = tuple(range(1, prob.ndim))
    n = float(np.prod(prob.shape[1:]))
    spectrum = np.fft.fftn(prob, axes=spatial_axes)
    corr = np.fft.ifftn(spectrum * np.conj(spectrum), axes=spatial_axes).real / n
    corr = np.fft.fftshift(corr, axes=spatial_axes)
    return _center_crop_np(corr, limit_to)


def _fft_auto_torch(prob: Any, limit_to: int = 16, **_: Any) -> Any:
    import torch

    spatial_axes = tuple(range(1, prob.ndim))
    n = float(np.prod(tuple(prob.shape[1:])))
    spectrum = torch.fft.fftn(prob, dim=spatial_axes)
    corr = torch.fft.ifftn(spectrum * spectrum.conj(), dim=spatial_axes).real / n
    corr = torch.fft.fftshift(corr, dim=spatial_axes)
    return _center_crop_torch(corr, limit_to)


def _phase_pairs(n_phases: int, include_self: bool = False) -> tuple[tuple[int, int], ...]:
    if include_self:
        return tuple((i, j) for i in range(n_phases) for j in range(i, n_phases))
    return tuple(combinations(range(n_phases), 2))


def _fft_cross_np(prob: np.ndarray, limit_to: int = 16, **_: Any) -> np.ndarray:
    pairs = _phase_pairs(prob.shape[0], include_self=False)
    if not pairs:
        return np.empty((0,), dtype=np.float64)
    axes = tuple(range(1, prob.ndim))
    n = float(np.prod(prob.shape[1:]))
    spectra = np.fft.fftn(prob, axes=axes)
    values = []
    for i, j in pairs:
        corr = np.fft.ifftn(spectra[i] * np.conj(spectra[j]), axes=tuple(range(prob.ndim - 1))).real / n
        corr = np.fft.fftshift(corr, axes=tuple(range(corr.ndim)))
        values.append(_center_crop_np(corr[None, ...], limit_to)[0])
    return np.stack(values, axis=0)


def _fft_cross_torch(prob: Any, limit_to: int = 16, **_: Any) -> Any:
    import torch

    pairs = _phase_pairs(prob.shape[0], include_self=False)
    if not pairs:
        return prob.new_empty((0,))
    axes = tuple(range(1, prob.ndim))
    n = float(np.prod(tuple(prob.shape[1:])))
    spectra = torch.fft.fftn(prob, dim=axes)
    values = []
    for i, j in pairs:
        corr = torch.fft.ifftn(spectra[i] * spectra[j].conj(), dim=tuple(range(prob.ndim - 1))).real / n
        corr = torch.fft.fftshift(corr, dim=tuple(range(corr.ndim)))
        values.append(_center_crop_torch(corr.unsqueeze(0), limit_to)[0])
    return torch.stack(values, dim=0)


def _axis_correlation_np(prob: np.ndarray, limit_to: int = 16, order: int = 2, **_: Any) -> np.ndarray:
    maximum = max(1, min(int(limit_to), min(prob.shape[1:]) // 2))
    outputs = []
    reduce_axes = tuple(range(1, prob.ndim))
    spatial_dimensions = prob.ndim - 1
    for axis in range(spatial_dimensions):
        axis_values = []
        for shift in range(maximum + 1):
            rolled = np.roll(prob, -shift, axis=axis + 1)
            value = prob * rolled
            if order >= 3:
                other_axis = (axis + 1) % spatial_dimensions
                value = value * np.roll(prob, -shift, axis=other_axis + 1)
            axis_values.append(value.mean(axis=reduce_axes))
        outputs.append(np.stack(axis_values, axis=-1))
    return np.stack(outputs, axis=-2)


def _axis_correlation_torch(prob: Any, limit_to: int = 16, order: int = 2, **_: Any) -> Any:
    import torch

    maximum = max(1, min(int(limit_to), min(prob.shape[1:]) // 2))
    outputs = []
    reduce_axes = tuple(range(1, prob.ndim))
    spatial_dimensions = prob.ndim - 1
    for axis in range(spatial_dimensions):
        axis_values = []
        for shift in range(maximum + 1):
            value = prob * torch.roll(prob, -shift, dims=axis + 1)
            if order >= 3:
                other_axis = (axis + 1) % spatial_dimensions
                value = value * torch.roll(prob, -shift, dims=other_axis + 1)
            axis_values.append(value.mean(dim=reduce_axes))
        outputs.append(torch.stack(axis_values, dim=-1))
    return torch.stack(outputs, dim=-2)


def _cross_axis_correlation_np(prob: np.ndarray, limit_to: int = 16, **_: Any) -> np.ndarray:
    pairs = _phase_pairs(prob.shape[0], include_self=False)
    if not pairs:
        return np.empty((0,), dtype=np.float64)
    maximum = max(1, min(int(limit_to), min(prob.shape[1:]) // 2))
    outputs = []
    for i, j in pairs:
        pair_values = []
        for axis in range(prob.ndim - 1):
            vals = []
            for shift in range(maximum + 1):
                vals.append(np.mean(prob[i] * np.roll(prob[j], -shift, axis=axis)))
            pair_values.append(vals)
        outputs.append(pair_values)
    return np.asarray(outputs, dtype=np.float64)


def _cross_axis_correlation_torch(prob: Any, limit_to: int = 16, **_: Any) -> Any:
    import torch

    pairs = _phase_pairs(prob.shape[0], include_self=False)
    if not pairs:
        return prob.new_empty((0,))
    maximum = max(1, min(int(limit_to), min(prob.shape[1:]) // 2))
    outputs = []
    for i, j in pairs:
        pair_values = []
        for axis in range(prob.ndim - 1):
            vals = []
            for shift in range(maximum + 1):
                vals.append((prob[i] * torch.roll(prob[j], -shift, dims=axis)).mean())
            pair_values.append(torch.stack(vals))
        outputs.append(torch.stack(pair_values))
    return torch.stack(outputs)


def _directions(ndim: int) -> tuple[tuple[int, ...], ...]:
    axes = []
    for axis in range(ndim):
        direction = [0] * ndim
        direction[axis] = 1
        axes.append(tuple(direction))
    if ndim >= 2:
        axes.append(tuple(1 for _ in range(ndim)))
        direction = [1] * ndim
        direction[-1] = -1
        axes.append(tuple(direction))
    return tuple(axes)


def _roll_nd_np(array: np.ndarray, direction: tuple[int, ...], step: int) -> np.ndarray:
    result = array
    for axis, delta in enumerate(direction):
        if delta:
            result = np.roll(result, -step * delta, axis=axis)
    return result


def _roll_nd_torch(array: Any, direction: tuple[int, ...], step: int) -> Any:
    import torch

    dims = []
    shifts = []
    for axis, delta in enumerate(direction):
        if delta:
            dims.append(axis)
            shifts.append(-step * delta)
    return torch.roll(array, tuple(shifts), tuple(dims)) if dims else array


def _lineal_path_np(prob: np.ndarray, limit_to: int = 16, **_: Any) -> np.ndarray:
    binary = prob >= 0.5
    maximum = max(1, min(int(limit_to), min(prob.shape[1:])))
    values = []
    for direction in _directions(prob.ndim - 1):
        per_length = []
        product = binary.copy()
        for length in range(1, maximum + 1):
            if length > 1:
                for phase in range(prob.shape[0]):
                    product[phase] &= _roll_nd_np(binary[phase], direction, length - 1)
            per_length.append(product.mean(axis=tuple(range(1, product.ndim))))
        values.append(np.stack(per_length, axis=-1))
    return np.stack(values, axis=-2).astype(np.float64)


def _lineal_path_approx_np(prob: np.ndarray, limit_to: int = 16, **_: Any) -> np.ndarray:
    maximum = max(1, min(int(limit_to), min(prob.shape[1:])))
    values = []
    for direction in _directions(prob.ndim - 1):
        per_length = []
        product = prob.copy()
        for length in range(1, maximum + 1):
            if length > 1:
                for phase in range(prob.shape[0]):
                    product[phase] = product[phase] * _roll_nd_np(prob[phase], direction, length - 1)
            per_length.append(product.mean(axis=tuple(range(1, product.ndim))))
        values.append(np.stack(per_length, axis=-1))
    return np.stack(values, axis=-2)


def _lineal_path_approx_torch(prob: Any, limit_to: int = 16, **_: Any) -> Any:
    import torch

    maximum = max(1, min(int(limit_to), min(prob.shape[1:])))
    values = []
    for direction in _directions(prob.ndim - 1):
        per_length = []
        product = prob
        for length in range(1, maximum + 1):
            if length > 1:
                phase_products = []
                for phase in range(prob.shape[0]):
                    phase_products.append(
                        product[phase] * _roll_nd_torch(prob[phase], direction, length - 1)
                    )
                product = torch.stack(phase_products, dim=0)
            per_length.append(product.mean(dim=tuple(range(1, product.ndim))))
        values.append(torch.stack(per_length, dim=-1))
    return torch.stack(values, dim=-2)


def _native_features_np(prob: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, laplace

    features = [prob]
    for sigma in (1.0, 2.0):
        smoothed = np.stack(
            [gaussian_filter(channel, sigma=sigma, mode="wrap") for channel in prob],
            axis=0,
        )
        features.append(smoothed)
    gradients = []
    for axis in range(1, prob.ndim):
        gradients.append(0.5 * (np.roll(prob, -1, axis=axis) - np.roll(prob, 1, axis=axis)))
    features.extend(gradients)
    features.append(np.stack([laplace(channel, mode="wrap") for channel in prob], axis=0))
    return np.concatenate(features, axis=0)


def _periodic_average_torch(prob: Any, kernel: int) -> Any:
    import torch.nn.functional as F

    pad = kernel // 2
    mode = "circular"
    if prob.ndim == 3:
        padded = F.pad(prob.unsqueeze(0), (pad, pad, pad, pad), mode=mode)
        return F.avg_pool2d(padded, kernel, stride=1).squeeze(0)
    if prob.ndim == 4:
        padded = F.pad(prob.unsqueeze(0), (pad, pad, pad, pad, pad, pad), mode=mode)
        return F.avg_pool3d(padded, kernel, stride=1).squeeze(0)
    raise ValueError("expected 2-D or 3-D probability field")


def _native_features_torch(prob: Any) -> Any:
    import torch

    features = [prob, _periodic_average_torch(prob, 3), _periodic_average_torch(prob, 5)]
    for axis in range(1, prob.ndim):
        features.append(0.5 * (torch.roll(prob, -1, dims=axis) - torch.roll(prob, 1, dims=axis)))
    laplace = sum(
        torch.roll(prob, 1, dims=axis) + torch.roll(prob, -1, dims=axis) - 2.0 * prob
        for axis in range(1, prob.ndim)
    )
    features.append(laplace)
    return torch.cat(features, dim=0)


def _gram_np(prob: np.ndarray, **_: Any) -> np.ndarray:
    features = _native_features_np(prob).reshape(-1, int(np.prod(prob.shape[1:])))
    return (features @ features.T) / float(features.shape[1])


def _gram_torch(prob: Any, **_: Any) -> Any:
    features = _native_features_torch(prob).reshape(-1, int(np.prod(tuple(prob.shape[1:]))))
    return (features @ features.transpose(0, 1)) / float(features.shape[1])


def _orientation_np(prob: np.ndarray, orientation_field: Any = None, **_: Any) -> np.ndarray:
    if orientation_field is None:
        # Structural orientation tensor from phase gradients, useful even when
        # crystallographic orientations are not supplied.
        field = prob[1] if prob.shape[0] > 1 else prob[0]
        grads = np.gradient(field.astype(np.float64))
        vectors = np.stack(grads, axis=-1).reshape(-1, field.ndim)
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors[norms > 1.0e-12]
        if not len(vectors):
            return np.zeros((field.ndim, field.ndim), dtype=np.float64)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.einsum("ni,nj->ij", vectors, vectors) / len(vectors)
    orientation = np.asarray(orientation_field, dtype=np.float64)
    if orientation.shape[-1] not in {2, 3, 4}:
        raise ValueError("orientation_field last dimension must be 2, 3, or 4")
    vectors = orientation.reshape(-1, orientation.shape[-1])
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 1.0e-12]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    second = np.einsum("ni,nj->ij", vectors, vectors) / max(len(vectors), 1)
    fourth = np.mean(vectors**4, axis=0)
    return np.concatenate((second.reshape(-1), fourth.reshape(-1)))


# Canonical descriptor set mirrors MCRpy's plugin surface while using modern
# NumPy/PyTorch kernels and full 2-D/3-D support.
_register(DescriptorDefinition("VolumeFractions", _volume_fraction_np, _volume_fraction_torch, True))
_register(DescriptorDefinition("Variation", _variation_np, _variation_torch, True, default_weight=100.0))
_register(DescriptorDefinition("FFTCorrelations", _fft_auto_np, _fft_auto_torch, True))
_register(DescriptorDefinition("TwoPointCorrelations", _axis_correlation_np, _axis_correlation_torch, True))
_register(DescriptorDefinition("Correlations", lambda p, **k: _axis_correlation_np(p, order=3, **k), lambda p, **k: _axis_correlation_torch(p, order=3, **k), True))
_register(DescriptorDefinition("FFTCrossCorrelations", _fft_cross_np, _fft_cross_torch, True))
_register(DescriptorDefinition("CrossCorrelations", _cross_axis_correlation_np, _cross_axis_correlation_torch, True))
_register(DescriptorDefinition("LineCorrelations", _axis_correlation_np, _axis_correlation_torch, True))
_register(DescriptorDefinition("LinealPath", _lineal_path_np, None, False))
_register(DescriptorDefinition("LinealPathApproximation", _lineal_path_approx_np, _lineal_path_approx_torch, True))
_register(DescriptorDefinition("LineLinealPathApproximation", _lineal_path_approx_np, _lineal_path_approx_torch, True))
_register(DescriptorDefinition("GramMatrices", _gram_np, _gram_torch, True))
_register(DescriptorDefinition("MultiPhaseGramMatrices", _gram_np, _gram_torch, True))
_register(DescriptorDefinition("OrientationDescriptor", _orientation_np, None, False))


def descriptor_definition(name: str) -> DescriptorDefinition:
    return DESCRIPTORS.get(name)


def compute_descriptor_numpy(
    name: str,
    prob: np.ndarray,
    *,
    limit_to: int = 16,
    periodic: bool = True,
    use_multigrid: bool = False,
    multigrid_levels: int | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> DescriptorResult:
    definition = descriptor_definition(name)
    parameters = dict(kwargs or {})
    values = []
    shapes = []
    for level_index, level in enumerate(
        _multigrid_levels_np(prob, use_multigrid, multigrid_levels)
    ):
        value = np.asarray(
            definition.numpy_fn(
                level,
                limit_to=limit_to,
                periodic=periodic,
                **parameters,
            ),
            dtype=np.float64,
        )
        values.append(value)
        shapes.append(tuple(map(int, value.shape)))
    packed = _pack_multigrid_np(values) if len(values) > 1 else values[0]
    return DescriptorResult(
        name=definition.name,
        values=packed,
        differentiable=definition.differentiable,
        metadata={
            "multigrid": bool(use_multigrid),
            "level_shapes": tuple(shapes),
            "limit_to": int(limit_to),
            **definition.metadata,
        },
    )


def compute_descriptor_torch(
    name: str,
    prob: Any,
    *,
    limit_to: int = 16,
    periodic: bool = True,
    use_multigrid: bool = False,
    multigrid_levels: int | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    definition = descriptor_definition(name)
    if definition.torch_fn is None:
        raise ValueError(f"descriptor {definition.name} is not differentiable")
    parameters = dict(kwargs or {})
    values = []
    for level in _multigrid_levels_torch(prob, use_multigrid, multigrid_levels):
        values.append(
            definition.torch_fn(
                level,
                limit_to=limit_to,
                periodic=periodic,
                **parameters,
            )
        )
    return _pack_multigrid_torch(values) if len(values) > 1 else values[0]


__all__ = [
    "DescriptorDefinition",
    "DescriptorResult",
    "compute_descriptor_numpy",
    "compute_descriptor_numpy_spatial",
    "compute_descriptor_torch",
    "compute_descriptor_torch_spatial",
    "descriptor_definition",
    "phase_probabilities",
]


def compute_descriptor_numpy_spatial(
    name: str,
    prob: np.ndarray,
    *,
    slice_mode: str = "full",
    isotropic: bool = False,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> DescriptorResult:
    """Compute a descriptor on a full field or MCR-style orthogonal 2-D slices."""

    if prob.ndim != 4 or slice_mode == "full":
        return compute_descriptor_numpy(name, prob, **kwargs)
    if slice_mode not in {"average", "sample", "sample_surface"}:
        raise ValueError(f"unsupported slice mode {slice_mode!r}")
    generator = np.random.default_rng(0) if rng is None else rng
    orientation_values = []
    template: DescriptorResult | None = None
    for spatial_axis in range(3):
        size = prob.shape[spatial_axis + 1]
        if slice_mode == "sample":
            indices = (int(generator.integers(0, size)),)
        elif slice_mode == "sample_surface":
            indices = (0,)
        else:
            indices = tuple(range(size))
        per_slice = []
        for index in indices:
            sliced = np.take(prob, index, axis=spatial_axis + 1)
            result = compute_descriptor_numpy(name, sliced, **kwargs)
            template = result
            per_slice.append(np.asarray(result.values, dtype=np.float64))
        orientation_values.append(np.mean(np.stack(per_slice, axis=0), axis=0))
    values = np.mean(np.stack(orientation_values, axis=0), axis=0) if isotropic else np.stack(orientation_values, axis=0)
    assert template is not None
    return DescriptorResult(
        name=template.name,
        values=values,
        differentiable=template.differentiable,
        metadata={
            **template.metadata,
            "slice_mode": slice_mode,
            "isotropic": bool(isotropic),
            "directional": not isotropic,
        },
    )


def compute_descriptor_torch_spatial(
    name: str,
    prob: Any,
    *,
    slice_mode: str = "full",
    isotropic: bool = False,
    sample_seed: int = 0,
    **kwargs: Any,
) -> Any:
    """Differentiable counterpart of :func:`compute_descriptor_numpy_spatial`."""

    import torch

    if prob.ndim != 4 or slice_mode == "full":
        return compute_descriptor_torch(name, prob, **kwargs)
    if slice_mode not in {"average", "sample", "sample_surface"}:
        raise ValueError(f"unsupported slice mode {slice_mode!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(sample_seed))
    orientation_values = []
    for spatial_axis in range(3):
        size = int(prob.shape[spatial_axis + 1])
        if slice_mode == "sample":
            indices = (int(torch.randint(size, (1,), generator=generator).item()),)
        elif slice_mode == "sample_surface":
            indices = (0,)
        else:
            indices = tuple(range(size))
        per_slice = []
        for index in indices:
            sliced = prob.select(spatial_axis + 1, index)
            per_slice.append(compute_descriptor_torch(name, sliced, **kwargs))
        orientation_values.append(torch.stack(per_slice, dim=0).mean(dim=0))
    stacked = torch.stack(orientation_values, dim=0)
    return stacked.mean(dim=0) if isotropic else stacked
