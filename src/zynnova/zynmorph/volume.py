"""Validated multi-phase volume with physical spacing and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.serialization import dump_json
from ..geometry.voxel import normalize_spacing, validate_voxels
from .schema import DEFAULT_PHASE_NAMES


@dataclass(frozen=True, slots=True)
class MicrostructureVolume:
    labels: np.ndarray
    voxel_size_m: float | tuple[float, float, float] = 1.0e-7
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_names: Mapping[int, str] = field(default_factory=lambda: dict(DEFAULT_PHASE_NAMES))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = validate_voxels(self.labels)
        spacing = normalize_spacing(self.voxel_size_m)
        origin = tuple(float(item) for item in self.origin_m)
        if len(origin) != 3 or not np.all(np.isfinite(origin)):
            raise ValueError("origin_m must contain three finite values")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "voxel_size_m", spacing)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(
            self,
            "phase_names",
            {int(key): str(value) for key, value in self.phase_names.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in self.labels.shape)

    @property
    def phases(self) -> tuple[int, ...]:
        return tuple(int(item) for item in np.unique(self.labels))

    @property
    def physical_size_m(self) -> tuple[float, float, float]:
        spacing = self.voxel_size_m
        assert isinstance(spacing, tuple)
        return tuple(self.shape[axis] * spacing[axis] for axis in range(3))


    def remap_regions(
        self,
        mapping: Mapping[int, int],
        *,
        region_names: Mapping[int, str] | None = None,
        require_complete: bool = False,
    ) -> "MicrostructureVolume":
        """Return a copy whose label IDs are collapsed into material regions.

        This is deliberately applied *before* PLC extraction.  It therefore
        removes artificial interfaces between objects that are geometrically
        distinct but belong to the same FEM material (for example thirty NMC
        particles carrying thirty tracking IDs that must become one COMSOL
        domain).  Unmapped labels are preserved unless ``require_complete`` is
        requested.
        """

        normalized = {int(source): int(target) for source, target in mapping.items()}
        phases = set(self.phases)
        unknown_sources = set(normalized) - phases
        if unknown_sources:
            raise ValueError(
                "region remap contains labels absent from the volume: "
                f"{sorted(unknown_sources)}"
            )
        if require_complete:
            missing = phases - set(normalized)
            if missing:
                raise ValueError(
                    "region remap does not cover all labels: "
                    f"{sorted(missing)}"
                )

        remapped = self.labels.copy()
        original = self.labels
        for source, target in normalized.items():
            remapped[original == source] = target

        output_phases = tuple(map(int, np.unique(remapped)))
        supplied_names = {
            int(key): str(value) for key, value in (region_names or {}).items()
        }
        names: dict[int, str] = {}
        inverse: dict[int, list[int]] = {}
        for source in phases:
            target = normalized.get(source, source)
            inverse.setdefault(target, []).append(source)
        for target in output_phases:
            if target in supplied_names:
                names[target] = supplied_names[target]
                continue
            sources = inverse.get(target, [target])
            source_names = [self.phase_names.get(source) for source in sources]
            source_names = [name for name in source_names if name]
            names[target] = (
                source_names[0]
                if len(set(source_names)) == 1
                else self.phase_names.get(target, f"region_{target}")
            )

        return MicrostructureVolume(
            labels=remapped,
            voxel_size_m=self.voxel_size_m,
            origin_m=self.origin_m,
            phase_names=names,
            metadata={
                **self.metadata,
                "region_remap": normalized,
                "source_phases": tuple(sorted(phases)),
                "material_phases": output_phases,
            },
        )

    def save_npz(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            labels=self.labels,
            voxel_size_m=np.asarray(self.voxel_size_m),
            origin_m=np.asarray(self.origin_m),
            phase_ids=np.asarray(sorted(self.phase_names), dtype=np.int32),
            phase_names=np.asarray([self.phase_names[key] for key in sorted(self.phase_names)]),
        )
        return target


    def export(self, directory: str | Path, formats: tuple[str, ...]) -> Mapping[str, Path]:
        """Export labels with an explicit metadata sidecar for lossy container formats."""

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        metadata = {
            "schema": "zynnova.microstructure-volume.v1",
            "shape_zyx": self.shape,
            "voxel_size_m_zyx": self.voxel_size_m,
            "origin_m_zyx": self.origin_m,
            "dtype": str(self.labels.dtype),
            "phase_names": self.phase_names,
            "metadata": self.metadata,
        }
        for requested in formats:
            kind = str(requested).lower().lstrip(".")
            if kind == "npz":
                outputs["npz"] = self.save_npz(root / "microstructure.npz")
            elif kind == "npy":
                path = root / "microstructure.npy"
                np.save(path, self.labels, allow_pickle=False)
                outputs["npy"] = path
            elif kind == "raw":
                path = root / "microstructure.int32.raw"
                np.asarray(self.labels, dtype="<i4").tofile(path)
                outputs["raw"] = path
            elif kind in {"tif", "tiff"}:
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise RuntimeError("TIFF export requires Pillow") from exc
                if int(self.labels.min()) < 0 or int(self.labels.max()) > 65535:
                    raise ValueError("TIFF label export supports phase ids in [0, 65535]")
                pages = [Image.fromarray(layer.astype(np.uint16)) for layer in self.labels]
                path = root / "microstructure.tiff"
                pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_deflate")
                outputs["tiff"] = path
            else:
                raise ValueError(f"unsupported microstructure format: {requested}")
        outputs["metadata"] = dump_json(root / "microstructure.metadata.json", metadata)
        return outputs

    @classmethod
    def load_npz(cls, path: str | Path) -> MicrostructureVolume:
        with np.load(Path(path), allow_pickle=False) as data:
            phase_names = {
                int(key): str(value)
                for key, value in zip(data["phase_ids"], data["phase_names"], strict=True)
            }
            return cls(
                labels=data["labels"],
                voxel_size_m=tuple(float(item) for item in data["voxel_size_m"]),
                origin_m=tuple(float(item) for item in data["origin_m"]),
                phase_names=phase_names,
                metadata={"loaded_from": str(Path(path).resolve())},
            )


__all__ = ["MicrostructureVolume"]
