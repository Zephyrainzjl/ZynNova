from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ....data import MaterialSample


@dataclass(slots=True)
class QM9PropertyNormalizer:
    """Per-property standardization fitted only on the training split."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    def __post_init__(self) -> None:
        self.names = tuple(str(name) for name in self.names)
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        self.count = np.asarray(self.count, dtype=np.int64).reshape(-1)
        size = len(self.names)
        if self.mean.shape != (size,) or self.std.shape != (size,):
            raise ValueError("mean/std must match property names")
        if self.count.shape != (size,):
            raise ValueError("count must match property names")
        if np.any(~np.isfinite(self.mean)) or np.any(~np.isfinite(self.std)):
            raise ValueError("normalizer statistics must be finite")
        if np.any(self.std <= 0):
            raise ValueError("normalizer std must be positive")

    @classmethod
    def fit(
        cls,
        samples: Sequence[MaterialSample] | Iterable[MaterialSample],
        names: Sequence[str],
        *,
        minimum_std: float = 1.0e-8,
    ) -> "QM9PropertyNormalizer":
        names = tuple(str(name) for name in names)
        values: list[list[float]] = [[] for _ in names]
        for sample in samples:
            for index, name in enumerate(names):
                value = sample.labels.get(name)
                if value is None:
                    continue
                scalar = float(np.asarray(value).reshape(-1)[0])
                if np.isfinite(scalar):
                    values[index].append(scalar)
        missing = [name for name, item in zip(names, values, strict=True) if not item]
        if missing:
            raise ValueError(f"cannot fit normalizer; no finite values for {missing}")
        mean = np.asarray([np.mean(item) for item in values], dtype=np.float64)
        std = np.asarray([np.std(item) for item in values], dtype=np.float64)
        std = np.maximum(std, minimum_std)
        count = np.asarray([len(item) for item in values], dtype=np.int64)
        return cls(names, mean, std, count)

    def encode_mapping(
        self,
        values: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        encoded = np.zeros(len(self.names), dtype=np.float64)
        mask = np.zeros(len(self.names), dtype=bool)
        unknown = sorted(set(values) - set(self.names))
        if unknown:
            raise ValueError(
                f"properties {unknown} were not used to train this model; "
                f"available={self.names}"
            )
        for index, name in enumerate(self.names):
            if name not in values:
                continue
            value = float(values[name])
            if not np.isfinite(value):
                raise ValueError(f"property {name!r} must be finite")
            encoded[index] = (value - self.mean[index]) / self.std[index]
            mask[index] = True
        return encoded, mask

    def decode_array(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array * self.std + self.mean

    def decode_mapping(self, values: Any) -> dict[str, float]:
        decoded = self.decode_array(values).reshape(-1)
        if len(decoded) != len(self.names):
            raise ValueError("property vector has the wrong size")
        return {
            name: float(decoded[index])
            for index, name in enumerate(self.names)
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "QM9PropertyNormalizer":
        return cls(
            names=tuple(state["names"]),
            mean=np.asarray(state["mean"], dtype=np.float64),
            std=np.asarray(state["std"], dtype=np.float64),
            count=np.asarray(state["count"], dtype=np.int64),
        )


__all__ = ["QM9PropertyNormalizer"]
