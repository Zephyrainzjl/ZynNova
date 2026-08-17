from __future__ import annotations

from importlib import import_module
from typing import Literal

import numpy as np

BackendName = Literal["auto", "python", "cpp"]


def _native_module():
    try:
        return import_module("zynnova._native._structure_native")
    except ImportError:
        return None


def native_available() -> bool:
    return _native_module() is not None


def resolve_backend(backend: BackendName) -> Literal["python", "cpp"]:
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("backend must be 'auto', 'python', or 'cpp'")
    if backend == "auto":
        return "cpp" if native_available() else "python"
    if backend == "cpp" and not native_available():
        raise RuntimeError(
            "The C++ backend is unavailable. Install/build ZynNova with its native extension, "
            "or use backend='python'."
        )
    return backend


def _image_bound(cell: np.ndarray, pbc: np.ndarray, cutoff: float) -> int:
    if not np.any(pbc):
        return 0
    if abs(np.linalg.det(cell)) < 1e-14:
        raise ValueError("Periodic cell is singular")
    reciprocal_bound = max(np.linalg.norm(np.linalg.inv(cell)[:, i]) for i in range(3))
    return max(1, int(np.ceil(cutoff * reciprocal_bound)) + 1)


def _canonical_undirected(source: int, target: int, shift: tuple[int, int, int]) -> bool:
    return source < target or (source == target and shift > (0, 0, 0))


def build_neighbor_graph_python(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    radii: np.ndarray,
    *,
    mode: str,
    cutoff: float,
    radius_scale: float,
    max_neighbors: int | None,
    directed: bool,
    self_edges: bool,
    tolerance: float,
) -> dict[str, np.ndarray]:
    n = len(positions)
    if mode not in {"cutoff", "radius", "knn"}:
        raise ValueError("mode must be 'cutoff', 'radius', or 'knn'")
    if mode == "knn" and (max_neighbors is None or max_neighbors <= 0):
        raise ValueError("knn mode requires max_neighbors > 0")
    if mode == "knn" and np.any(pbc) and cutoff <= 0:
        raise ValueError("Periodic knn mode requires a positive candidate cutoff")
    max_cutoff = cutoff
    if mode == "radius":
        max_cutoff = 2.0 * radius_scale * float(np.max(radii, initial=0.0))
    if mode != "knn" and max_cutoff <= 0:
        raise ValueError("cutoff must be positive")

    bound = _image_bound(cell, pbc, max(max_cutoff, 0.0))
    ranges = [range(-bound, bound + 1) if pbc[d] else range(0, 1) for d in range(3)]
    candidates: list[tuple[int, float, int, tuple[int, int, int], np.ndarray]] = []
    for source in range(n):
        for target in range(n):
            for sx in ranges[0]:
                for sy in ranges[1]:
                    for sz in ranges[2]:
                        shift = (sx, sy, sz)
                        zero_self = source == target and shift == (0, 0, 0)
                        if zero_self and not self_edges:
                            continue
                        if not directed and not _canonical_undirected(source, target, shift):
                            continue
                        vector = positions[target] + np.asarray(shift) @ cell - positions[source]
                        distance = float(np.linalg.norm(vector))
                        if distance <= tolerance and not self_edges:
                            continue
                        pair_cutoff = (
                            radius_scale * float(radii[source] + radii[target])
                            if mode == "radius"
                            else cutoff
                        )
                        if mode in {"cutoff", "radius"} and distance > pair_cutoff + tolerance:
                            continue
                        if mode == "knn" and cutoff > 0 and distance > cutoff + tolerance:
                            continue
                        candidates.append((source, distance, target, shift, vector))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    if max_neighbors is not None and max_neighbors > 0:
        limited = []
        counts = np.zeros(n, dtype=np.int64)
        for item in candidates:
            source = item[0]
            if counts[source] < max_neighbors:
                limited.append(item)
                counts[source] += 1
        candidates = limited

    edge_count = len(candidates)
    edge_index = np.empty((2, edge_count), dtype=np.int64)
    edge_shift = np.empty((edge_count, 3), dtype=np.int64)
    edge_vec = np.empty((edge_count, 3), dtype=np.float64)
    edge_dist = np.empty(edge_count, dtype=np.float64)
    for e, (source, distance, target, shift, vector) in enumerate(candidates):
        edge_index[:, e] = (source, target)
        edge_shift[e] = shift
        edge_vec[e] = vector
        edge_dist[e] = distance
    return {
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_vec": edge_vec,
        "edge_dist": edge_dist,
    }


def build_neighbor_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    radii: np.ndarray,
    *,
    backend: BackendName,
    mode: str,
    cutoff: float,
    radius_scale: float,
    max_neighbors: int | None,
    directed: bool,
    self_edges: bool,
    tolerance: float,
) -> tuple[dict[str, np.ndarray], str]:
    selected = resolve_backend(backend)
    if selected == "python":
        result = build_neighbor_graph_python(
            positions, cell, pbc, radii,
            mode=mode,
            cutoff=cutoff,
            radius_scale=radius_scale,
            max_neighbors=max_neighbors,
            directed=directed,
            self_edges=self_edges,
            tolerance=tolerance,
        )
    else:
        native = _native_module()
        assert native is not None
        result = native.build_neighbor_graph(
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(cell, dtype=np.float64),
            np.ascontiguousarray(pbc, dtype=bool),
            np.ascontiguousarray(radii, dtype=np.float64),
            mode,
            float(cutoff),
            float(radius_scale),
            int(max_neighbors or 0),
            bool(directed),
            bool(self_edges),
            float(tolerance),
        )
        result = {key: np.asarray(value) for key, value in result.items()}
    return result, selected


def validate_graph_cpp(num_nodes: int, edge_index: np.ndarray, edge_shift: np.ndarray) -> None:
    native = _native_module()
    if native is None:
        raise RuntimeError("The C++ backend is unavailable")
    native.validate_graph(
        int(num_nodes),
        np.ascontiguousarray(edge_index, dtype=np.int64),
        np.ascontiguousarray(edge_shift, dtype=np.int32),
    )
