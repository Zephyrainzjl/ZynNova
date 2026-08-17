from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class KANOracleDiagnostics:
    train_r2: float
    validation_r2: float
    best_epoch: int
    epochs_completed: int
    device: str
    dtype: str
    parameter_count: int
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    target_center: float
    target_scale: float


class PhysicsKANOracle:
    """Sparse RBF-KAN oracle for nonlinear derivatives and distillation.

    The public wrapper imports PyTorch only when ``fit`` is called. Each edge
    carries a learnable univariate radial-basis expansion plus a smooth base
    function, preserving the key KAN idea that functions live on edges.
    """

    def __init__(
        self,
        *,
        hidden_width: int = 16,
        grid_size: int = 12,
        layers: int = 2,
        epochs: int = 500,
        learning_rate: float = 2.0e-3,
        weight_decay: float = 1.0e-6,
        sparsity_weight: float = 1.0e-5,
        patience: int = 60,
        device: str = "auto",
        dtype: str = "float64",
        random_seed: int = 42,
        monotonic_constraints: Mapping[int, int] | None = None,
        monotonicity_weight: float = 0.05,
    ) -> None:
        if hidden_width < 2 or grid_size < 4 or layers < 1:
            raise ValueError("invalid KAN architecture")
        if epochs < 1 or learning_rate <= 0 or patience < 1:
            raise ValueError("invalid KAN optimizer configuration")
        if dtype not in {"float32", "float64"}:
            raise ValueError("KAN dtype must be float32 or float64")
        self.hidden_width = int(hidden_width)
        self.grid_size = int(grid_size)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.sparsity_weight = float(sparsity_weight)
        self.patience = int(patience)
        self.device = str(device)
        self.dtype = dtype
        self.random_seed = int(random_seed)
        self.monotonic_constraints = {
            int(index): int(direction)
            for index, direction in (monotonic_constraints or {}).items()
        }
        if any(direction not in {-1, 1} for direction in self.monotonic_constraints.values()):
            raise ValueError("monotonic directions must be -1 or +1")
        self.monotonicity_weight = float(monotonicity_weight)
        self._model = None
        self._torch = None
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._target_center = 0.0
        self._target_scale = 1.0
        self.diagnostics_: KANOracleDiagnostics | None = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("torch") is not None

    def fit(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        *,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
    ) -> PhysicsKANOracle:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PhysicsKANOracle requires PyTorch; install zynnova[physics-discovery]"
            ) from exc

        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        if matrix.ndim != 2 or len(matrix) != len(target):
            raise ValueError("matrix and target shapes do not match")
        if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(target)):
            raise ValueError("KAN training data must be finite")
        if not len(train_indices) or not len(validation_indices):
            raise ValueError("KAN requires non-empty train and validation indices")

        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        resolved_device = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else ("cpu" if self.device == "auto" else self.device)
        )
        torch_dtype = torch.float64 if self.dtype == "float64" else torch.float32

        center = np.mean(matrix[train_indices], axis=0)
        scale = np.std(matrix[train_indices], axis=0)
        scale[scale < 1.0e-12] = 1.0
        target_center = float(np.mean(target[train_indices]))
        target_scale = max(float(np.std(target[train_indices])), 1.0e-12)
        standardized = (matrix - center) / scale
        target_z = (target - target_center) / target_scale

        train_x = torch.as_tensor(
            standardized[train_indices],
            dtype=torch_dtype,
            device=resolved_device,
        )
        train_y = torch.as_tensor(
            target_z[train_indices, None],
            dtype=torch_dtype,
            device=resolved_device,
        )
        valid_x = torch.as_tensor(
            standardized[validation_indices],
            dtype=torch_dtype,
            device=resolved_device,
        )
        valid_y = torch.as_tensor(
            target_z[validation_indices, None],
            dtype=torch_dtype,
            device=resolved_device,
        )

        model = _build_rbf_kan(
            torch,
            input_width=matrix.shape[1],
            hidden_width=self.hidden_width,
            grid_size=self.grid_size,
            layers=self.layers,
        ).to(device=resolved_device, dtype=torch_dtype)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        epochs_completed = 0

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            needs_gradient = bool(self.monotonic_constraints)
            batch_x = train_x.detach().clone().requires_grad_(needs_gradient)
            prediction = model(batch_x)
            loss = torch.mean((prediction - train_y) ** 2)
            if self.sparsity_weight > 0:
                loss = loss + self.sparsity_weight * model.edge_sparsity()
            if needs_gradient and self.monotonicity_weight > 0:
                gradient = torch.autograd.grad(
                    prediction.sum(),
                    batch_x,
                    create_graph=True,
                )[0]
                violations = []
                for index, direction in self.monotonic_constraints.items():
                    if index < 0 or index >= matrix.shape[1]:
                        raise ValueError(
                            f"monotonic feature index out of range: {index}"
                        )
                    violations.append(
                        torch.relu(-float(direction) * gradient[:, index]).mean()
                    )
                if violations:
                    loss = loss + self.monotonicity_weight * torch.stack(
                        violations
                    ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    torch.mean((model(valid_x) - valid_y) ** 2).detach().cpu()
                )
            epochs_completed = epoch + 1
            if validation_loss + 1.0e-10 < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("KAN training did not produce a finite validation state")
        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._torch = torch
        self._center = center
        self._scale = scale
        self._target_center = target_center
        self._target_scale = target_scale

        train_prediction = self.predict(matrix[train_indices])
        valid_prediction = self.predict(matrix[validation_indices])
        self.diagnostics_ = KANOracleDiagnostics(
            train_r2=_r2(target[train_indices], train_prediction),
            validation_r2=_r2(target[validation_indices], valid_prediction),
            best_epoch=best_epoch,
            epochs_completed=epochs_completed,
            device=resolved_device,
            dtype=self.dtype,
            parameter_count=sum(
                int(parameter.numel()) for parameter in model.parameters()
            ),
            feature_center=tuple(float(value) for value in center),
            feature_scale=tuple(float(value) for value in scale),
            target_center=target_center,
            target_scale=target_scale,
        )
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        standardized = self.standardize(matrix)
        return self.predict_standardized(standardized)

    def standardize(self, matrix: np.ndarray) -> np.ndarray:
        if self._center is None or self._scale is None:
            raise RuntimeError("fit the KAN oracle before prediction")
        matrix = np.asarray(matrix, dtype=float)
        return (matrix - self._center) / self._scale

    def predict_standardized(self, matrix: np.ndarray) -> np.ndarray:
        if self._model is None or self._torch is None:
            raise RuntimeError("fit the KAN oracle before prediction")
        torch = self._torch
        parameter = next(self._model.parameters())
        tensor = torch.as_tensor(
            np.asarray(matrix, dtype=float),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.no_grad():
            values = self._model(tensor).reshape(-1).detach().cpu().numpy()
        return self._target_center + self._target_scale * values

    def mixed_hessian_standardized(self, matrix: np.ndarray) -> np.ndarray:
        """Return exact autograd Hessians in standardized feature coordinates."""

        if self._model is None or self._torch is None:
            raise RuntimeError("fit the KAN oracle before requesting derivatives")
        torch = self._torch
        parameter = next(self._model.parameters())
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2:
            raise ValueError("matrix must have shape (samples, features)")
        if np.any(~np.isfinite(values)):
            raise ValueError("matrix contains non-finite values")
        hessians = []
        for row in values:
            tensor = torch.as_tensor(
                row,
                dtype=parameter.dtype,
                device=parameter.device,
            ).detach().clone().requires_grad_(True)

            def scalar_function(sample):
                standardized_target = self._model(sample[None, :]).reshape(())
                return float(self._target_scale) * standardized_target

            hessian = torch.autograd.functional.hessian(
                scalar_function,
                tensor,
                create_graph=False,
                strict=False,
            )
            hessians.append(hessian.detach().cpu().numpy())
        return np.asarray(hessians, dtype=float)


def _build_rbf_kan(
    torch,
    *,
    input_width: int,
    hidden_width: int,
    grid_size: int,
    layers: int,
):
    nn = torch.nn

    class RBFKANLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            centers = torch.linspace(-3.0, 3.0, grid_size)
            self.register_buffer("centers", centers)
            self.log_inverse_width = nn.Parameter(
                torch.tensor(math.log(max(grid_size - 1, 1) / 6.0))
            )
            self.rbf_weight = nn.Parameter(
                torch.empty(out_features, in_features, grid_size)
            )
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features)
            )
            self.bias = nn.Parameter(torch.zeros(out_features))
            nn.init.xavier_uniform_(self.rbf_weight)
            nn.init.xavier_uniform_(self.base_weight)

        def forward(self, values):
            inverse_width = torch.nn.functional.softplus(
                self.log_inverse_width
            ) + 1.0e-4
            basis = torch.exp(
                -((values[..., None] - self.centers) * inverse_width) ** 2
            )
            rbf = torch.einsum("big,oig->bo", basis, self.rbf_weight)
            base = torch.einsum(
                "bi,oi->bo",
                torch.nn.functional.silu(values),
                self.base_weight,
            )
            return rbf + base + self.bias

        def sparsity(self):
            edge_norm = torch.sqrt(
                torch.sum(self.rbf_weight**2, dim=-1)
                + self.base_weight**2
                + 1.0e-12
            )
            return edge_norm.mean()

    class RBFKAN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = [
                input_width,
                *([hidden_width] * layers),
                1,
            ]
            self.layers = nn.ModuleList(
                RBFKANLayer(left, right)
                for left, right in zip(widths[:-1], widths[1:], strict=True)
            )

        def forward(self, values):
            for layer in self.layers:
                values = layer(values)
            return values

        def edge_sparsity(self):
            return torch.stack([layer.sparsity() for layer in self.layers]).mean()

    return RBFKAN()


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    return float("nan") if total < 1.0e-15 else 1.0 - residual / total


__all__ = [
    "KANOracleDiagnostics",
    "PhysicsKANOracle",
]
