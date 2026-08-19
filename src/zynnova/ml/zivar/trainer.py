"""Training engine for the single variational ZIVAR energy functional."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from ._deps import require_torch
from .losses import ZIVARLoss
from .types import Conditions, Targets

torch = require_torch()
TRAINER_STATE_SCHEMA = "zivar-trainer-state-0.2.1"
_EXACT_RESUME_SCHEDULERS = frozenset(
    {
        "torch.optim.lr_scheduler.ConstantLR",
        "torch.optim.lr_scheduler.CosineAnnealingLR",
        "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "torch.optim.lr_scheduler.ExponentialLR",
        "torch.optim.lr_scheduler.LinearLR",
        "torch.optim.lr_scheduler.MultiStepLR",
        "torch.optim.lr_scheduler.PolynomialLR",
        "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "torch.optim.lr_scheduler.StepLR",
    }
)


def _qualified_class_name(value: Any | None) -> str | None:
    if value is None:
        return None
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _loss_manifest(loss: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"class": _qualified_class_name(loss)}
    if isinstance(loss, ZIVARLoss):
        result.update(
            {
                "weights": asdict(loss.weights),
                "stationarity_delta": loss.stationarity_delta,
            }
        )
    return result


def _optimizer_parameter_names(model: Any, optimizer: Any) -> tuple[tuple[str, ...], ...]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[tuple[str, ...]] = []
    used: set[str] = set()
    for group in optimizer.param_groups:
        local: list[str] = []
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer contains a parameter outside the ZIVAR model")
            if name in used:
                raise ValueError(f"optimizer parameter {name!r} appears more than once")
            used.add(name)
            local.append(name)
        groups.append(tuple(local))
    return tuple(groups)


def _validate_scheduler_binding(scheduler: Any | None, optimizer: Any) -> None:
    if (
        scheduler is not None
        and hasattr(scheduler, "optimizer")
        and scheduler.optimizer is not optimizer
    ):
        raise ValueError("scheduler is bound to a different optimizer instance")
    scheduler_class = _qualified_class_name(scheduler)
    if scheduler_class is not None and scheduler_class not in _EXACT_RESUME_SCHEDULERS:
        raise ValueError(
            f"scheduler {scheduler_class!r} is not certified for exact resume; "
            "callable/custom scheduler behaviour is not recoverable from a "
            "PyTorch state_dict"
        )


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    amp: bool = False
    allow_amp_second_order: bool = False
    force_training: bool = True
    spin_field_training: bool = True
    verify_optimizer_state: bool = True
    validation_interval: int = 1
    monitor: str = "total"
    monitor_mode: str = "min"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer scales are invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be 'min' or 'max'")


class ZIVARTrainer:
    def __init__(
        self,
        model: Any,
        *,
        loss: ZIVARLoss | None = None,
        config: TrainerConfig | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.model = model
        seal = getattr(model, "seal_backbone", None)
        if callable(seal):
            seal()
        self.loss = loss or ZIVARLoss()
        self.config = config or TrainerConfig()
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = scheduler
        _validate_scheduler_binding(self.scheduler, self.optimizer)
        self.epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None
        device = next(model.parameters()).device
        enabled = self.config.amp and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler(device.type, enabled=enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    @staticmethod
    def _validate_no_label_leakage(conditions: dict[str, Any]) -> None:
        forbidden = {
            "initial_charge_multipoles",
            "target_charges",
            "target_magnetic_moments",
        }
        leaked = sorted(forbidden.intersection(conditions))
        if leaked:
            raise ValueError(f"electronic supervision leaked into conditions: {leaked}")

    def _forward_loss(
        self,
        batch: dict[str, Any],
        *,
        training: bool,
    ) -> dict[str, Any]:
        data, target_value = batch["data"], batch["targets"]
        targets = target_value.as_dict() if isinstance(target_value, Targets) else target_value
        condition_value = batch.get("conditions", {})
        conditions = (
            condition_value.as_dict()
            if isinstance(condition_value, Conditions)
            else dict(condition_value)
        )
        self._validate_no_label_leakage(conditions)
        need_forces = self.config.force_training and "forces" in targets
        need_stress = "stress" in targets
        need_spin_fields = self.config.spin_field_training and any(
            name in targets for name in ("effective_field_T", "magnetic_torque_eV")
        )
        second_order = training and (need_forces or need_stress or need_spin_fields)
        device_type = next(self.model.parameters()).device.type
        if self.scaler.is_enabled() and second_order and not self.config.allow_amp_second_order:
            raise RuntimeError(
                "AMP second-order force/field training is disabled until the exact "
                "GPU, PyTorch and equivariant-kernel stack passes the release matrix; "
                "use amp=False or explicitly validate and set allow_amp_second_order=True"
            )
        with torch.amp.autocast(
            device_type=device_type,
            enabled=self.scaler.is_enabled(),
        ):
            output = self.model.energy_forces_stress(
                data,
                conditions=conditions,
                create_graph=second_order,
                compute_stress=need_stress,
                compute_spin_fields=need_spin_fields,
            )
            return self.loss(output, targets)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Take one update without ever copying electronic labels into inputs."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        terms = self._forward_loss(batch, training=True)
        if not bool(torch.isfinite(terms["total"]).detach()):
            raise FloatingPointError("non-finite ZIVAR loss before backward")
        self.scaler.scale(terms["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(norm).detach()):
            raise FloatingPointError("non-finite ZIVAR gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.config.verify_optimizer_state:
            assert_model_optimizer_finite(self.model, self.optimizer)
        self.global_step += 1
        return {name: float(value.detach()) for name, value in terms.items()}

    def validation_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Evaluate exactly the same physical core without optimizer mutation."""

        self.model.eval()
        # Forces and stresses still require position derivatives, so inference
        # mode/no_grad would be incorrect here.
        with torch.enable_grad():
            terms = self._forward_loss(batch, training=False)
        return {name: float(value.detach()) for name, value in terms.items()}

    def fit_epoch(self, loader: Iterable[dict[str, Any]]) -> dict[str, float]:
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            values = self.train_step(batch)
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1
        if batches == 0:
            raise ValueError("training loader is empty")
        self.epoch += 1
        return {name: value / batches for name, value in totals.items()}

    def evaluate_epoch(self, loader: Iterable[dict[str, Any]]) -> dict[str, float]:
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            values = self.validation_step(batch)
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1
        if batches == 0:
            raise ValueError("validation loader is empty")
        return {name: value / batches for name, value in totals.items()}

    def fit(
        self,
        train_loader: Iterable[dict[str, Any]],
        *,
        epochs: int,
        validation_loader: Iterable[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run train/validation epochs and retain deterministic engine state."""

        if epochs < 1:
            raise ValueError("epochs must be positive")
        history: list[dict[str, Any]] = []
        for _ in range(epochs):
            train_metrics = self.fit_epoch(train_loader)
            validation_metrics = None
            if (
                validation_loader is not None
                and self.epoch % self.config.validation_interval == 0
            ):
                validation_metrics = self.evaluate_epoch(validation_loader)
                metric = validation_metrics.get(self.config.monitor)
                if metric is None:
                    raise KeyError(
                        f"validation metric {self.config.monitor!r} is unavailable"
                    )
                improved = self.best_metric is None or (
                    metric < self.best_metric
                    if self.config.monitor_mode == "min"
                    else metric > self.best_metric
                )
                if improved:
                    self.best_metric = float(metric)
            if self.scheduler is not None:
                if validation_metrics is not None and self.config.monitor in validation_metrics:
                    try:
                        self.scheduler.step(validation_metrics[self.config.monitor])
                    except TypeError:
                        self.scheduler.step()
                else:
                    self.scheduler.step()
            history.append(
                {
                    "epoch": self.epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
        return history

    def state_dict(self) -> dict[str, Any]:
        _validate_scheduler_binding(self.scheduler, self.optimizer)
        return {
            "schema": TRAINER_STATE_SCHEMA,
            "trainer_config": asdict(self.config),
            "loss": _loss_manifest(self.loss),
            "optimizer_class": _qualified_class_name(self.optimizer),
            "optimizer_parameter_names": _optimizer_parameter_names(
                self.model, self.optimizer
            ),
            "scheduler_class": _qualified_class_name(self.scheduler),
            "model_config": self.model.config.to_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        _validate_scheduler_binding(self.scheduler, self.optimizer)
        if state.get("schema") != TRAINER_STATE_SCHEMA:
            raise ValueError("trainer state schema is incompatible")
        if strict and state.get("trainer_config") != asdict(self.config):
            raise ValueError("trainer configuration differs from the checkpoint")
        if strict and state.get("loss") != _loss_manifest(self.loss):
            raise ValueError("trainer loss configuration differs from the checkpoint")
        if strict and state.get("optimizer_class") != _qualified_class_name(
            self.optimizer
        ):
            raise ValueError("trainer optimizer class differs from the checkpoint")
        if strict and state.get("optimizer_parameter_names") != (
            _optimizer_parameter_names(self.model, self.optimizer)
        ):
            raise ValueError(
                "trainer optimizer parameter groups/order differ from the checkpoint"
            )
        if strict and state.get("scheduler_class") != _qualified_class_name(
            self.scheduler
        ):
            raise ValueError("trainer scheduler class differs from the checkpoint")
        if strict and state.get("model_config") != self.model.config.to_dict():
            raise ValueError("trainer model configuration differs from the checkpoint")
        self.optimizer.load_state_dict(state["optimizer"])
        scheduler_state = state.get("scheduler")
        if scheduler_state is not None:
            if self.scheduler is None:
                if strict:
                    raise ValueError("checkpoint contains scheduler state but trainer has none")
            else:
                self.scheduler.load_state_dict(scheduler_state)
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        value = state.get("best_metric")
        self.best_metric = None if value is None else float(value)


def assert_model_optimizer_finite(model: Any, optimizer: Any) -> None:
    """Check parameters and optimizer tensors without mixing CPU/CUDA devices."""

    grouped: dict[Any, list[Any]] = {}
    for parameter in model.parameters():
        grouped.setdefault(parameter.device, []).append(torch.isfinite(parameter).all())
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                grouped.setdefault(value.device, []).append(torch.isfinite(value).all())
    failures = []
    for device, checks in grouped.items():
        if checks and not bool(torch.stack(checks).all().detach().cpu()):
            failures.append(str(device))
    if failures:
        raise FloatingPointError(
            "optimizer update produced non-finite parameters/state on "
            + ", ".join(sorted(failures))
        )


__all__ = [
    "TRAINER_STATE_SCHEMA",
    "TrainerConfig",
    "ZIVARTrainer",
    "assert_model_optimizer_finite",
]
