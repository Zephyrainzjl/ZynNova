from __future__ import annotations

import math

from ...common import require_torch

torch = require_torch()


def balanced_student_t_nll(output, target, mask):
    """Balanced negative log likelihood under a Normal-Inverse-Gamma posterior."""

    mean, nu, alpha, beta = (
        output["mean"],
        output["nu"],
        output["alpha"],
        output["beta"],
    )
    error = target - mean
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(2.0 * beta * (1.0 + nu))
        + (alpha + 0.5) * torch.log(nu * error.square() + 2.0 * beta * (1.0 + nu))
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    per_property = [
        nll[mask[:, column], column].mean()
        for column in range(target.shape[1])
        if mask[:, column].any()
    ]
    if not per_property:
        raise ValueError("batch contains no observed targets")
    return torch.stack(per_property).mean()


def evidential_regularizer(output, target, mask):
    evidence = 2.0 * output["nu"] + output["alpha"]
    penalty = (target - output["mean"]).abs() * evidence
    return penalty[mask].mean() if mask.any() else penalty.sum() * 0.0


__all__ = [
    "balanced_student_t_nll",
    "evidential_regularizer",
]
