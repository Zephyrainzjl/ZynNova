from __future__ import annotations

import math

from ...common import require_torch

torch = require_torch()


def cosine_corrupt(token_ids, attention_mask, time, *, tokenizer):
    eligible = (
        attention_mask.bool()
        & token_ids.ne(tokenizer.pad_id)
        & token_ids.ne(tokenizer.bos_id)
        & token_ids.ne(tokenizer.eos_id)
    )
    probability = torch.sin(0.5 * math.pi * time).square()
    selected = torch.rand_like(token_ids, dtype=torch.float32) < probability[:, None]
    selected &= eligible
    for row in range(token_ids.shape[0]):
        if not selected[row].any() and eligible[row].any():
            choices = torch.nonzero(eligible[row], as_tuple=False).flatten()
            random_index = torch.randint(choices.numel(), (1,), device=choices.device)
            selected[row, choices[random_index]] = True
    noisy = token_ids.clone()
    noisy[selected] = tokenizer.mask_id
    return noisy, selected


def polyloom_losses(model, batch, *, tokenizer, config, training: bool):
    batch_size = batch["token_ids"].shape[0]
    time = torch.rand(
        batch_size, device=batch["token_ids"].device, dtype=batch["properties"].dtype
    ).clamp_(1.0e-3, 1.0 - 1.0e-3)
    noisy, flow_mask = cosine_corrupt(
        batch["token_ids"], batch["attention_mask"], time, tokenizer=tokenizer
    )
    property_mask = batch["property_mask"].clone()
    process_mask = batch["process_condition_mask"].clone()
    if training and config.train.condition_dropout:
        dropped = torch.rand(batch_size, device=time.device) < config.train.condition_dropout
        property_mask[dropped] = False
        process_mask[dropped] = False
    self_condition = None
    if training and torch.rand((), device=time.device) < config.train.self_condition_probability:
        with torch.no_grad():
            first = model(
                noisy, batch["attention_mask"], time, batch["properties"], property_mask,
                batch["process_conditions"], process_mask,
            )
            self_condition = first["logits"].softmax(dim=-1)
    output = model(
        noisy, batch["attention_mask"], time, batch["properties"], property_mask,
        batch["process_conditions"], process_mask, self_condition=self_condition,
    )
    token_loss = torch.nn.functional.cross_entropy(
        output["logits"][flow_mask], batch["token_ids"][flow_mask]
    )
    property_prediction = model.predict_properties(
        batch["token_ids"], batch["attention_mask"]
    )
    observed = batch["property_mask"]
    property_loss = (
        torch.nn.functional.smooth_l1_loss(
            property_prediction[observed], batch["properties"][observed]
        )
        if observed.any() else token_loss * 0.0
    )
    length_loss = torch.nn.functional.cross_entropy(
        model.predict_length(
            batch["properties"], property_mask,
            batch["process_conditions"], process_mask,
        ),
        batch["length"],
    )
    endpoint_target = (
        batch["token_ids"].eq(tokenizer.eos_id)
        | batch["token_ids"].eq(tokenizer.bos_id)
    ).to(output["endpoint_logits"].dtype)
    endpoint_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["endpoint_logits"][batch["attention_mask"]],
        endpoint_target[batch["attention_mask"]],
    )
    total = (
        token_loss
        + config.train.property_loss_weight * property_loss
        + config.train.length_loss_weight * length_loss
        + config.train.endpoint_loss_weight * endpoint_loss
        + config.train.expert_balance_weight * output["expert_balance_loss"]
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "flow_token_nll": float(token_loss.detach().cpu()),
        "property_loss": float(property_loss.detach().cpu()),
        "length_loss": float(length_loss.detach().cpu()),
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
    }


__all__ = ["cosine_corrupt", "polyloom_losses"]
