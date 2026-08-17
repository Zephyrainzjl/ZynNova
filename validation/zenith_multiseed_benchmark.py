from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import torch


def _graph_energy(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> torch.Tensor:
    energies: list[torch.Tensor] = []
    for start, stop in offsets:
        z = atomic_numbers[start:stop].to(positions.dtype)
        pos = positions[start:stop]
        displacement = pos[:, None, :] - pos[None, :, :]
        distance = torch.sqrt(displacement.square().sum(dim=-1) + 1.0e-12)
        pair_mask = torch.triu(
            torch.ones((stop - start, stop - start), dtype=torch.bool), diagonal=1
        )
        zi = z[:, None].expand_as(distance)[pair_mask]
        zj = z[None, :].expand_as(distance)[pair_mask]
        rij = distance[pair_mask]
        chemical = 0.55 + 0.015 * torch.sqrt(zi * zj) + 0.003 * torch.abs(zi - zj)
        equilibrium = 0.78 + 0.012 * (torch.sqrt(zi) + torch.sqrt(zj))
        pair = 0.34 * chemical * (rij - equilibrium).square()
        pair = pair + 0.025 * chemical * torch.exp(-1.7 * rij)

        center_vector = pos[1:] - pos[0]
        center_distance = torch.sqrt(center_vector.square().sum(dim=-1) + 1.0e-12)
        unit = center_vector / center_distance[:, None]
        angle_energy = positions.new_zeros(())
        for left in range(unit.shape[0]):
            for right in range(left + 1, unit.shape[0]):
                cosine = torch.sum(unit[left] * unit[right])
                neighbor_factor = 0.04 + 0.001 * (
                    z[left + 1] + z[right + 1]
                )
                target = -0.25 + 0.02 * torch.tanh(
                    (z[left + 1] - z[right + 1]) / 10.0
                )
                radial_gate = torch.exp(
                    -0.35
                    * (
                        (center_distance[left] - 1.25).square()
                        + (center_distance[right] - 1.25).square()
                    )
                )
                angle_energy = angle_energy + (
                    neighbor_factor * radial_gate * (cosine - target).square()
                )

        density = torch.sum(
            (0.65 + 0.01 * z[1:])
            * torch.exp(-0.75 * center_distance.square())
        )
        coordination = 0.035 * density.square() + 0.015 * density**3
        energies.append(pair.sum() + angle_energy + coordination)
    return torch.stack(energies)


def _build_dataset(
    seed: int,
    count: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    atomic_numbers: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    batches: list[torch.Tensor] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    element_pool = torch.tensor([3, 6, 8, 14, 28], dtype=torch.long)
    for graph in range(count):
        atom_count = 3 + graph % 3
        indices = torch.randint(
            0, element_pool.numel(), (atom_count,), generator=generator
        )
        z = element_pool[indices]
        z[0] = element_pool[graph % element_pool.numel()]
        direction = torch.randn((atom_count - 1, 3), generator=generator)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1)[:, None]
        radius = 0.92 + 0.78 * torch.rand(atom_count - 1, generator=generator)
        local = torch.zeros((atom_count, 3))
        local[1:] = direction * radius[:, None]
        local[1:] += 0.04 * torch.randn(
            (atom_count - 1, 3), generator=generator
        )
        atomic_numbers.append(z)
        positions.append(local)
        batches.append(torch.full((atom_count,), graph, dtype=torch.long))
        offsets.append((cursor, cursor + atom_count))
        cursor += atom_count

    z_all = torch.cat(atomic_numbers)
    pos_all = torch.cat(positions).requires_grad_(True)
    target_energy = _graph_energy(z_all, pos_all, offsets)
    target_force = -torch.autograd.grad(target_energy.sum(), pos_all)[0]
    graph_count = len(offsets)
    inputs = {
        "z": z_all,
        "pos": pos_all.detach(),
        "batch": torch.cat(batches),
        "cell": torch.zeros((graph_count, 3, 3)),
        "pbc": torch.zeros((graph_count, 3), dtype=torch.bool),
    }
    return inputs, target_energy.detach(), target_force.detach()


def _model_config(overrides: dict[str, object]):
    from zynnova.ml.zynforge.field import JouleWeaveModelConfig

    known = {item.name for item in fields(JouleWeaveModelConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown config overrides: {unknown}")
    values: dict[str, object] = {
        "hidden_dim": 8,
        "num_layers": 1,
        "num_radial": 4,
        "max_ell": 2,
        "correlation_order": 2,
        "tensor_product_rank": 4,
        "directional_edge_rank": 4,
        "num_attention_heads": 1,
        "num_experts": 1,
        "expert_top_k": 1,
        "interaction_cutoff_A": 3.5,
        "max_neighbors": None,
        "use_pair_chemical_bias": True,
        "pair_chemical_rank": 8,
        "use_hybrid_irrep_norm": True,
        "use_learned_residual_scales": True,
        "use_electronic_depth_context": False,
        "use_layer_energy_mixing": False,
        "use_magmoms": False,
        "use_charge_head": False,
        "use_oxidation_states": False,
        "use_zbl": False,
        "use_dispersion": False,
        "use_qeq": False,
        "use_latent_ewald": False,
    }
    values.update(overrides)
    return JouleWeaveModelConfig.specialist(**values)


def _loss_and_output(
    model,
    inputs: dict[str, torch.Tensor],
    target_energy: torch.Tensor,
    target_force: torch.Tensor,
    *,
    create_graph: bool,
):
    output = model.energy_and_forces(inputs, create_graph=create_graph)
    atom_count = torch.bincount(inputs["batch"]).to(target_energy)
    energy_residual = (output["energy"] - target_energy) / atom_count
    force_residual = output["forces"] - target_force
    loss = energy_residual.square().mean() + force_residual.square().mean()
    return loss, output


def run_one(seed: int, steps: int, overrides: dict[str, object]) -> dict[str, float]:
    from zynnova.ml.zynforge.field import ZynForgeSymmetryPotential

    torch.manual_seed(seed)
    train_inputs, train_energy, train_force = _build_dataset(1000, 12)
    test_inputs, test_energy, test_force = _build_dataset(2000, 12)
    model = ZynForgeSymmetryPotential(_model_config(overrides)).float().train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4.0e-3, weight_decay=0.0, amsgrad=True
    )
    initial = float(
        _loss_and_output(
            model,
            train_inputs,
            train_energy,
            train_force,
            create_graph=False,
        )[0]
    )
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss_and_output(
            model,
            train_inputs,
            train_energy,
            train_force,
            create_graph=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval()
    train_loss, train_output = _loss_and_output(
        model,
        train_inputs,
        train_energy,
        train_force,
        create_graph=False,
    )
    test_loss, test_output = _loss_and_output(
        model,
        test_inputs,
        test_energy,
        test_force,
        create_graph=False,
    )
    train_count = torch.bincount(train_inputs["batch"]).to(train_energy)
    test_count = torch.bincount(test_inputs["batch"]).to(test_energy)
    return {
        "seed": float(seed),
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "seconds": elapsed,
        "initial_loss": initial,
        "train_loss_ratio": float(train_loss) / max(initial, 1.0e-12),
        "train_energy_mae_per_atom": float(
            torch.mean(torch.abs(train_output["energy"] - train_energy) / train_count)
        ),
        "train_force_mae": float(
            torch.mean(torch.abs(train_output["forces"] - train_force))
        ),
        "test_loss": float(test_loss),
        "test_energy_mae_per_atom": float(
            torch.mean(torch.abs(test_output["energy"] - test_energy) / test_count)
        ),
        "test_force_mae": float(
            torch.mean(torch.abs(test_output["forces"] - test_force))
        ),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", default="7,123,991,2027")
    parser.add_argument("--overrides", default="{}")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    overrides = json.loads(args.overrides)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows = [run_one(seed, args.steps, overrides) for seed in seeds]
    finally:
        torch.set_num_threads(previous_threads)
    summary = {
        key: _mean(rows, key)
        for key in (
            "parameters",
            "seconds",
            "train_loss_ratio",
            "train_energy_mae_per_atom",
            "train_force_mae",
            "test_loss",
            "test_energy_mae_per_atom",
            "test_force_mae",
        )
    }
    payload = {"rows": rows, "mean": summary, "overrides": overrides}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
