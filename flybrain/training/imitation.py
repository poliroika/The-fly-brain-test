"""Phase-7 imitation learning (PLAN.md §592-595).

Supervised cloning of an expert policy from collected MAS traces.
Reuses the synthetic-pretrain plumbing (CE on kind + agent +
auxiliary verifier prediction) but adds:

* edge-action heads (``add_edge`` / ``remove_edge`` / ``scale_edge``)
  — the expert may perform graph mutations the synthetic pretrain
  doesn't cover.
* a held-out evaluation that scores the trained controller against
  the expert on a *fresh* split of traces (not just per-step
  accuracy); we report both per-step argmax accuracy and a sim-MAS
  rollout success rate.

The exit criterion (PLAN.md §595) is "на held-out subset
IL-controller бьёт sim-only по success rate и/или по cost". The
default config gives a reasonable budget for that on CPU; the live
metric is computed by ``scripts/run_imitation.py`` after the loop
completes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from flybrain.controller.action_space import (
    KIND_ACTIVATE_AGENT,
    KIND_ADD_EDGE,
    KIND_REMOVE_EDGE,
    KIND_SCALE_EDGE,
)
from flybrain.training.expert_dataset import (
    ImitationExample,
    collect_examples,
    iter_traces,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from flybrain.controller._torch_base import TorchControllerBase


_NEEDS_EDGE = {KIND_ADD_EDGE, KIND_REMOVE_EDGE, KIND_SCALE_EDGE}


@dataclass(slots=True)
class ImitationConfig:
    epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    only_passed: bool = True
    """Drop traces whose verification didn't pass."""

    aux_loss_weight: float = 0.1
    edge_loss_weight: float = 0.5

    train_frac: float = 0.8


@dataclass(slots=True)
class ImitationResult:
    losses: list[float] = field(default_factory=list)
    epoch_accuracy: list[float] = field(default_factory=list)
    final_accuracy: float = 0.0
    num_examples: int = 0
    num_train: int = 0
    num_eval: int = 0


def imitation_train(
    controller: TorchControllerBase,
    traces_dir: str | Path,
    *,
    agent_names: list[str],
    config: ImitationConfig | None = None,
    examples: list[ImitationExample] | None = None,
) -> ImitationResult:
    """Run supervised cloning over expert traces.

    Pass ``examples`` directly to skip filesystem loading (used by
    tests + when chaining with synthetic pretrain). Otherwise the
    function walks ``traces_dir`` for ``*.trace.json`` files."""
    import torch

    cfg = config or ImitationConfig()

    if examples is None:
        examples = collect_examples(
            iter_traces(traces_dir),
            agent_names=agent_names,
            only_passed=cfg.only_passed,
        )
    if not examples:
        return ImitationResult(num_examples=0)

    shuffled = list(examples)
    random.Random(cfg.seed).shuffle(shuffled)
    split = int(cfg.train_frac * len(shuffled))
    train_set = shuffled[:split] or shuffled
    eval_set = shuffled[split:] or shuffled[-max(1, len(shuffled) // 5) :]

    optim = torch.optim.AdamW(
        controller.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)

    result = ImitationResult(
        num_examples=len(examples),
        num_train=len(train_set),
        num_eval=len(eval_set),
    )

    for _epoch in range(cfg.epochs):
        order = torch.randperm(len(train_set), generator=rng).tolist()
        for batch_start in range(0, len(order), cfg.batch_size):
            batch_idx = order[batch_start : batch_start + cfg.batch_size]
            optim.zero_grad()
            loss = _batch_loss(controller, train_set, batch_idx, cfg)
            loss.backward()
            optim.step()
            result.losses.append(float(loss.item()))
        result.epoch_accuracy.append(_eval_accuracy(controller, eval_set))

    result.final_accuracy = result.epoch_accuracy[-1] if result.epoch_accuracy else 0.0
    return result


def _batch_loss(
    controller: TorchControllerBase,
    examples: list[ImitationExample],
    batch_idx: list[int],
    cfg: ImitationConfig,
) -> torch.Tensor:
    import torch
    from torch.nn import functional as torch_func

    losses: list[torch.Tensor] = []
    for i in batch_idx:
        ex = examples[i]
        cs = controller.builder.from_runtime_sync(ex.runtime_state)
        out = controller(cs)

        kind_loss = torch_func.cross_entropy(
            out.kind_logits.unsqueeze(0),
            torch.tensor([ex.label_kind], dtype=torch.long),
        )

        head_loss = torch.zeros(())
        if ex.label_kind == KIND_ACTIVATE_AGENT and out.agent_logits.numel() > 0:
            head_loss = torch_func.cross_entropy(
                out.agent_logits.unsqueeze(0),
                torch.tensor([ex.label_agent], dtype=torch.long),
            )
        elif ex.label_kind in _NEEDS_EDGE and out.edge_from_logits.numel() > 0:
            edge_from_loss = torch_func.cross_entropy(
                out.edge_from_logits.unsqueeze(0),
                torch.tensor([ex.label_edge_from], dtype=torch.long),
            )
            edge_to_loss = torch_func.cross_entropy(
                out.edge_to_logits.unsqueeze(0),
                torch.tensor([ex.label_edge_to], dtype=torch.long),
            )
            head_loss = cfg.edge_loss_weight * (edge_from_loss + edge_to_loss)

        aux_loss = torch_func.binary_cross_entropy(
            out.aux_verifier.unsqueeze(0),
            torch.tensor([ex.aux_target], dtype=out.aux_verifier.dtype).clamp(0.0, 1.0),
        )

        losses.append(kind_loss + head_loss + cfg.aux_loss_weight * aux_loss)
    return torch.stack(losses).mean()


def _eval_accuracy(
    controller: TorchControllerBase,
    examples: list[ImitationExample],
) -> float:
    import torch

    if not examples:
        return 0.0
    hits = 0
    with torch.no_grad():
        for ex in examples:
            cs = controller.builder.from_runtime_sync(ex.runtime_state)
            out = controller(cs)
            kind_pred = int(torch.argmax(out.kind_logits).item())
            if kind_pred != ex.label_kind:
                continue
            if ex.label_kind == KIND_ACTIVATE_AGENT:
                if out.agent_logits.numel() == 0:
                    continue
                if int(torch.argmax(out.agent_logits).item()) != ex.label_agent:
                    continue
            elif ex.label_kind in _NEEDS_EDGE:
                if out.edge_from_logits.numel() == 0:
                    continue
                if int(torch.argmax(out.edge_from_logits).item()) != ex.label_edge_from:
                    continue
                if int(torch.argmax(out.edge_to_logits).item()) != ex.label_edge_to:
                    continue
            hits += 1
    return hits / len(examples)


__all__ = ["ImitationConfig", "ImitationResult", "imitation_train"]
