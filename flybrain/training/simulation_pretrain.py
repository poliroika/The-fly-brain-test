"""Phase-6 supervised pretraining (PLAN.md §585-590).

For each synthetic task we walk the optimal route, snapshot the
controller's view of the world at every step, and label that snapshot
with the next agent the expert would activate. The controller is
then trained with cross-entropy on the kind + agent heads.

The exit criterion (PLAN.md §590) is "controller сходится за <10
минут на CPU и решает sim-задачи на ≥0.85 success". The smoke test
in `tests/python/unit/test_simulation_pretrain.py` exercises a tiny
budget: 5 epochs × 8 tasks; the script itself is the larger run that
delivers the actual ≥0.85 metric on a held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flybrain.controller.action_space import (
    KIND_ACTIVATE_AGENT,
    KIND_NAMES,
    KIND_TERMINATE,
)
from flybrain.runtime.state import RuntimeState
from flybrain.sim.optimal_routes import (
    OPTIMAL_ROUTES,
    component_for_agent,
)
from flybrain.sim.task_generator import SyntheticTask, TaskGenerator

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from flybrain.controller._torch_base import TorchControllerBase


@dataclass(slots=True)
class PretrainConfig:
    n_per_type: int = 32
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 0
    weight_decay: float = 0.0
    aux_loss_weight: float = 0.1


@dataclass(slots=True)
class PretrainResult:
    losses: list[float] = field(default_factory=list)
    epoch_accuracy: list[float] = field(default_factory=list)
    final_accuracy: float = 0.0
    num_examples: int = 0


@dataclass(slots=True)
class _LabeledExample:
    runtime_state: RuntimeState
    agent_names: list[str]  # what was available at this step
    label_kind: int
    label_agent: int  # ignored when label_kind != activate_agent
    aux_target: float


def _runtime_state_for_step(
    task: SyntheticTask,
    step: int,
    *,
    available_agents: list[str],
) -> RuntimeState:
    """Build the synthetic RuntimeState the expert sees at step
    ``step`` of the optimal route for ``task``."""
    route = OPTIMAL_ROUTES[task.task_type]
    produced: set[str] = set()
    for prev in route[:step]:
        produced.add(component_for_agent(prev))
    return RuntimeState(
        task_id=task.task_id,
        task_type=task.task_type,
        prompt=task.prompt,
        step_id=step,
        available_agents=list(available_agents),
        pending_inbox={},
        last_active_agent=route[step - 1] if step > 0 else None,
        produced_components=produced,
    )


def expert_dataset(
    tasks: list[SyntheticTask],
    *,
    agent_names: list[str],
) -> list[_LabeledExample]:
    """Walk every task's optimal route and emit one labelled example
    per step (including the trailing ``terminate`` action)."""
    name_to_id = {n: i for i, n in enumerate(agent_names)}
    examples: list[_LabeledExample] = []
    for task in tasks:
        route = OPTIMAL_ROUTES.get(task.task_type)
        if route is None:
            continue
        for step in range(len(route) + 1):  # +1 for terminate
            rs = _runtime_state_for_step(task, step, available_agents=agent_names)
            if step >= len(route):
                examples.append(
                    _LabeledExample(
                        runtime_state=rs,
                        agent_names=agent_names,
                        label_kind=KIND_TERMINATE,
                        label_agent=0,
                        aux_target=1.0,
                    )
                )
            else:
                target_agent = route[step]
                if target_agent not in name_to_id:
                    continue
                examples.append(
                    _LabeledExample(
                        runtime_state=rs,
                        agent_names=agent_names,
                        label_kind=KIND_ACTIVATE_AGENT,
                        label_agent=name_to_id[target_agent],
                        # The aux head predicts the upcoming verifier
                        # score; in the simulation we simply ramp it
                        # linearly along the route.
                        aux_target=(step + 1) / max(1, len(route)),
                    )
                )
    return examples


def simulation_pretrain(
    controller: TorchControllerBase,
    *,
    agent_names: list[str],
    config: PretrainConfig | None = None,
    tasks: list[SyntheticTask] | None = None,
) -> PretrainResult:
    """Supervised pretraining loop.

    Subset of TorchControllerBase contract used here:
        - ``forward(controller_state) -> HeadOutputs``
        - ``builder.from_runtime_sync(runtime_state)``

    Returns a `PretrainResult` with per-step losses + per-epoch
    held-out accuracy.
    """
    import torch

    cfg = config or PretrainConfig()
    if tasks is None:
        gen = TaskGenerator(seed=cfg.seed)
        tasks = gen.balanced_dataset(n_per_type=cfg.n_per_type)

    examples = expert_dataset(tasks, agent_names=agent_names)
    if not examples:
        return PretrainResult(losses=[], epoch_accuracy=[], final_accuracy=0.0, num_examples=0)

    # Deterministic 80/20 split — examples come out of `expert_dataset`
    # grouped by task and step, so we shuffle (with a fixed RNG seeded
    # from the config) before splitting to avoid eval-set bias.
    import random as _random

    shuffled = list(examples)
    _random.Random(cfg.seed).shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_examples = shuffled[:split]
    eval_examples = shuffled[split:] or shuffled[-max(1, len(shuffled) // 5) :]

    optim = torch.optim.AdamW(
        controller.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    rng = torch.Generator()
    rng.manual_seed(cfg.seed)

    result = PretrainResult(num_examples=len(examples))

    for _epoch in range(cfg.epochs):
        # Shuffle train examples per epoch.
        order = torch.randperm(len(train_examples), generator=rng).tolist()
        for batch_start in range(0, len(order), cfg.batch_size):
            batch_idx = order[batch_start : batch_start + cfg.batch_size]
            optim.zero_grad()
            loss = _batch_loss(controller, train_examples, batch_idx, cfg)
            loss.backward()
            optim.step()
            result.losses.append(float(loss.item()))

        result.epoch_accuracy.append(_eval_accuracy(controller, eval_examples))

    result.final_accuracy = result.epoch_accuracy[-1] if result.epoch_accuracy else 0.0
    return result


def _batch_loss(
    controller: TorchControllerBase,
    examples: list[_LabeledExample],
    batch_idx: list[int],
    cfg: PretrainConfig,
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
        if ex.label_kind == KIND_ACTIVATE_AGENT and out.agent_logits.numel() > 0:
            agent_loss = torch_func.cross_entropy(
                out.agent_logits.unsqueeze(0),
                torch.tensor([ex.label_agent], dtype=torch.long),
            )
        else:
            agent_loss = torch.zeros(())
        aux_loss = torch_func.binary_cross_entropy(
            out.aux_verifier.unsqueeze(0),
            torch.tensor([ex.aux_target], dtype=out.aux_verifier.dtype).clamp(0.0, 1.0),
        )
        losses.append(kind_loss + agent_loss + cfg.aux_loss_weight * aux_loss)
    return torch.stack(losses).mean()


def _eval_accuracy(
    controller: TorchControllerBase,
    examples: list[_LabeledExample],
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
                agent_pred = int(torch.argmax(out.agent_logits).item())
                if agent_pred != ex.label_agent:
                    continue
            hits += 1
    return hits / len(examples)


__all__ = [
    "KIND_NAMES",
    "PretrainConfig",
    "PretrainResult",
    "expert_dataset",
    "simulation_pretrain",
]
