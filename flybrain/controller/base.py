"""`Controller` Protocol — the contract every controller must satisfy.

Each tick, the runner calls `controller.select_action(state) -> dict`
where the dict is a JSON-serialised `GraphAction`
(see `crates/flybrain-core/src/action.rs`). The same protocol is
satisfied by:

* `ManualController` (Phase 2, scripted plans),
* `RandomController` (Phase 2 / Phase 9 baseline),
* `LearnedRouterController` (Phase 5),
* `FlyBrainGNNController` and `FlyBrainRNNController` (Phase 5).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from flybrain.runtime.state import RuntimeState


@runtime_checkable
class Controller(Protocol):
    name: str

    def select_action(self, state: RuntimeState) -> dict[str, Any]: ...
