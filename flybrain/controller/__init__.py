"""Controllers consumed by `MAS.run`.

Phase 2 ships:

* `Controller` Protocol (the public contract).
* `ManualController` — scripted, hand-tuned plan per task type.
* `RandomController` — uniform-random baseline.

GNN / RNN / LearnedRouter controllers (Phase 5) plug into the same
Protocol.
"""

from __future__ import annotations

from flybrain.controller.base import Controller
from flybrain.controller.manual import ManualController
from flybrain.controller.random_ctrl import RandomController

__all__ = ["Controller", "ManualController", "RandomController"]
