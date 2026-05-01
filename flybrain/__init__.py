"""FlyBrain Optimizer Python package.

The Python layer wraps the Rust core (`flybrain_native`) and provides:

* LLM clients (Yandex AI Studio + mock) under `flybrain.llm`.
* Agent specs and runtime glue under `flybrain.agents` / `flybrain.runtime`.
* Embeddings, controllers, training, baselines, benchmarks, and evaluation —
  see the corresponding submodules.

Phase 0 ships only the LLM clients and type round-trip helpers; everything
else is a documented placeholder. See `PLAN.md` for the full roadmap.
"""

from __future__ import annotations

try:
    import importlib

    native = importlib.import_module("flybrain.flybrain_native")
except ImportError:  # pragma: no cover
    native = None  # type: ignore[assignment]

__all__ = ["__version__", "native"]
__version__ = "0.1.0"
