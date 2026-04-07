"""UI-agnostic task orchestration.

This package owns the full task pipeline (analysis → gather → execute →
review) as a single function ``run_task`` that knows nothing about
``click``, ``prompt_toolkit``, ``threading.Event``, or any UI framework.
Every interaction with the outside world goes through an
``OrchestrationHooks`` Protocol that the caller implements.

Why this exists:

  Before this module, the same pipeline was implemented THREE separate
  times — once in ``cli/main.py::_run_main`` (interactive classic),
  once in ``cli/main.py::_run_single_prompt`` (one-shot ``--prompt``),
  and once in ``ui/workers.py::run_engine_task`` (TUI). Improvements
  to one path silently failed to reach the other two. Now there is
  exactly one pipeline; each entry point is a thin adapter that
  instantiates the appropriate hooks implementation and calls
  :func:`run_task`.

Public API:

  * :class:`OrchestrationHooks` — Protocol every caller must implement
  * :func:`run_task`            — the unified pipeline entry point
  * :class:`NoOpHooks`          — silent default for tests
  * :class:`ClickHooks`         — terminal output via ``click``
  * :class:`NonInteractiveHooks` — refuses questions, used by ``--prompt``
"""

from infinidev.engine.orchestration.pipeline import (
    OrchestrationHooks,
    run_task,
    run_flow_task,
)
from infinidev.engine.orchestration.hooks import (
    NoOpHooks,
    ClickHooks,
    NonInteractiveHooks,
)

__all__ = [
    "OrchestrationHooks",
    "run_task",
    "run_flow_task",
    "NoOpHooks",
    "ClickHooks",
    "NonInteractiveHooks",
]
