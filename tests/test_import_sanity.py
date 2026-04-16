"""Smoke tests that the top-level imports don't deadlock in a cycle.

A prior commit hoisted ``run_chat_agent`` and ``run_planner`` to
top-level imports in ``engine/orchestration/pipeline.py``. That
triggered a circular import at CLI startup because
``engine/analysis/planner.py`` imports from
``engine/orchestration/escalation_packet``, which in turn triggers
``engine/orchestration/__init__.py`` — which eagerly imports
``pipeline`` — which was now importing back into the still-loading
``planner``. The CLI crashed the moment the user typed ``infinidev``.

These tests load the public entry points fresh (so a previously
cached module doesn't hide the regression) and assert nothing raises.
They don't exercise behaviour — just the import graph.
"""

from __future__ import annotations

import importlib
import sys


def _fresh_import(modname: str):
    """Drop every cached module under ``infinidev`` and re-import *modname*.

    Required because pytest's test order means the modules under test
    may already be loaded from an earlier test — and a bad top-level
    import only fails on the FIRST load. Without this, the regression
    would be invisible in CI.
    """
    for name in [m for m in sys.modules if m.startswith("infinidev")]:
        del sys.modules[name]
    return importlib.import_module(modname)


class TestImportAcyclic:
    def test_cli_main_imports_cleanly(self):
        mod = _fresh_import("infinidev.cli.main")
        assert hasattr(mod, "main")

    def test_pipeline_imports_cleanly(self):
        mod = _fresh_import("infinidev.engine.orchestration.pipeline")
        assert hasattr(mod, "run_task")

    def test_planner_imports_cleanly(self):
        mod = _fresh_import("infinidev.engine.analysis.planner")
        assert hasattr(mod, "run_planner")

    def test_chat_agent_imports_cleanly(self):
        mod = _fresh_import("infinidev.engine.orchestration.chat_agent")
        assert hasattr(mod, "run_chat_agent")

    def test_orchestration_package_imports_cleanly(self):
        mod = _fresh_import("infinidev.engine.orchestration")
        assert hasattr(mod, "run_task")
        assert hasattr(mod, "OrchestrationHooks")
