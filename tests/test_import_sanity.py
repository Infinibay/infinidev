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

import subprocess
import sys

def _fresh_import(
    modname: str,
    *,
    exports: tuple[str, ...] = (),
    absent_modules: tuple[str, ...] = (),
) -> None:
    """Import one entry point in a clean interpreter and assert its surface.

    Removing live modules from ``sys.modules`` is not thread-safe: deferred
    embedding workers can still be importing them. A subprocess is both a
    truer cold-start check and isolated from pytest's background workers.
    """
    script = (
        "import importlib, sys\n"
        f"module = importlib.import_module({modname!r})\n"
        f"exports = {exports!r}\n"
        f"absent = {absent_modules!r}\n"
        "assert all(hasattr(module, name) for name in exports)\n"
        "assert all(name not in sys.modules for name in absent)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


class TestImportAcyclic:
    def test_cli_main_imports_cleanly(self):
        _fresh_import("infinidev.cli.main", exports=("main",))

    def test_pipeline_imports_cleanly(self):
        _fresh_import(
            "infinidev.engine.orchestration.pipeline", exports=("run_task",)
        )

    def test_planner_imports_cleanly(self):
        _fresh_import("infinidev.engine.analysis.planner", exports=("run_planner",))

    def test_chat_agent_imports_cleanly(self):
        _fresh_import(
            "infinidev.engine.orchestration.chat_agent", exports=("run_chat_agent",)
        )

    def test_orchestration_package_imports_cleanly(self):
        _fresh_import(
            "infinidev.engine.orchestration",
            exports=("run_task", "OrchestrationHooks"),
        )

    def test_guidance_library_imports_cleanly(self):
        _fresh_import("infinidev.engine.guidance.library", exports=("GuidanceEntry",))

    def test_guidance_public_exports_import_cleanly(self):
        _fresh_import(
            "infinidev.engine.guidance",
            exports=("detect_stuck_pattern", "drain_pending_guidance"),
        )

    def test_loop_models_imports_without_loading_loop_engine(self):
        _fresh_import(
            "infinidev.engine.loop.models",
            absent_modules=("infinidev.engine.loop.engine",),
        )
