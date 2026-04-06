"""Tool: serve a filtered view of the most recent test runner output.

Small models often re-run tests just to see the failure message
because they don't know how to use shell pipes (``2>&1 | tail -40``,
``| grep FAIL``). This tool exists so they don't have to: the engine
captures the stdout/stderr of the most recent ``execute_command`` that
``is_test_command`` recognised, and this tool returns a filtered slice
of it on demand. Zero re-run cost.

Three modes, picked via the ``mode`` argument (default: ``"tail"``):

  * **tail** — return the last N raw lines of the captured output.
    Cheap and predictable. Good when the model wants to see the
    pytest/jest summary line and the last failure traceback.

  * **failures** — return only lines that contain a failure marker
    (E, FAILED, AssertionError, ●, --- FAIL, panicked at, etc.).
    Good when the test output is huge and the model only needs to
    see WHAT failed, not the full session log. Generic regex.

  * **structured** — invoke the runner-specific parser
    (``engine.test_parsers.parse_test_failures``) and return a list
    of structured ``ParsedFailure`` dicts with ``test_name``,
    ``file``, ``line``, ``error_type``, and ``message`` per failure.
    The HIGHEST signal-to-noise mode — typically 5-10x fewer tokens
    than ``tail`` for the same information. Auto-detects pytest /
    jest / vitest / mocha / go test / cargo test from the output.

Returns an error if no test has been run yet in the current task.
"""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


# Substrings that mark a "failure-related" line across the runners we
# already recognise in engine/guidance/test_runners.py.
_FAILURE_MARKERS: tuple[str, ...] = (
    "E   ",            # pytest assertion / exception lines
    "E    ",
    "FAILED",
    "FAIL ",
    "FAIL\t",
    "--- FAIL",        # go test
    "✗",               # mocha / vitest
    "●",               # jest fail block header
    "AssertionError",
    "assert",
    "panicked at",     # rust
    "thread '",
    "<<< FAILURE!",    # mvn / surefire
    "[xUnit.net",      # dotnet
    "Error:",
    "ERROR:",
    "Traceback",
    "expected",
    "Expected",
)


class TailTestOutputInput(BaseModel):
    mode: str = Field(
        default="tail",
        description=(
            "How to filter the captured output. One of: "
            "'tail' (last N raw lines), "
            "'failures' (only lines with failure markers — generic regex), "
            "'structured' (runner-specific parser returning a JSON list of "
            "ParsedFailure objects with test_name/file/line/error_type/message). "
            "'structured' is the highest-signal mode — use it first."
        ),
    )
    lines: int = Field(
        default=40,
        description="How many lines to return in tail/failures mode. Capped at 200.",
        ge=1,
        le=200,
    )
    runner_hint: str = Field(
        default="",
        description=(
            "Optional runner name for 'structured' mode: pytest, jest, "
            "go, cargo, mocha. Leave empty to auto-detect from output."
        ),
    )


class TailTestOutputTool(InfinibayBaseTool):
    name: str = "tail_test_output"
    description: str = (
        "Show a filtered view of the LAST test runner output captured "
        "by the engine. Use this AFTER running pytest/jest/cargo test/etc. "
        "via execute_command, to see the failure messages without "
        "re-running the tests or scrolling through huge logs. "
        "Modes: "
        "'structured' (runner-specific parser, returns a JSON list of "
        "ParsedFailure with test_name/file/line/error_type/message — "
        "USE THIS FIRST, it's the highest signal-to-noise option), "
        "'failures' (only lines with failure markers), "
        "'tail' (last N raw lines, default). "
        "Returns an error if no test has been run yet in this task."
    )
    args_schema: Type[BaseModel] = TailTestOutputInput

    def _run(self, mode: str = "tail", lines: int = 40, runner_hint: str = "") -> str:
        from infinidev.tools.base.context import get_context_for_agent
        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not getattr(ctx, "loop_state", None):
            return self._error("No active loop context")

        state = ctx.loop_state
        captured = getattr(state, "last_test_output", "") or ""
        if not captured:
            return self._error(
                "No test output captured yet. Run a test command first "
                "(e.g. `execute_command('pytest -v')`), then call this tool."
            )

        last_cmd = getattr(state, "last_test_command", "") or "(unknown)"
        mode = (mode or "tail").lower().strip()

        # ── structured: runner-specific parsed failures ────────────
        if mode == "structured":
            try:
                from infinidev.engine.test_parsers import parse_test_failures
                failures = parse_test_failures(captured, runner_hint=runner_hint or None)
            except Exception as exc:
                return self._error(f"Parser failed: {exc}")
            if not failures:
                return self._success({
                    "command": last_cmd,
                    "mode": "structured",
                    "failures": [],
                    "note": (
                        "No failures parsed. Either all tests passed, "
                        "or the runner output format isn't recognised. "
                        "Try mode='tail' to see the raw output."
                    ),
                })
            return self._success({
                "command": last_cmd,
                "mode": "structured",
                "runner": failures[0].runner,
                "failure_count": len(failures),
                "failures": [f.to_dict() for f in failures],
            })

        all_lines = captured.splitlines()

        # ── failures: generic regex line filter ───────────────────
        if mode == "failures":
            kept = [
                line for line in all_lines
                if any(marker in line for marker in _FAILURE_MARKERS)
            ]
            if not kept:
                return self._success({
                    "command": last_cmd,
                    "mode": "failures",
                    "matched_lines": 0,
                    "note": "No failure markers found. The tests may have all passed.",
                })
            shown = kept[-lines:] if len(kept) > lines else kept
            return self._success({
                "command": last_cmd,
                "mode": "failures",
                "matched_lines": len(kept),
                "shown_lines": len(shown),
                "output": "\n".join(shown),
            })

        # ── tail (default): last N raw lines ──────────────────────
        shown = all_lines[-lines:] if len(all_lines) > lines else all_lines
        truncated_above = max(0, len(all_lines) - len(shown))
        result: dict = {
            "command": last_cmd,
            "mode": "tail",
            "total_lines": len(all_lines),
            "shown_lines": len(shown),
            "output": "\n".join(shown),
        }
        if truncated_above:
            result["truncated_above"] = truncated_above
        return self._success(result)
