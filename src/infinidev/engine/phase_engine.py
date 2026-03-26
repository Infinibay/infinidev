"""Phase-based execution engine: ANALYZE → PLAN → EXECUTE.

Orchestrates three phases using the LoopEngine internally:
1. ANALYZE: Read-only exploration, mandatory note-taking
2. PLAN: Generate granular plan, engine-validated
3. EXECUTE: Step-by-step implementation with test verification

Each phase uses task-type-specific prompts with concrete examples.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine.loop_engine import LoopEngine
from infinidev.engine.loop_models import LoopState
from infinidev.engine.phase_prompts import get_strategy, PhaseStrategy
from infinidev.engine.plan_validator import validate_plan, format_rejection
from infinidev.engine.test_checkpoint import TestCheckpoint
from infinidev.engine.engine_logging import (
    log as _log,
    emit_log as _emit_log,
    emit_loop_event as _emit_loop_event,
    DIM, BOLD, RESET, CYAN, GREEN, YELLOW, RED,
)

logger = logging.getLogger(__name__)

# Tools that are read-only (allowed in ANALYZE phase)
_READ_ONLY_TOOLS = {
    "read_file", "list_directory", "glob", "code_search",
    "project_structure", "find_definition", "find_references",
    "list_symbols", "search_symbols", "get_symbol_code",
    "search_knowledge", "read_findings", "search_findings",
    "web_search", "web_fetch",
}

# execute_command is allowed in ANALYZE but only for read operations
# The engine allows it but the prompt says "read-only commands only"
_ANALYZE_TOOLS = _READ_ONLY_TOOLS | {"execute_command"}


class PhaseEngine:
    """Three-phase execution: ANALYZE → PLAN → EXECUTE.

    Uses LoopEngine internally for each phase with different configs.
    """

    def __init__(self) -> None:
        self._last_file_tracker = None
        self._test_checkpoint: TestCheckpoint | None = None

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        task_type: str = "feature",
        *,
        verbose: bool = True,
        task_tools: list | None = None,
        test_command: str | None = None,
    ) -> str:
        """Run the full three-phase execution.

        Args:
            agent: InfinidevAgent instance
            task_prompt: (description, expected_output) tuple
            task_type: One of "bug", "feature", "refactor", "other", "sysadmin"
            verbose: Enable logging
            task_tools: Tools available to the agent
            test_command: Override test command (auto-detected if None)

        Returns:
            Final result string
        """
        strategy = get_strategy(task_type)
        description, expected_output = task_prompt
        all_tools = task_tools

        if verbose:
            _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — strategy: {task_type}")

        # Initialize test checkpoint
        from infinidev.tools.base.context import get_current_workspace_path
        workdir = get_current_workspace_path()
        self._test_checkpoint = TestCheckpoint(test_command, workdir)

        # ── Phase 1: ANALYZE ──────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📖 Phase 1: ANALYZE{RESET}")

        analyze_notes = self._run_analyze(
            agent, description, strategy, all_tools, verbose,
        )

        # ── Phase 2: PLAN ─────────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📋 Phase 2: PLAN{RESET}")

        plan_steps = self._run_plan(
            agent, description, analyze_notes, strategy, verbose,
        )

        if not plan_steps:
            return "Failed to generate a valid plan. Try with /think for deeper analysis."

        if verbose:
            _log(f"  {DIM}Plan: {len(plan_steps)} steps{RESET}")
            for s in plan_steps:
                files_str = ", ".join(s.get("files", [])) or "(verify)"
                _log(f"    {DIM}{s['step']}. {s['description'][:70]} [{files_str}]{RESET}")

        # ── Phase 3: EXECUTE ──────────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}🔨 Phase 3: EXECUTE{RESET}")

        result = self._run_execute(
            agent, description, expected_output, analyze_notes,
            plan_steps, strategy, all_tools, verbose,
        )

        # Final test run
        if strategy.auto_test and self._test_checkpoint:
            passed, total = self._test_checkpoint.run()
            if verbose:
                progress = self._test_checkpoint.progress_str()
                _log(f"\n  {BOLD}Final: {progress}{RESET}")

        self._last_file_tracker = getattr(self, '_last_engine', None)
        return result

    def _run_analyze(
        self,
        agent: Any,
        description: str,
        strategy: PhaseStrategy,
        all_tools: list | None,
        verbose: bool,
    ) -> list[str]:
        """Phase 1: Read-only analysis with mandatory note-taking.

        Returns the list of notes collected.
        """
        # Filter tools to read-only
        if all_tools:
            analyze_tools = [
                t for t in all_tools
                if getattr(t, 'name', '') in _ANALYZE_TOOLS
            ]
        else:
            analyze_tools = None  # Let the engine use agent's tools

        # Build analyze prompt
        analyze_prompt = (
            f"{strategy.analyze_prompt}\n\n"
            f"## YOUR TASK\n{description}\n\n"
            f"When done analyzing, call step_complete with status='done' "
            f"and a summary of your findings."
        )

        engine = LoopEngine()
        engine.execute(
            agent=agent,
            task_prompt=(analyze_prompt, "Complete analysis with notes."),
            verbose=verbose,
            task_tools=analyze_tools,
            max_iterations=5,
            max_total_tool_calls=strategy.analyze_max_tool_calls,
            nudge_threshold=0,  # Don't nudge in analyze — let it explore
            summarizer_enabled=False,
        )

        # Extract notes from engine state
        notes = []
        if engine._last_state:
            notes = list(engine._last_state.notes)

        if verbose:
            _log(f"  {DIM}Analysis complete: {len(notes)} notes collected{RESET}")

        return notes

    def _run_plan(
        self,
        agent: Any,
        description: str,
        notes: list[str],
        strategy: PhaseStrategy,
        verbose: bool,
    ) -> list[dict[str, Any]]:
        """Phase 2: Generate and validate a granular plan.

        Uses a direct LLM call (no tools, no loop) — the model just outputs
        a JSON plan based on the analysis notes.

        Returns validated list of step dicts, or empty list on failure.
        """
        from infinidev.engine.llm_client import call_llm
        from infinidev.config.llm import get_litellm_params
        from infinidev.engine.loop_context import build_system_prompt

        notes_text = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(notes))

        # Test baseline if auto_test enabled
        baseline_str = ""
        if strategy.auto_test and self._test_checkpoint:
            passed, total = self._test_checkpoint.run()
            if total > 0:
                baseline_str = f"\nTest baseline: {passed}/{total} passing\n"

        plan_prompt = (
            f"{strategy.plan_prompt}\n\n"
            f"## YOUR TASK\n{description}\n\n"
            f"## YOUR ANALYSIS NOTES\n{notes_text}\n"
            f"{baseline_str}\n"
            f"You have NO tools available. Do NOT call any tools.\n"
            f"Output ONLY a JSON array of steps. No other text, no markdown fences."
        )

        llm_params = get_litellm_params()
        system_prompt = build_system_prompt(
            agent.backstory,
            identity_override=getattr(agent, '_system_prompt_identity', None),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": plan_prompt},
        ]

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = call_llm(llm_params, messages)
                content = response.choices[0].message.content or ""
                content = content.strip()

                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                if content.startswith("json"):
                    content = content[4:].strip()

                if not content:
                    continue

                # Validate the plan
                is_valid, steps, errors = validate_plan(content, strategy)

                if is_valid:
                    return steps

                if verbose:
                    for err in errors:
                        _log(f"  {YELLOW}⚠ {err}{RESET}")

                if attempt < max_attempts - 1:
                    rejection = format_rejection(errors)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": rejection})
                    if verbose:
                        _log(f"  {DIM}Re-prompting with validation feedback...{RESET}")

            except Exception as exc:
                logger.warning("Plan generation failed (attempt %d): %s", attempt + 1, str(exc)[:200])
                if verbose:
                    _log(f"  {RED}⚠ Plan generation error: {str(exc)[:100]}{RESET}")

        logger.warning("Plan validation failed after %d attempts", max_attempts)
        return []

    def _run_execute(
        self,
        agent: Any,
        description: str,
        expected_output: str,
        notes: list[str],
        plan_steps: list[dict[str, Any]],
        strategy: PhaseStrategy,
        all_tools: list | None,
        verbose: bool,
    ) -> str:
        """Phase 3: Execute the plan step by step with verification.

        Returns the final result string.
        """
        notes_text = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(notes))
        completed_summaries: list[str] = []
        last_result = ""

        for step in plan_steps:
            step_num = step["step"]
            step_desc = step["description"]
            step_files = step.get("files", [])
            total_steps = len(plan_steps)

            # Build per-step prompt
            files_str = ", ".join(step_files) if step_files else "(no files — verification step)"
            completed_str = "\n".join(
                f"  ✓ Step {i + 1}: {s}" for i, s in enumerate(completed_summaries)
            ) if completed_summaries else "  (none yet)"

            # Progress info from test checkpoint
            progress_str = ""
            regression_warning = ""
            if self._test_checkpoint and self._test_checkpoint.total > 0:
                progress_str = f"\n{self._test_checkpoint.progress_str()}\n"
                regression_warning = self._test_checkpoint.regression_warning()
                if regression_warning:
                    regression_warning = f"\n{regression_warning}\n"

            # Format execute prompt with step details
            step_prompt = strategy.execute_prompt.replace(
                "{{step_num}}", str(step_num)
            ).replace(
                "{{total_steps}}", str(total_steps)
            ).replace(
                "{{step_description}}", step_desc
            ).replace(
                "{{step_files}}", files_str
            )

            full_prompt = (
                f"{step_prompt}\n\n"
                f"## ANALYSIS NOTES\n{notes_text}\n\n"
                f"## COMPLETED STEPS\n{completed_str}\n"
                f"{progress_str}"
                f"{regression_warning}"
            )

            if verbose:
                _log(f"\n  {CYAN}Step {step_num}/{total_steps}: {step_desc[:80]}{RESET}")

            # Run the step
            engine = LoopEngine()
            result = engine.execute(
                agent=agent,
                task_prompt=(full_prompt, expected_output),
                verbose=verbose,
                task_tools=all_tools,
                max_iterations=3,
                max_total_tool_calls=strategy.execute_max_tool_calls_per_step,
                nudge_threshold=strategy.execute_max_tool_calls_per_step - 2,
                summarizer_enabled=True,
            )

            # Store last engine for file tracker access
            self._last_engine = engine

            last_result = result or step_desc
            completed_summaries.append(f"{step_desc}: {(result or 'done')[:100]}")

            # Auto-run tests after code-modifying steps
            if strategy.auto_test and step_files and self._test_checkpoint:
                passed, total = self._test_checkpoint.run()
                if verbose:
                    progress = self._test_checkpoint.progress_str()
                    color = RED if self._test_checkpoint.has_regression() else GREEN
                    _log(f"    {color}{progress}{RESET}")

        return last_result

    # ── Public accessors (match LoopEngine interface) ─────────────────

    def get_changed_files_summary(self) -> str:
        if hasattr(self, '_last_engine') and self._last_engine:
            return self._last_engine.get_changed_files_summary()
        return ""

    def has_file_changes(self) -> bool:
        if hasattr(self, '_last_engine') and self._last_engine:
            return self._last_engine.has_file_changes()
        return False
