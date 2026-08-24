"""Phase-based execution: QUESTIONS → INVESTIGATE → PLAN → EXECUTE.

Orchestrates four phases by delegating to specialized components:
- QuestionGenerator: generates investigation questions
- Investigator: answers questions by reading the codebase
- PlanGenerator: creates an implementation plan from findings
- PlanExecutor: executes the plan step-by-step

Each component is in its own module (question_generator.py, investigator.py,
plan_generator.py, plan_executor.py).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from infinidev.engine.loop import LoopEngine
from infinidev.engine.test_checkpoint import TestCheckpoint
from infinidev.engine.engine_logging import (
    log as _log,
    DIM, BOLD, RESET, CYAN, GREEN, YELLOW, RED,
)
from infinidev.prompts.phases import get_strategy, PhaseStrategy

# Phase components
from infinidev.engine.phases.question_generator import (
    _generate_questions,
    _generate_followups,
)
from infinidev.engine.phases.investigator import _investigate_iteratively
from infinidev.engine.phases.plan_generator import _generate_plan
from infinidev.engine.phases.plan_executor import (
    _execute_minimal,
    _execute_plan,
)

logger = logging.getLogger(__name__)


class PhaseEngine:
    """Four-phase execution: QUESTIONS → INVESTIGATE → PLAN → EXECUTE.

    Thin orchestrator that delegates each phase to its component module.
    """

    def __init__(self, loop_engine: LoopEngine | None = None) -> None:
        self._execution_engine = loop_engine or LoopEngine()
        self._last_engine: LoopEngine | None = None
        self._last_plan_steps: list[dict] = []
        self._test_checkpoint: TestCheckpoint | None = None
        self._terminal_status: str = ""

    def execute(
        self,
        agent: Any,
        task_prompt: tuple[str, str],
        task_type: str = "feature",
        *,
        verbose: bool = True,
        task_tools: list | None = None,
        test_command: str | None = None,
        depth_config: Any | None = None,
        prompt_configuration: Any | None = None,
    ) -> str:
        self._last_engine = self._execution_engine
        self._last_plan_steps = []
        self._terminal_status = ""
        self._execution_engine._last_status = ""

        from infinidev.gather.models import DepthLevel, DepthConfig, DEPTH_CONFIGS
        from infinidev.prompts.profiles import EffectivePromptConfiguration

        prompt_configuration = (
            prompt_configuration or EffectivePromptConfiguration.compile()
        )
        description, expected_output = task_prompt

        # Init test checkpoint
        from infinidev.tools.base.context import get_current_workspace_path
        workdir = get_current_workspace_path()
        self._test_checkpoint = TestCheckpoint(test_command, workdir)

        # ── Step 0: CLASSIFY ─────────────────────────────────────
        from infinidev.config.llm import _is_small_model
        if depth_config is None:
            if _is_small_model():
                # Skip classification LLM call for small models — saves
                # an entire round-trip and small models get it wrong anyway.
                from infinidev.gather.models import ClassificationResult, TicketType
                classification = ClassificationResult(
                    ticket_type=TicketType.feature,
                    reasoning="Small model — skipped classification.",
                    depth=DepthLevel.standard,
                    depth_reasoning="Small model default.",
                )
                if verbose:
                    _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: feature, depth: standard (small model — skipped classify)")
            else:
                classification = self._classify(
                    agent, description, verbose, prompt_configuration
                )
            task_type = classification.ticket_type.value
            depth_config = DEPTH_CONFIGS.get(classification.depth, DEPTH_CONFIGS[DepthLevel.standard])
            if verbose and not _is_small_model():
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: {classification.depth.value}")
                if classification.depth_reasoning:
                    _log(f"  {DIM}{classification.depth_reasoning}{RESET}")
        else:
            if verbose:
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: (provided)")

        if self._cancel_requested():
            return self._finish_cancelled()

        strategy = get_strategy(task_type)

        # ── MINIMAL: single free LoopEngine run ─────────────────
        if depth_config.skip_questions and depth_config.skip_investigate and depth_config.plan_min_steps <= 1:
            result, engine = _execute_minimal(
                agent,
                description,
                expected_output,
                strategy,
                task_tools,
                depth_config,
                verbose,
                prompt_configuration=prompt_configuration,
                loop_engine=self._execution_engine,
            )
            self._last_engine = engine
            if self._cancel_requested():
                cancelled_result = self._finish_cancelled()
                return result or cancelled_result
            if strategy.auto_test and self._test_checkpoint:
                passed, total = self._test_checkpoint.run()
                if verbose and total > 0:
                    _log(f"\n  {BOLD}{self._test_checkpoint.progress_str()}{RESET}")
            return result

        # ── Phase 1+2: QUESTIONS + INVESTIGATE ──────────────────
        answers: list[dict[str, str]] = []
        all_notes: list[str] = []

        if not depth_config.skip_questions:
            strategy.investigate_max_tool_calls = depth_config.investigate_max_tool_calls

            answers, all_notes = _investigate_iteratively(
                agent,
                description,
                strategy,
                task_tools,
                verbose,
                max_questions=depth_config.questions_max,
                skip_investigate=depth_config.skip_investigate,
                prompt_configuration=prompt_configuration,
                loop_engine=self._execution_engine,
                cancel_check=self._cancel_requested,
            )
            if self._cancel_requested():
                return self._finish_cancelled()

        # ── Phase 3: PLAN ───────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📋 Phase 3: PLAN{RESET}")

        strategy.plan_min_steps = depth_config.plan_min_steps

        plan_steps = _generate_plan(
            agent,
            description,
            answers,
            all_notes,
            strategy,
            task_tools,
            verbose,
            test_checkpoint=self._test_checkpoint,
            prompt_configuration=prompt_configuration,
            cancel_check=self._cancel_requested,
            max_rounds=depth_config.plan_max_rounds,
        )

        if self._cancel_requested():
            return self._finish_cancelled()
        if not plan_steps:
            self._last_status = "failed"
            return "Failed to generate a valid plan."

        self._last_plan_steps = plan_steps

        if verbose:
            _log(f"  {DIM}Plan: {len(plan_steps)} steps{RESET}")
            for s in plan_steps:
                files_str = ", ".join(s.get("files", [])) or "(verify)"
                _log(f"    {DIM}{s['step']}. {s.get('title', s.get('explanation', ''))[:70]} [{files_str}]{RESET}")

        # ── Phase 4: EXECUTE (with re-plan loop) ────────────────
        result = ""
        for plan_round in range(depth_config.replan_max_rounds):
            if verbose:
                round_label = f" (round {plan_round + 1})" if plan_round > 0 else ""
                _log(f"\n{BOLD}🔨 Phase 4: EXECUTE{round_label}{RESET}")

            result, engine = _execute_plan(
                agent, description, expected_output, answers, all_notes,
                plan_steps, strategy, task_tools, depth_config, verbose,
                test_checkpoint=self._test_checkpoint,
                prompt_configuration=prompt_configuration,
                loop_engine=self._execution_engine,
                preserve_file_tracker=plan_round > 0,
            )
            self._last_engine = engine
            if self._cancel_requested():
                cancelled_result = self._finish_cancelled()
                return result or cancelled_result
            if self._last_status not in {"done", "completed"}:
                return result

            # Check test progress
            if strategy.auto_test and self._test_checkpoint:
                passed, total = self._test_checkpoint.run()
                if verbose and total > 0:
                    _log(f"\n  {BOLD}{self._test_checkpoint.progress_str()}{RESET}")

                if total == 0 or passed == total:
                    break

                if plan_round < depth_config.replan_max_rounds - 1:
                    if verbose:
                        _log(f"\n{BOLD}📋 Re-planning: {passed}/{total} tests passing...{RESET}")

                    all_notes.append(f"PROGRESS: {passed}/{total} tests passing after round {plan_round + 1}")

                    plan_steps = _generate_plan(
                        agent,
                        description,
                        answers,
                        all_notes,
                        strategy,
                        task_tools,
                        verbose,
                        prompt_configuration=prompt_configuration,
                        cancel_check=self._cancel_requested,
                        max_rounds=depth_config.plan_max_rounds,
                    )
                    if self._cancel_requested():
                        return self._finish_cancelled()
                    if not plan_steps:
                        break

                    self._last_plan_steps = plan_steps

                    if verbose:
                        _log(f"  {DIM}Re-plan: {len(plan_steps)} new steps{RESET}")
                        for s in plan_steps:
                            _log(f"    {DIM}{s['step']}. {s.get('title', s.get('explanation', ''))[:70]}{RESET}")
            else:
                break

        return result

    # ── Public: plan-review interactive flow ────────────────────────────

    def execute_with_plan_review(
        self,
        *,
        agent: Any,
        task_description: str,
        expected_output: str = "Complete the task.",
        on_plan_ready: Callable[[list[dict]], tuple[str, str]],
        on_step_start: Callable[[int, int, list, list], None] | None = None,
        verbose: bool = True,
        prompt_configuration: Any | None = None,
    ) -> str:
        """Run the phase pipeline with a human-in-the-loop plan review.

        This is the supported public entry point for the ``/plan`` flow
        in the TUI (and any future caller that wants the same behaviour).
        It replaces direct calls to the underscore-prefixed phase
        functions (``_classify``, ``_investigate_iteratively``,
        ``_generate_plan``, ``_execute_plan``) that the TUI worker used
        to make — those are now considered private to this module.

        The pipeline:
            1. **Classify** the task to pick a depth profile.
            2. **Investigate** the codebase iteratively.
            3. **Plan + review loop**: generate a plan, hand it to
               *on_plan_ready*, and act on the verdict:
                 * ``("approve", "")``  → break out of the loop and execute.
                 * ``("cancel", "")``   → return ``"Plan cancelled."`` immediately.
                 * ``("feedback", txt)`` → re-generate the plan with the
                   user feedback appended to the task description.
            4. **Execute** the approved plan, optionally calling
               *on_step_start* before each step starts so a UI can refresh.

        *on_plan_ready* MUST be safe to call from a worker thread. The
        intended pattern is for the caller to block on a UI event and
        return only once the user has answered — the TUI does this with
        ``threading.Event``; a CLI could use ``input()``.
        """
        self._last_engine = self._execution_engine
        self._last_plan_steps = []
        self._terminal_status = ""
        self._execution_engine._last_status = ""

        from infinidev.gather.models import DEPTH_CONFIGS
        from infinidev.prompts.profiles import EffectivePromptConfiguration

        prompt_configuration = (
            prompt_configuration or EffectivePromptConfiguration.compile()
        )
        # Init test checkpoint
        from infinidev.tools.base.context import get_current_workspace_path
        workdir = get_current_workspace_path()
        self._test_checkpoint = TestCheckpoint(None, workdir)

        # 1. Classify
        classification = self._classify(
            agent, task_description, verbose, prompt_configuration
        )
        depth_config = DEPTH_CONFIGS.get(classification.depth)
        task_type = classification.ticket_type.value
        if self._cancel_requested():
            return self._finish_cancelled()

        strategy = get_strategy(task_type)

        # 2. Investigate
        strategy.investigate_max_tool_calls = depth_config.investigate_max_tool_calls
        answers, all_notes = _investigate_iteratively(
            agent,
            task_description,
            strategy,
            None,
            verbose=verbose,
            max_questions=depth_config.questions_max,
            skip_investigate=depth_config.skip_investigate,
            prompt_configuration=prompt_configuration,
            loop_engine=self._execution_engine,
            cancel_check=self._cancel_requested,
        )
        if self._cancel_requested():
            return self._finish_cancelled()

        # 3. Plan + review loop
        feedback_context = ""
        strategy.plan_min_steps = depth_config.plan_min_steps

        plan_steps: list[dict] = []
        while True:
            plan_desc = task_description
            if feedback_context:
                plan_desc += f"\n\n## USER FEEDBACK ON PREVIOUS PLAN\n{feedback_context}"

            plan_steps = _generate_plan(
                agent,
                plan_desc,
                answers,
                all_notes,
                strategy,
                None,
                verbose=verbose,
                prompt_configuration=prompt_configuration,
                cancel_check=self._cancel_requested,
                max_rounds=depth_config.plan_max_rounds,
            )
            if self._cancel_requested():
                return self._finish_cancelled()
            if not plan_steps:
                self._last_status = "failed"
                return "Failed to generate a plan."

            verdict, feedback = on_plan_ready(plan_steps)

            if verdict == "approve":
                self._last_plan_steps = plan_steps
                break
            if verdict == "cancel":
                self._last_status = "cancelled"
                return "Plan cancelled."
            # "feedback" → loop back with feedback context
            feedback_context = feedback

        # 4. Execute (single round here — the auto re-plan loop in
        # `execute()` is intentionally not used in plan-review mode
        # because the user has already approved this plan and a silent
        # re-plan would surprise them).
        result, last_engine = _execute_plan(
            agent, task_description, expected_output,
            answers, all_notes, plan_steps, strategy, None, depth_config,
            verbose=verbose,
            on_step_start=on_step_start,
            prompt_configuration=prompt_configuration,
            loop_engine=self._execution_engine,
        )
        self._last_engine = last_engine
        if self._cancel_requested():
            cancelled_result = self._finish_cancelled()
            return result or cancelled_result
        return result

    # ── Classify ─────────────────────────────────────────────────

    def _cancel_requested(self) -> bool:
        """Return whether the owning turn requested full task cancellation."""
        return bool(getattr(self._execution_engine, "is_cancelled", False))

    def _finish_cancelled(self) -> str:
        """Record a cancelled terminal outcome at a phase boundary."""
        self._last_status = "cancelled"
        return "Task cancelled by user."

    def _classify(
        self,
        agent: Any,
        description: str,
        verbose: bool,
        prompt_configuration: Any | None = None,
    ) -> Any:
        """Run ticket classification to determine task_type and depth."""
        from infinidev.gather.classifier import classify_ticket

        if verbose:
            _log(f"\n{BOLD}🏷️  Step 0: CLASSIFY{RESET}")

        result = classify_ticket(
            description,
            agent=agent,
            prompt_configuration=prompt_configuration,
        )

        if verbose:
            _log(f"  {DIM}Type: {result.ticket_type.value} — {result.reasoning}{RESET}")
            _log(f"  {DIM}Depth: {result.depth.value} — {result.depth_reasoning}{RESET}")

        return result

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_llm_text(response: Any) -> str:
        """Extract and clean text from LLM response."""
        content = response.choices[0].message.content or ""
        content = content.strip()
        content = re.sub(
            r"<(?:think|thinking)>.*?</(?:think|thinking)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        )
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        if content.startswith("json"):
            content = content[4:].strip()
        return content

    @property
    def _last_status(self) -> str:
        if self._last_engine is not None:
            return str(getattr(self._last_engine, "_last_status", "") or "")
        return self._terminal_status

    @_last_status.setter
    def _last_status(self, status: str) -> None:
        self._terminal_status = str(status or "")
        if self._last_engine is not None:
            self._last_engine._last_status = self._terminal_status

    @property
    def _last_state(self) -> Any | None:
        if self._last_engine is None:
            return None
        return getattr(self._last_engine, "_last_state", None)

    @property
    def _last_total_tool_calls(self) -> int:
        if self._last_engine is None:
            return 0
        return int(getattr(self._last_engine, "_last_total_tool_calls", 0) or 0)

    def cancel(self) -> None:
        """Cancel the full phase task through its shared LoopEngine."""
        self._execution_engine.cancel()

    def cancel_active_tool(self) -> bool:
        """Cancel only the current foreground tool batch."""
        return self._execution_engine.cancel_active_tool()

    @property
    def has_active_tool(self) -> bool:
        """Return whether the shared LoopEngine is running tools."""
        return bool(self._execution_engine.has_active_tool)

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested()

    def get_objective_checks(self) -> list[dict[str, Any]]:
        if self._last_engine is None:
            return []
        getter = getattr(self._last_engine, "get_objective_checks", None)
        return list(getter() or []) if callable(getter) else []

    def build_work_summary(self, result: str, status: str) -> str | None:
        if self._last_engine is None:
            return None
        builder = getattr(self._last_engine, "build_work_summary", None)
        if not callable(builder):
            return None
        return builder(result, status)

    def get_changed_files_summary(self) -> str:
        if self._last_engine:
            return self._last_engine.get_changed_files_summary()
        return ""

    def has_file_changes(self) -> bool:
        if self._last_engine:
            return self._last_engine.has_file_changes()
        return False

    def get_plan_steps(self) -> list[dict]:
        """Return the last plan used for execution (empty if none planned)."""
        return list(self._last_plan_steps)

    def get_file_contents(self) -> dict[str, str]:
        if self._last_engine:
            return self._last_engine.get_file_contents()
        return {}

    def get_file_change_reasons(self) -> dict[str, list[str]]:
        if self._last_engine:
            return self._last_engine.get_file_change_reasons()
        return {}

    def get_file_tracker(self):
        if self._last_engine:
            return self._last_engine.get_file_tracker()
        return None
