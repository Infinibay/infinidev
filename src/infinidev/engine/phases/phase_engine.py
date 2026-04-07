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
from infinidev.engine.phases.investigator import (
    _investigate,
    _investigate_iteratively,
)
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

    def __init__(self) -> None:
        self._last_engine: LoopEngine | None = None
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
        depth_config: Any | None = None,
    ) -> str:
        from infinidev.gather.models import DepthLevel, DepthConfig, DEPTH_CONFIGS

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
                classification = self._classify(agent, description, verbose)
            task_type = classification.ticket_type.value
            depth_config = DEPTH_CONFIGS.get(classification.depth, DEPTH_CONFIGS[DepthLevel.standard])
            if verbose and not _is_small_model():
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: {classification.depth.value}")
                if classification.depth_reasoning:
                    _log(f"  {DIM}{classification.depth_reasoning}{RESET}")
        else:
            if verbose:
                _log(f"\n{BOLD}{CYAN}⚡ Phase Engine{RESET} — type: {task_type}, depth: (provided)")

        strategy = get_strategy(task_type)

        # ── MINIMAL: single free LoopEngine run ─────────────────
        if depth_config.skip_questions and depth_config.skip_investigate and depth_config.plan_min_steps <= 1:
            result, engine = _execute_minimal(agent, description, expected_output, strategy,
                                              task_tools, depth_config, verbose)
            self._last_engine = engine
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
                agent, description, strategy, task_tools, verbose,
                max_questions=depth_config.questions_max,
                skip_investigate=depth_config.skip_investigate,
            )

        # ── Phase 3: PLAN ───────────────────────────────────────
        if verbose:
            _log(f"\n{BOLD}📋 Phase 3: PLAN{RESET}")

        strategy.plan_min_steps = depth_config.plan_min_steps

        plan_steps = _generate_plan(
            agent, description, answers, all_notes, strategy, task_tools, verbose,
            test_checkpoint=self._test_checkpoint,
        )

        if not plan_steps:
            return "Failed to generate a valid plan."

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
            )
            self._last_engine = engine

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
                        agent, description, answers, all_notes, strategy, task_tools, verbose,
                    )
                    if not plan_steps:
                        break

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
        from infinidev.gather.models import DEPTH_CONFIGS

        # Init test checkpoint
        from infinidev.tools.base.context import get_current_workspace_path
        workdir = get_current_workspace_path()
        self._test_checkpoint = TestCheckpoint(None, workdir)

        # 1. Classify
        classification = self._classify(agent, task_description, verbose)
        depth_config = DEPTH_CONFIGS.get(classification.depth)
        task_type = classification.ticket_type.value
        strategy = get_strategy(task_type)

        # 2. Investigate
        strategy.investigate_max_tool_calls = depth_config.investigate_max_tool_calls
        answers, all_notes = _investigate_iteratively(
            agent, task_description, strategy, None, verbose=verbose,
            max_questions=depth_config.questions_max,
            skip_investigate=depth_config.skip_investigate,
        )

        # 3. Plan + review loop
        feedback_context = ""
        strategy.plan_min_steps = depth_config.plan_min_steps

        plan_steps: list[dict] = []
        while True:
            plan_desc = task_description
            if feedback_context:
                plan_desc += f"\n\n## USER FEEDBACK ON PREVIOUS PLAN\n{feedback_context}"

            plan_steps = _generate_plan(
                agent, plan_desc, answers, all_notes, strategy, None, verbose=verbose,
            )
            if not plan_steps:
                return "Failed to generate a plan."

            verdict, feedback = on_plan_ready(plan_steps)

            if verdict == "approve":
                break
            if verdict == "cancel":
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
            verbose=verbose, on_step_start=on_step_start,
        )
        self._last_engine = last_engine
        return result

    # ── Classify ─────────────────────────────────────────────────

    def _classify(self, agent: Any, description: str, verbose: bool) -> Any:
        """Run ticket classification to determine task_type and depth."""
        from infinidev.gather.classifier import classify_ticket

        if verbose:
            _log(f"\n{BOLD}🏷️  Step 0: CLASSIFY{RESET}")

        result = classify_ticket(description, agent=agent)

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

    def get_changed_files_summary(self) -> str:
        if self._last_engine:
            return self._last_engine.get_changed_files_summary()
        return ""

    def has_file_changes(self) -> bool:
        if self._last_engine:
            return self._last_engine.has_file_changes()
        return False
