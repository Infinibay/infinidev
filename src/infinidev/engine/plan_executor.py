"""Phase 4: Plan execution for the phase engine."""

from __future__ import annotations

import logging
from typing import Any

from infinidev.engine.loop_engine import LoopEngine
from infinidev.engine.engine_logging import log as _log, DIM, BOLD, RESET, CYAN, GREEN, RED
from infinidev.prompts.phases import PhaseStrategy

logger = logging.getLogger(__name__)


def _execute_minimal(
    agent: Any,
    description: str,
    expected_output: str,
    strategy: PhaseStrategy,
    task_tools: list | None,
    depth_config: Any,
    verbose: bool,
) -> tuple[str, LoopEngine]:
    """Minimal depth: single LoopEngine run with no phase separation.

    Returns (result_string, engine) so the orchestrator can track the engine.
    """
    if verbose:
        _log(f"\n{BOLD}🔨 EXECUTE (minimal — single run){RESET}")

    engine = LoopEngine()
    result = engine.execute(
        agent=agent,
        task_prompt=(description, expected_output),
        verbose=verbose,
        task_tools=task_tools,
        max_iterations=50,
        max_total_tool_calls=1000,
        max_tool_calls_per_action=0,
        nudge_threshold=0,
        summarizer_enabled=True,
        identity_override=strategy.execute_identity or None,
    )

    return result or "", engine


def _execute_plan(
    agent: Any,
    description: str,
    expected_output: str,
    answers: list[dict[str, str]],
    all_notes: list[str],
    plan_steps: list[dict[str, Any]],
    strategy: PhaseStrategy,
    all_tools: list | None,
    depth_config: Any,
    verbose: bool,
    test_checkpoint: Any | None = None,
) -> tuple[str, LoopEngine]:
    """Execute each plan step via LoopEngine.

    Returns (result_string, last_engine) so the orchestrator can track changes.
    """
    answers_text = "\n".join(
        f"  Q: {a['question']}\n  A: {a['answer']}"
        for a in answers
    )
    notes_text = ""
    if all_notes:
        notes_text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(all_notes))
    completed: list[str] = []
    last_result = ""
    last_engine: LoopEngine | None = None

    for step in plan_steps:
        step_num = step["step"]
        step_desc = step["description"]
        step_files = step.get("files", [])
        total = len(plan_steps)

        files_str = ", ".join(step_files) if step_files else "(verification step)"
        completed_str = "\n".join(
            f"  ✓ {s}" for s in completed
        ) if completed else "  (none yet)"

        progress_str = ""
        regression_warning = ""
        if test_checkpoint and test_checkpoint.total > 0:
            progress_str = f"\n{test_checkpoint.progress_str()}\n"
            rw = test_checkpoint.regression_warning()
            if rw:
                regression_warning = f"\n{rw}\n"

        # Use conditional prompt generation for small models
        from infinidev.config.llm import _is_small_model
        if _is_small_model():
            from infinidev.prompts.tool_hints import (
                build_execute_prompt, get_available_tool_names,
            )
            _tool_names = get_available_tool_names(all_tools)
            step_prompt = build_execute_prompt(
                available_tools=_tool_names,
                step_num=step_num,
                total_steps=total,
                step_description=step_desc,
                step_files=files_str,
            )
        else:
            step_prompt = strategy.execute_prompt.replace(
                "{{step_num}}", str(step_num)
            ).replace(
                "{{total_steps}}", str(total)
            ).replace(
                "{{step_description}}", step_desc
            ).replace(
                "{{step_files}}", files_str
            )

        notes_section = f"## NOTES\n{notes_text}\n\n" if notes_text else ""
        depth_suffix = depth_config.prompt_suffix if depth_config else ""
        full_prompt = (
            f"{step_prompt}\n\n"
            f"{notes_section}"
            f"## INVESTIGATION RESULTS\n{answers_text}\n\n"
            f"## COMPLETED STEPS\n{completed_str}\n"
            f"{progress_str}{regression_warning}"
            f"{depth_suffix}"
        )

        if verbose:
            _log(f"\n  {CYAN}Step {step_num}/{total}: {step_desc[:80]}{RESET}")

        engine = LoopEngine()
        result = engine.execute(
            agent=agent,
            task_prompt=(full_prompt, expected_output),
            verbose=verbose,
            task_tools=all_tools,
            max_iterations=50,
            max_total_tool_calls=1000,
            max_tool_calls_per_action=0,
            nudge_threshold=0,
            summarizer_enabled=not (depth_config and depth_config.aggressive_summarizer),
            identity_override=strategy.execute_identity or None,
            allow_only_add_steps=depth_config.allow_only_add_steps if depth_config else True,
            reject_write_on_existing=depth_config.reject_write_on_existing if depth_config else False,
            require_test_before_complete=depth_config.require_test_before_complete if depth_config else False,
        )

        last_engine = engine
        last_result = result or step_desc
        completed.append(f"Step {step_num}: {step_desc}: {(result or 'done')[:80]}")

        # Auto-test after code-modifying steps
        if strategy.auto_test and step_files and test_checkpoint:
            passed, total_tests = test_checkpoint.run()
            if verbose and total_tests > 0:
                progress = test_checkpoint.progress_str()
                color = RED if test_checkpoint.has_regression() else GREEN
                _log(f"    {color}{progress}{RESET}")

    return last_result, last_engine or LoopEngine()
