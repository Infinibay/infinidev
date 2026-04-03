"""CLI pipeline phases — extracted from the main while loop.

Each phase is a standalone function that takes session context in and
returns results out. The main loop orchestrates the pipeline:
command → analysis → gather → execution → review.

Uses the Pipeline pattern: each phase transforms/enriches the task context.
"""

from __future__ import annotations

import logging
from typing import Any

import click

logger = logging.getLogger(__name__)


def run_analysis_phase(
    user_input: str,
    analyst: Any,
    session: Any,
) -> tuple[tuple[str, str], Any, str]:
    """Run the analysis phase: classify task, ask questions, build spec.

    Returns (task_prompt, analysis, flow_name).
    If analysis is disabled, returns defaults.
    """
    from infinidev.config.settings import settings

    if not settings.ANALYSIS_ENABLED:
        return (user_input, "Complete the task and report findings."), None, "develop"

    analyst.reset()
    click.echo(click.style("Analyzing request...", fg="cyan", dim=True))

    analysis = analyst.analyze(user_input)

    # Question loop
    while analysis.action == "ask" and analyst.can_ask_more:
        questions_text = analysis.format_questions_for_user()
        click.echo(click.style("\n" + questions_text, fg="cyan"))
        answer = session.prompt("Your answer> ")
        if not answer.strip():
            break
        analyst.add_answer(questions_text, answer)
        analysis = analyst.analyze(user_input + "\n\nUser clarification: " + answer)

    task_prompt = analysis.build_flow_prompt()

    # "done" pseudo-flow (greetings, simple questions)
    if analysis.flow == "done":
        click.echo(click.style("\n" + (analysis.reason or analysis.original_input), fg="green"))
        return task_prompt, analysis, "done"

    # Show spec and ask for confirmation
    if analysis.action == "proceed":
        spec = analysis.specification
        click.echo(click.style("\n── Analysis Result ──", fg="cyan", bold=True))
        if spec.get("summary"):
            click.echo(click.style(f"Summary: {spec['summary']}", fg="cyan"))
        for key in ("requirements", "hidden_requirements", "assumptions", "out_of_scope"):
            items = spec.get(key, [])
            if items:
                click.echo(click.style(f"\n{key.replace('_', ' ').title()}:", fg="cyan", bold=True))
                for item in items:
                    click.echo(f"  • {item}")
        if spec.get("technical_notes"):
            click.echo(click.style(f"\nTechnical Notes:", fg="cyan", bold=True))
            click.echo(f"  {spec['technical_notes']}")
        click.echo(click.style("─" * 40, fg="cyan"))

        confirm = session.prompt("Proceed with implementation? [Y/n/feedback] ").strip()
        if confirm.lower() in ("n", "no", "cancel"):
            return task_prompt, analysis, "cancelled"
        if confirm and confirm.lower() not in ("y", "yes", ""):
            desc, expected = task_prompt
            desc += f"\n\n## Additional User Feedback\n{confirm}"
            task_prompt = (desc, expected)

    # Apply flow config
    from infinidev.engine.flows import get_flow_config
    flow_config = get_flow_config(analysis.flow)
    desc, _ = task_prompt
    task_prompt = (desc, flow_config.expected_output)

    return task_prompt, analysis, analysis.flow


def run_gather_phase(
    user_input: str,
    agent: Any,
    task_prompt: tuple[str, str],
    analysis: Any,
    session_id: str,
    force_gather: bool = False,
) -> tuple[str, str]:
    """Run the gather phase: collect codebase context before execution.

    Returns the enriched task_prompt.
    """
    from infinidev.config.settings import settings
    from infinidev.db.service import get_recent_summaries

    if not (settings.GATHER_ENABLED or force_gather):
        return task_prompt

    try:
        from infinidev.gather import run_gather
        agent.activate_context(session_id=session_id)
        click.echo(click.style("Gathering context...", fg="cyan", dim=True))
        chat_history = [
            {"role": "user" if "[user]" in s.lower() else "assistant", "content": s}
            for s in get_recent_summaries(session_id, limit=10)
        ]
        brief = run_gather(user_input, chat_history, analysis, agent)
        desc, expected = task_prompt
        desc = brief.render() + "\n\n" + desc
        task_prompt = (desc, expected)
        click.echo(click.style(f"  {brief.summary()}", fg="cyan", dim=True))
    except Exception as exc:
        click.echo(click.style(f"  Gather failed (proceeding without): {exc}", fg="yellow", dim=True))

    return task_prompt


def run_execution_phase(
    agent: Any,
    engine: Any,
    task_prompt: tuple[str, str],
    flow: str,
    analysis: Any,
    session_id: str,
    use_phase_engine: bool = False,
) -> str:
    """Run the execution phase: invoke the appropriate engine.

    Returns the execution result string.
    """
    click.echo(click.style(f"[{flow}] Working on: {task_prompt[0][:120]}", fg="yellow"))

    agent.activate_context(session_id=session_id)
    try:
        if flow in ("explore", "brainstorm"):
            from infinidev.engine.tree_engine import TreeEngine
            tree_engine = TreeEngine()
            result = tree_engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                mode=flow,
            )
        elif use_phase_engine:
            from infinidev.engine.phase_engine import PhaseEngine
            _task_type = "feature"
            if analysis and hasattr(analysis, 'specification'):
                _task_type = analysis.specification.get("task_type", "feature")
            _depth_config = None
            if hasattr(agent, '_gather_brief') and agent._gather_brief:
                try:
                    from infinidev.gather.models import DEPTH_CONFIGS
                    _depth_config = DEPTH_CONFIGS.get(agent._gather_brief.classification.depth)
                except Exception:
                    pass
            phase_eng = PhaseEngine()
            result = phase_eng.execute(
                agent=agent,
                task_prompt=task_prompt,
                task_type=_task_type,
                verbose=True,
                depth_config=_depth_config,
            )
        else:
            result = engine.execute(
                agent=agent,
                task_prompt=task_prompt,
                verbose=True,
            )
        if not result or not result.strip():
            result = "Done. (no additional output)"
    finally:
        agent.deactivate()

    return result


def run_review_phase(
    engine: Any,
    agent: Any,
    session_id: str,
    task_prompt: tuple[str, str],
    result: str,
    reviewer: Any,
    flow: str,
    flow_config: Any,
) -> str:
    """Run the code review phase with review-rework loop.

    Returns the final result (possibly modified by rework).
    """
    from infinidev.config.settings import settings
    from infinidev.db.service import get_recent_summaries

    run_review = flow_config.run_review if flow_config else True
    if not (settings.REVIEW_ENABLED and run_review and flow != "explore" and engine.has_file_changes()):
        return result

    click.echo(click.style("\nRunning code review...", fg="magenta", dim=True))
    from infinidev.engine.review_engine import run_review_rework_loop

    def _cli_review_status(level: str, msg: str) -> None:
        colors = {
            "verification_pass": ("green", True),
            "verification_fail": ("red", False),
            "approved": ("green", True),
            "rejected": ("red", False),
            "max_reviews": ("yellow", True),
        }
        color, dim = colors.get(level, ("white", False))
        click.echo(click.style(msg if level in ("rejected",) else f"{level.replace('_', ' ').title()}: {msg}",
                                fg=color, dim=dim))
        if level == "verification_fail":
            click.echo(click.style("Re-running developer to fix test failures...", fg="magenta", dim=True))
        elif level == "rejected":
            click.echo(click.style("Re-running developer to fix review issues...", fg="magenta", dim=True))

    result, _ = run_review_rework_loop(
        engine=engine,
        agent=agent,
        session_id=session_id,
        task_prompt=task_prompt,
        initial_result=result,
        reviewer=reviewer,
        recent_messages=get_recent_summaries(session_id, limit=5),
        on_status=_cli_review_status,
    )
    return result
