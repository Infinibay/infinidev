"""Execute a team assignment using the existing provider-compatible loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.engines.base import normalize_loop_status
from infinidev.engine.team.runtime import clone_tools
from infinidev.tools.base.context import clear_agent_context, set_context, set_repository_path


@dataclass
class TeamAgent:
    """An agent with a fixed toolset and explicit workspace/session identity."""

    agent_id: str
    name: str
    project_id: int
    workspace_path: str
    tools: list
    backstory: str = "Evidence-grounded research and implementation."
    role: str = "team_worker"


def run_worker(team: Any, member: dict, ticket: dict, assignment: bool) -> tuple[str, str]:
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.prompts.team import build_team_identity
    from infinidev.tools.team import build_team_tools

    worker_id = member["id"]
    tools = clone_tools([team.catalog[name] for name in member["tools"]], worker_id)
    tools += build_team_tools(team, worker_id, orchestrator=False)
    agent = TeamAgent(worker_id, member["name"], team.project_id, team.workspace_path, tools)
    engine = LoopEngine()
    engine._team_runtime = team
    engine._team_actor = worker_id
    engine._tool_allowlist_locked = True
    engine._repository_path = team.repository_path
    team.attach_engine(worker_id, engine)
    set_context(agent_id=worker_id, project_id=team.project_id,
                session_id=f"{team.session_id}:worker:{worker_id}",
                workspace_path=team.workspace_path)
    set_repository_path(worker_id, team.repository_path)
    mode = ("Complete the assigned ticket." if assignment else
            "Answer new teammate requests; the earlier ticket report remains a delivered artifact.")
    prompt = ('<user-request authority="USER_LITERAL">\n' + escape(team.user_request)
              + "\n</user-request>\n\n" + team.turn_context + "\n\n" + mode
              + "\n\nTicket (orchestrator-derived scope):\n" + json.dumps(ticket)
              + "\n\nPrior report:\n" + member.get("last_result", "")
              + "\n\nTeam board:\n" + json.dumps(team.read()))
    try:
        result = engine.execute(
            agent, (prompt, "Report claims, evidence/checks, and limitations to the orchestrator."),
            task_tools=tools,
            identity_override=build_team_identity(
                orchestrator=False, specialist=member["system_prompt"],
                configuration=team.prompt_configuration,
            ),
            prompt_configuration=team.prompt_configuration,
            initial_attachments=team.attachments or None,
            skip_plan=True, allow_plan_mutation=False, allow_explore=False,
            max_iterations=settings.TEAM_WORKER_MAX_ITERATIONS,
            max_total_tool_calls=settings.TEAM_WORKER_MAX_TOOL_CALLS,
            max_tool_calls_per_action=0,
        )
        return result, normalize_loop_status(engine._last_status)
    finally:
        clear_agent_context(worker_id)
