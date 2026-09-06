"""The user-facing agent owns the objective and delegates scoped execution."""

from __future__ import annotations

import json
from html import escape
from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.engines.base import EngineResult, loop_observed_metrics, normalize_loop_status
from infinidev.engine.team.runtime import ROOT, TeamRuntime, clone_tools
from infinidev.engine.team.worker import TeamAgent
from infinidev.prompts.team import build_team_identity
from infinidev.tools.base.context import set_context, set_repository_path
from infinidev.tools.chat import SendMessageTool
from infinidev.tools.team import build_team_tools

_ROOT_READS = {"read_file", "code_search", "list_directory", "glob", "git_diff",
               "git_status", "preview_changes", "web_search", "web_fetch", "read_report",
               "read_command_output", "history_read", "history_search"}


class OrchestratorAdapter:
    """Run the existing loop with a persistent team and review-owned completion."""

    name = "orchestrator"

    def run(self, **kwargs: Any) -> EngineResult:
        agent, engine, hooks = kwargs["agent"], kwargs["engine"], kwargs["hooks"]
        escalation = kwargs["escalation"]
        session_id = kwargs["session_id"]
        workspace = kwargs["workspace_path"]
        team = TeamRuntime(
            session_id=session_id, project_id=kwargs["project_id"], workspace_path=workspace,
            root_agent_id=agent.agent_id, catalog=list(agent.tools),
            max_workers=settings.TEAM_MAX_WORKERS, max_agents=settings.TEAM_MAX_AGENTS,
            max_followups=settings.TEAM_MAX_FOLLOWUPS,
            repository_path=getattr(engine, "_repository_path", None),
            prompt_configuration=kwargs.get("prompt_configuration"), on_status=hooks.on_status,
            user_request=escalation.user_request, turn_context=kwargs.get("turn_context", ""),
            catalog_supplier=lambda: list(agent.tools),
            attachments=escalation.attachments,
        )
        try:
            reads = [t for name, t in team.catalog.items() if name in _ROOT_READS or (
                "ken" in name and name.endswith(("find", "read", "recall", "related", "remember"))
            )]
            tools = clone_tools(reads + [SendMessageTool()], agent.agent_id)
            tools += build_team_tools(team, agent.agent_id, orchestrator=True)
            scoped = TeamAgent(agent.agent_id, getattr(agent, "name", "Infinidev"),
                               kwargs["project_id"], workspace, tools, role="orchestrator")
            scoped._session_summaries = getattr(agent, "_session_summaries", None)
            engine._team_runtime = team
            engine._team_actor = ROOT
            engine._tool_allowlist_locked = True
            set_context(agent_id=agent.agent_id, project_id=kwargs["project_id"],
                        session_id=session_id, workspace_path=workspace)
            set_repository_path(agent.agent_id, team.repository_path)
            hooks.on_status("info", "Orchestrator: scoped workers, shared notes and peer messages")
            prompt = ('<user-request authority="USER_LITERAL">\n'
                      + escape(escalation.user_request) + "\n</user-request>\n\n"
                      + kwargs.get("turn_context", "")
                      + "\n\nPersisted team board:\n" + json.dumps(team.read()))
            task_prompt = (prompt, "Answer the user with reviewed evidence and limits.")
            if settings.GATHER_ENABLED or kwargs.get("force_gather", False):
                from infinidev.engine.orchestration.pipeline import _run_gather_phase

                task_prompt = _run_gather_phase(
                    user_input=escalation.user_request, agent=agent, task_prompt=task_prompt,
                    session_id=session_id, force_gather=kwargs.get("force_gather", False),
                    hooks=hooks, prompt_configuration=kwargs.get("prompt_configuration"),
                )
            hooks.on_phase("execute")
            result = engine.execute(
                scoped, task_prompt,
                task_tools=tools, identity_override=build_team_identity(
                    orchestrator=True, configuration=kwargs.get("prompt_configuration")),
                prompt_configuration=kwargs.get("prompt_configuration"),
                skip_plan=True, allow_plan_mutation=False, allow_explore=False,
                initial_attachments=escalation.attachments or None,
                max_iterations=settings.TASK_MAX_ITERATIONS,
                max_total_tool_calls=settings.TASK_MAX_TOOL_CALLS,
                max_tool_calls_per_action=0,
            )
            status = normalize_loop_status(engine._last_status)
            if engine.is_cancelled:
                status = "cancelled"
            # Also enforce the contract for plain-text/alternative loop endings.
            if status == "completed" and (blocker := team.completion_blocker()):
                status, result = "blocked", f"{result}\n\nTeam work remains: {blocker}"
                engine._last_status = "blocked"
        finally:
            try:
                team.close()
            finally:
                engine._team_runtime = None
                engine._team_actor = None
                engine._tool_allowlist_locked = False
        metrics = loop_observed_metrics(engine)
        for key, value in team.observed_worker_metrics().items():
            metrics[key] = metrics.get(key, 0) + value
        return EngineResult(
            engine_name=self.name, status=status, user_message=result, engine=engine,
            summary=result[:1000], state=team.read(), resume_token=team.store.team_id,
            metrics=metrics,
        )
