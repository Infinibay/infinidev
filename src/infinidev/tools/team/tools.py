"""Validated collaboration tools; workers receive communication, not delegation."""

from __future__ import annotations

import json
import sqlite3
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, StringConstraints

from infinidev.engine.team.waiting import IdleInput
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.context import bind_tools_to_agent

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class CreateTicketInput(BaseModel):
    title: Name
    objective: Text
    acceptance: list[Text] = Field(min_length=1, max_length=20)
    constraints: list[Text] = Field(default_factory=list, max_length=20)
    dependencies: list[Name] = Field(default_factory=list, max_length=20)


class DelegateInput(BaseModel):
    ticket_id: Name
    name: Name = Field(description="Unique human-readable given name, e.g. Lucía or Mateo")
    role: Name = Field(description="Visible responsibility, e.g. Researcher, Developer, Reviewer")
    system_prompt: Text = Field(description="Specialist guidance within inherited user/project scope")
    tools: list[Name] = Field(max_length=80, description="Exact names from team_tool_catalog")
    worker_id: Name | None = Field(default=None, description="Reuse an idle worker from the roster")


class ReviewTicketInput(BaseModel):
    ticket_id: Name
    decision: Literal["accepted", "needs_work", "cancelled"]
    reason: Text = Field(description="Evidence against acceptance criteria, or reason for cancellation")


class MessageInput(BaseModel):
    recipient: Name = Field(description="Worker ID/name, orchestrator, or all")
    content: Text
    reply_to: int | None = Field(default=None, ge=1)
    ticket_id: Name | None = None
    message_type: Literal["request", "reply", "info"] | None = Field(
        default=None, description="request asks for action; reply answers reply_to; info needs no answer. "
        "Default: reply when reply_to is set, otherwise request.",
    )


class NoteInput(BaseModel):
    content: Text
    kind: Literal["hypothesis", "observation", "decision", "handoff"]
    refs: list[Name] = Field(default_factory=list, max_length=30)
    ticket_id: Name | None = None
    supersedes: int | None = Field(default=None, ge=1)


class ReadTeamInput(BaseModel):
    view: Literal["board", "notes", "messages", "events"] = "board"
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)
    ticket_id: Name | None = None
    thread_id: int | None = Field(default=None, ge=1, description="Read one conversation thread")


class WaitTeamInput(BaseModel):
    seconds: float | None = Field(default=None, ge=0, le=604800)


class CatalogInput(BaseModel):
    query: str = Field(default="", max_length=160)


class TeamTool(InfinibayBaseTool):
    """Derive identity from the binding; models cannot supply an author field."""

    _team: Any = PrivateAttr(default=None)
    _operation: str = PrivateAttr(default="")

    def _run(self, **kwargs: Any) -> str:
        if self._team is None:
            return self._error("No active research team")
        try:
            actor = self._team.actor(self.agent_id)
            operation = self._operation
            if operation == "catalog":
                self._team._require_root(actor)
                self._team.refresh_catalog()
                query = kwargs.get("query", "").casefold()
                result = [{"name": t.name, "description": t.description[:180],
                           "read_only": t.is_read_only}
                          for t in self._team.catalog.values()
                          if query in (t.name + " " + t.description).casefold()]
            elif operation in {"read", "wait"}:
                if operation == "wait":
                    result = self._team.idle(actor, timeout=kwargs.get("seconds"))
                else:
                    result = self._team.read(**kwargs)
            else:
                result = getattr(self._team, operation)(actor, **kwargs)
            return json.dumps(result, ensure_ascii=False)
        except (ValueError, KeyError, sqlite3.Error) as exc:
            return self._error(str(exc))


class CreateTicketTool(TeamTool):
    name: str = "team_create_ticket"
    description: str = "Create a bounded ticket with deliverable, checks and existing dependencies."
    args_schema: type[BaseModel] = CreateTicketInput


class DelegateTool(TeamTool):
    name: str = "team_delegate"
    description: str = (
        "Start a ticket worker with a chosen name, visible role, specialist prompt and exact tools."
    )
    args_schema: type[BaseModel] = DelegateInput


class ReviewTicketTool(TeamTool):
    name: str = "team_review_ticket"
    description: str = "Accept a delivered report, request rework, or cancel obsolete work with evidence."
    args_schema: type[BaseModel] = ReviewTicketInput


class TeamMessageTool(TeamTool):
    name: str = "team_send_message"
    description: str = (
        "Ask a teammate a question, reply using reply_to, or share information. "
        "Requests and their answers can wake recipients; info does not start a completed worker. "
        "Returns a durable ID/thread and delivery status. Use team_idle(reply_to=ID) for an answer."
    )
    args_schema: type[BaseModel] = MessageInput


class TeamNoteTool(TeamTool):
    name: str = "team_write_note"
    description: str = (
        "Publish a shared note with automatic author/date and evidence refs. "
        "To correct a note set supersedes to its ID; history is preserved."
    )
    args_schema: type[BaseModel] = NoteInput


class ReadTeamTool(TeamTool):
    name: str = "team_read"
    description: str = (
        "Read roster/ticket board, notes, messages or events. History pages use after/next_after; "
        "superseded_by identifies outdated notes. All members can read team conversations."
    )
    args_schema: type[BaseModel] = ReadTeamInput


class WaitTeamTool(TeamTool):
    name: str = "team_wait"
    description: str = (
        "Idle until a message, report or user guidance arrives, without model polling. "
        "Optional seconds sets a timeout. Use team_idle for event filters."
    )
    args_schema: type[BaseModel] = WaitTeamInput


class IdleTeamTool(TeamTool):
    name: str = "team_idle"
    description: str = (
        "Suspend this agent until a selected event arrives, retaining context without model calls. "
        "Choose message/report/background_task/note/ticket, with optional sender, reply_to, task_ids "
        "and ticket_id filters. Releases worker capacity and workspace lease. User guidance and "
        "cancellation always interrupt. Omit timeout to sleep until an event."
    )
    args_schema: type[BaseModel] = IdleInput


class TeamCatalogTool(TeamTool):
    name: str = "team_tool_catalog"
    description: str = "Find available worker tool names/descriptions by text, without loading schemas."
    args_schema: type[BaseModel] = CatalogInput


def build_team_tools(team: Any, agent_id: str, *, orchestrator: bool) -> list:
    """Construct fresh, scoped tool instances for one team member."""
    entries = [(TeamMessageTool, "send"), (TeamNoteTool, "write_note"), (ReadTeamTool, "read"),
               (IdleTeamTool, "idle"), (WaitTeamTool, "wait")]
    if orchestrator:
        entries += [(CreateTicketTool, "create_ticket"), (DelegateTool, "delegate"),
                    (ReviewTicketTool, "review"),
                    (TeamCatalogTool, "catalog")]
    tools = []
    for cls, operation in entries:
        tool = cls()
        tool._team = team
        tool._operation = operation
        tools.append(tool)
    bind_tools_to_agent(tools, agent_id)
    return tools
