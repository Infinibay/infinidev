"""Chat agent orchestrator — the default entry point of every user turn.

Replaces the legacy conversational fastpath (deleted in Commit 7).
Runs a short, read-only LLM loop that terminates when the model calls
either ``respond`` (conversational reply, turn ends) or ``escalate``
(hand off to the planner).

This is a **self-contained loop**, not a LoopEngine invocation:

  * The LoopEngine is built for plan-execute-summarize with steps,
    notes, guards, behavior tracking, and summarizer — all of which
    are overkill for a 5-iteration conversational turn.
  * The preamble (which we're replacing) used a similar purpose-built
    shape: single litellm call + direct tool_calls parsing. We keep
    that simplicity but add multi-iteration tool dispatch.

Every turn is fresh: we read ``get_recent_turns_full`` from the DB on
entry and accumulate no state across calls. The DB is the single
source of truth for conversation history.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from infinidev.config.llm import get_litellm_params_for_behavior
from infinidev.engine.loop.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.loop.tools import build_tool_dispatch, execute_tool_call
from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.prompts.chat_agent import CHAT_AGENT_SYSTEM_PROMPT
from infinidev.tools import get_tools_for_role
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)

logger = logging.getLogger(__name__)


_DEFAULT_MAX_ITERATIONS = 5
_MAX_RESULT_CHARS = 8000  # trim overly long tool outputs before re-prompting


def run_chat_agent(
    user_input: str,
    *,
    session_id: Optional[str] = None,
    project_id: Optional[int] = None,
    workspace_path: Optional[str] = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> ChatAgentResult:
    """Run one turn of the chat agent and return its result.

    The caller (pipeline.run_task) surfaces the result to the user:
    ``kind="respond"`` → show ``reply`` and end the turn;
    ``kind="escalate"`` → continue to the analyst planner with
    ``escalation`` as the handoff packet.

    When the loop can't produce a decision (LLM error, no tool call,
    max-iter exhaustion), a fallback ``respond`` is returned rather
    than raising — the UI must always get something to show the user.
    """
    if not user_input or not user_input.strip():
        return ChatAgentResult(
            kind="respond",
            reply="(empty message)",
        )

    agent_id = f"chat-agent-{uuid.uuid4().hex[:8]}"
    tools = get_tools_for_role("chat_agent")
    bind_tools_to_agent(tools, agent_id)
    set_context(
        agent_id=agent_id,
        project_id=project_id,
        session_id=session_id,
        workspace_path=workspace_path,
    )

    dispatch = build_tool_dispatch(tools)
    tool_schemas = [tool_to_openai_schema(t) for t in tools]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CHAT_AGENT_SYSTEM_PROMPT},
    ]
    _append_recent_turns(messages, session_id)
    messages.append({"role": "user", "content": user_input.strip()})

    try:
        return _run_llm_loop(
            messages=messages,
            tool_schemas=tool_schemas,
            dispatch=dispatch,
            user_input=user_input,
            max_iterations=max_iterations,
        )
    except Exception as exc:
        logger.exception("Chat agent loop failed")
        return _fallback_respond(
            f"Tuve un problema procesando tu mensaje ({exc}). ¿Podrías repetirlo?"
        )
    finally:
        clear_agent_context(agent_id)


# ─────────────────────────────────────────────────────────────────────────
# Loop driver
# ─────────────────────────────────────────────────────────────────────────


def _run_llm_loop(
    *,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    dispatch: dict[str, Any],
    user_input: str,
    max_iterations: int,
) -> ChatAgentResult:
    import litellm

    base_kwargs = get_litellm_params_for_behavior()

    for iteration in range(max_iterations):
        call_kwargs = dict(base_kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["tools"] = tool_schemas
        call_kwargs.setdefault("temperature", 0.1)
        call_kwargs.setdefault("stream", False)
        call_kwargs.setdefault("max_tokens", 2000)

        response = litellm.completion(**call_kwargs)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            # The model chatted in plain text instead of calling a tool.
            # Treat it as a respond — we still terminate cleanly and the
            # user sees the text.
            content = getattr(message, "content", None) or ""
            return ChatAgentResult(
                kind="respond",
                reply=content.strip() or "(no reply)",
            )

        # Add the assistant turn to the transcript so tool results can
        # reference the tool_use IDs.
        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
        })

        # Check whether ANY call is a terminator — if yes, we're done.
        for tc in tool_calls:
            name = tc.function.name
            if name == "respond":
                return _build_respond(tc, user_input)
            if name == "escalate":
                return _build_escalate(tc, user_input)

        # No terminator — execute read-only tools and continue.
        for tc in tool_calls:
            result = execute_tool_call(
                dispatch, tc.function.name, tc.function.arguments,
            )
            trimmed = result if len(result) <= _MAX_RESULT_CHARS else (
                result[:_MAX_RESULT_CHARS] + "\n...[truncated]"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": trimmed,
            })

    # Max iterations reached without terminator. Return a graceful
    # respond so the user sees SOMETHING — never a silent return.
    logger.warning("Chat agent hit max_iterations=%d without terminator", max_iterations)
    return _fallback_respond(
        "Me quedé dando vueltas sin decidir. ¿Podrías reformular lo que "
        "necesitás o decirme explícitamente si querés que lo implemente?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Terminator parsing
# ─────────────────────────────────────────────────────────────────────────


def _build_respond(tc: Any, user_input: str) -> ChatAgentResult:
    args = _parse_args(tc)
    message = (args.get("message") or "").strip()
    if not message:
        return _fallback_respond(
            "No supe cómo contestarte. ¿Podrías reformular?"
        )
    return ChatAgentResult(kind="respond", reply=message)


def _build_escalate(tc: Any, user_input: str) -> ChatAgentResult:
    args = _parse_args(tc)
    understanding = (args.get("understanding") or "").strip()
    if not understanding:
        # Defensive: escalate with empty understanding is a useless
        # handoff. Fall back to respond so the user isn't stranded.
        return _fallback_respond(
            "Detecté que querías que haga algo, pero no me quedó clara la "
            "consigna. ¿Podrías decirme qué implementar?"
        )
    opened = args.get("opened_files") or []
    if not isinstance(opened, list):
        opened = []
    packet = EscalationPacket(
        user_request=user_input.strip(),
        understanding=understanding,
        opened_files=[str(p) for p in opened],
        user_visible_preview=(args.get("user_visible_preview") or "").strip(),
        user_signal=(args.get("user_signal") or "").strip(),
        suggested_flow="develop",  # v1 restriction
    )
    return ChatAgentResult(kind="escalate", escalation=packet)


def _parse_args(tc: Any) -> dict[str, Any]:
    raw = getattr(tc.function, "arguments", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Serialize a tool_call object back to the dict shape LiteLLM
    accepts on the next call. Provider-specific message objects don't
    serialize automatically."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": (
                tc.function.arguments
                if isinstance(tc.function.arguments, str)
                else json.dumps(tc.function.arguments)
            ),
        },
    }


def _fallback_respond(msg: str) -> ChatAgentResult:
    return ChatAgentResult(kind="respond", reply=msg)


# ─────────────────────────────────────────────────────────────────────────
# Session history
# ─────────────────────────────────────────────────────────────────────────


def _append_recent_turns(
    messages: list[dict[str, Any]], session_id: Optional[str],
) -> None:
    """Append prior USER/ASSISTANT turns as a context snapshot.

    Uses the same DB helper the legacy preamble used, so self-referential
    follow-ups see the same signal (plus the chat agent can now go read
    the real files referenced in prior turns via its tool access).
    """
    if not session_id:
        return
    try:
        from infinidev.db.service import get_recent_turns_full
        turns = get_recent_turns_full(session_id, limit=6, max_chars_per_turn=2000)
    except Exception:
        return
    if not turns:
        return
    snapshot_lines = [
        "Recent conversation (for context; use tools to reground facts):",
    ]
    for role, content in turns:
        tag = "USER" if role == "user" else "AGENT"
        snapshot_lines.append(f'<turn role="{tag}">')
        snapshot_lines.append(content)
        snapshot_lines.append("</turn>")
    messages.append({"role": "user", "content": "\n".join(snapshot_lines)})


__all__ = ["run_chat_agent"]
