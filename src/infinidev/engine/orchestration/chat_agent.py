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

    # Ephemeral agent_id isolates this turn's tool-context binding from the
    # developer agent's context. set_context writes into a process-global
    # dict keyed by agent_id, so the chat agent, the planner, and the
    # developer each own independent slots that do not stomp each other.
    # clear_agent_context in `finally` ensures no leak across turns.
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
        {"role": "user", "content": _build_user_message(user_input, session_id)},
    ]

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
        return _fallback_respond(_detect_lang(user_input), exc=exc)
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
    return _fallback_respond(_detect_lang(user_input), reason="max_iter")


# ─────────────────────────────────────────────────────────────────────────
# Terminator parsing
# ─────────────────────────────────────────────────────────────────────────


def _build_respond(tc: Any, user_input: str) -> ChatAgentResult:
    args = _parse_args(tc)
    message = (args.get("message") or "").strip()
    if not message:
        return _fallback_respond(_detect_lang(user_input), reason="empty_respond")
    return ChatAgentResult(kind="respond", reply=message)


def _build_escalate(tc: Any, user_input: str) -> ChatAgentResult:
    args = _parse_args(tc)
    understanding = (args.get("understanding") or "").strip()
    if not understanding:
        # Defensive: escalate with empty understanding is a useless
        # handoff. Fall back to respond so the user isn't stranded.
        return _fallback_respond(_detect_lang(user_input), reason="empty_escalate")
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


# Fallback messages keyed by (language, reason). The chat agent's system
# prompt tells the model to match the user's language, but the fallback
# paths bypass the model entirely — so we have to localize ourselves or
# hardcode one language and surprise users in the other. A tiny heuristic
# is good enough here (greetings / action verbs / punctuation cover most
# short chats); the LLM still handles the normal path.
_FALLBACK_MESSAGES: dict[tuple[str, str], str] = {
    ("es", "max_iter"): (
        "Me quedé dando vueltas sin decidir. ¿Podrías reformular lo que "
        "necesitás o decirme explícitamente si querés que lo implemente?"
    ),
    ("en", "max_iter"): (
        "I went in circles without reaching a decision. Could you "
        "rephrase what you need, or say explicitly whether you want me "
        "to implement it?"
    ),
    ("es", "empty_respond"): "No supe cómo contestarte. ¿Podrías reformular?",
    ("en", "empty_respond"): "I don't know how to answer that. Could you rephrase?",
    ("es", "empty_escalate"): (
        "Detecté que querías que haga algo, pero no me quedó clara la "
        "consigna. ¿Podrías decirme qué implementar?"
    ),
    ("en", "empty_escalate"): (
        "I detected that you wanted me to do something, but the request "
        "isn't clear. Could you tell me what to implement?"
    ),
    ("es", "exception"): "Tuve un problema procesando tu mensaje. ¿Podrías repetirlo?",
    ("en", "exception"): "I ran into a problem processing your message. Could you try again?",
}


def _detect_lang(text: str) -> str:
    """Cheap Spanish-vs-English heuristic for fallback messages only.

    Not used for the normal LLM path — the model handles that. Returns
    ``"es"`` when obvious Spanish markers appear, ``"en"`` otherwise.
    Default English because the codebase and prompts lean English.
    """
    if not text:
        return "en"
    lowered = text.lower()
    # Unicode markers that only appear in Spanish (ñ, ¿, ¡, accented vowels)
    if any(ch in lowered for ch in "ñáéíóúü¿¡"):
        return "es"
    # Common Spanish function words
    spanish_markers = (" que ", " por ", " para ", " con ", " los ", " las ",
                       " una ", " dale", " hacé", " decí", " implementá",
                       " arreglá", " agregá", " borrá", "hola", "gracias", "chau")
    padded = f" {lowered} "
    if any(m in padded for m in spanish_markers):
        return "es"
    return "en"


def _fallback_respond(
    lang: str, *, reason: str = "exception", exc: Exception | None = None,
) -> ChatAgentResult:
    """Build a respond result from a localized fallback message.

    The exception (if any) is logged — not surfaced in the reply —
    because tracebacks in chat messages are noise.
    """
    if exc is not None:
        logger.warning("chat_agent fallback (reason=%s): %s", reason, exc)
    message = _FALLBACK_MESSAGES.get(
        (lang, reason),
        _FALLBACK_MESSAGES[("en", reason if (("en", reason) in _FALLBACK_MESSAGES) else "exception")],
    )
    return ChatAgentResult(kind="respond", reply=message)


# ─────────────────────────────────────────────────────────────────────────
# Session history
# ─────────────────────────────────────────────────────────────────────────


def _build_user_message(user_input: str, session_id: Optional[str]) -> str:
    """Combine the session-history snapshot and the current user input
    into a SINGLE ``role="user"`` message.

    Two consecutive ``role="user"`` turns trip some providers (Anthropic
    strictly alternates), so we merge: the snapshot is rendered first,
    then the actual request. The snapshot is optional — missing /
    empty session id / DB failure all degrade gracefully.
    """
    trimmed = user_input.strip()
    turns: list[tuple[str, str]] = []
    if session_id:
        try:
            from infinidev.db.service import get_recent_turns_full
            turns = get_recent_turns_full(
                session_id, limit=6, max_chars_per_turn=2000,
            )
        except Exception as exc:
            logger.warning(
                "chat_agent: session history fetch failed (continuing "
                "without snapshot): %s", exc,
            )
            turns = []
    if not turns:
        return trimmed
    lines = [
        "Recent conversation (for context; use tools to reground facts):",
    ]
    for role, content in turns:
        tag = "USER" if role == "user" else "AGENT"
        lines.append(f'<turn role="{tag}">')
        lines.append(content)
        lines.append("</turn>")
    lines.append("")
    lines.append("Current user message:")
    lines.append(trimmed)
    return "\n".join(lines)


__all__ = ["run_chat_agent"]
