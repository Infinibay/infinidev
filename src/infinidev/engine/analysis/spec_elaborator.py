"""Spec-elaboration loop — vague requirement → GroundedSpec.

Runs once per task between the chat agent's escalation and the planner, on
the SINGLE configured model (no tiering). It externalises the loop a SOTA
model runs internally — analyse, find gaps, resolve them against evidence,
critique/discard, converge — as staged passes the harness owns:

    Pass A (analyze)   one LLM call  → restatement + tagged gaps
    Pass B (ground)    bounded read-only exploration loop → resolved facts,
                       assumptions, product-intent clarifications
    Pass C (critique)  one LLM call generates N candidate directions; a
                       DETERMINISTIC check (file/symbol existence) discards
                       the hallucinated ones — the discard is code, not the
                       model's self-judgement.

Every pass degrades gracefully: any failure yields a partial (or None)
GroundedSpec and the pipeline proceeds exactly as before. See
``docs_spec_elaboration_loop.md``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

from infinidev.config.llm import get_litellm_params
from infinidev.config.settings import settings
from infinidev.engine.analysis.grounded_spec import (
    Assumption,
    GroundedSpec,
    RejectedAlternative,
    ResolvedFact,
)
from infinidev.engine.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.tool_dispatch import build_tool_dispatch, execute_tool_call
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.tools import get_tools_for_role
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 6000


# ── Public entry point ────────────────────────────────────────────────────


def should_elaborate(escalation: EscalationPacket) -> bool:
    """Complexity gate — skip elaboration for trivial requests.

    Conservative by design: when unsure, elaborate. A trivial request is a
    short one (a typo fix, a one-line change). Everything else pays the
    once-per-task cost. Tunable via ``SPEC_ELABORATION_*`` settings.
    """
    if not settings.SPEC_ELABORATION_ENABLED:
        return False
    request = (escalation.user_request or "").strip()
    if len(request) < settings.SPEC_ELABORATION_MIN_CHARS:
        return False
    return True


def elaborate(
    escalation: EscalationPacket,
    *,
    session_id: Optional[str] = None,
    project_id: Optional[int] = None,
    workspace_path: Optional[str] = None,
) -> Optional[GroundedSpec]:
    """Produce a GroundedSpec from the escalation, or None on skip/failure.

    Never raises — the pipeline treats None as "no elaboration" and proceeds
    with the raw escalation, identical to today's behaviour.
    """
    if not should_elaborate(escalation):
        return None

    agent_id = f"elaborator-{uuid.uuid4().hex[:8]}"
    workspace_path = workspace_path or os.getcwd()
    try:
        # Read-only exploration tools (no terminator) for the grounding pass.
        tools = get_tools_for_role("assistant_critic")
        bind_tools_to_agent(tools, agent_id)
        set_context(
            agent_id=agent_id,
            project_id=project_id,
            session_id=session_id,
            workspace_path=workspace_path,
        )
        dispatch = build_tool_dispatch(tools)
        read_schemas = [tool_to_openai_schema(t) for t in tools]

        request = escalation.user_request.strip()
        understanding = (escalation.understanding or "").strip()
        opened = escalation.opened_files or []

        analysis = _pass_analyze(request, understanding, opened)
        grounding = _pass_ground(
            request, understanding, analysis, read_schemas, dispatch
        )
        critique = _pass_critique(request, analysis, grounding)
        winner, rejected, risks = _deterministic_discard(
            critique.get("candidates", []), workspace_path, project_id
        )

        return _assemble(
            request, understanding, analysis, grounding, winner, rejected, risks
        )
    except Exception:
        logger.exception("Spec elaboration failed; proceeding without a GroundedSpec")
        return None
    finally:
        clear_agent_context(agent_id)


# ── Pass A: analyze (restatement + tagged gaps) ───────────────────────────

_ANALYZE_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_analysis",
        "description": "Emit the restatement and the enumerated gaps.",
        "parameters": {
            "type": "object",
            "properties": {
                "deliverable": {"type": "string", "description": "One sentence: the concrete thing to deliver."},
                "in_scope": {"type": "array", "items": {"type": "string"}},
                "out_of_scope": {"type": "array", "items": {"type": "string"}, "description": "What is explicitly NOT being asked."},
                "gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "kind": {"type": "string", "enum": ["technical", "theory", "product_intent"]},
                            "why_it_matters": {"type": "string"},
                        },
                        "required": ["question", "kind"],
                    },
                },
            },
            "required": ["deliverable", "gaps"],
        },
    },
}


def _pass_analyze(request: str, understanding: str, opened: list[str]) -> dict:
    sys = (
        "You are elaborating a software task before it is planned. Do NOT design "
        "or write code. Analyse the request precisely and enumerate the GAPS that "
        "must be resolved before it is implementable. Tag each gap:\n"
        "- technical: resolvable by reading the codebase (does X exist? sync or async?).\n"
        "- theory: needs external knowledge (which algorithm? standard approach?).\n"
        "- product_intent: a PRODUCT decision only the user can make (which behaviour? "
        "what value?). NEVER invent answers to these.\n"
        "State what is OUT of scope — what the request does NOT ask for."
    )
    opened_str = ("\nFiles already read upstream:\n  " + "\n  ".join(opened)) if opened else ""
    user = (
        f"user_request (verbatim):\n  {request}\n\n"
        f"chat agent understanding:\n  {understanding}{opened_str}\n\n"
        "Call emit_analysis."
    )
    args = _structured_call(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        terminator=_ANALYZE_TOOL,
        terminator_name="emit_analysis",
    )
    return args or {"deliverable": request, "in_scope": [], "out_of_scope": [], "gaps": []}


# ── Pass B: ground (resolve gaps against evidence) ────────────────────────

_GROUNDING_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_grounding",
        "description": "Emit resolved facts, assumptions, and product clarifications.",
        "parameters": {
            "type": "object",
            "properties": {
                "resolved_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "evidence": {"type": "string", "description": "file:line read, or a URL/citation. Empty if none."},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        "required": ["question", "answer"],
                    },
                },
                "assumptions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string"},
                            "why_no_evidence": {"type": "string"},
                        },
                        "required": ["statement"],
                    },
                },
                "clarifications_needed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product-intent questions for the USER. Never answer these.",
                },
            },
            "required": ["resolved_facts"],
        },
    },
}


def _pass_ground(
    request: str,
    understanding: str,
    analysis: dict,
    read_schemas: list[dict],
    dispatch: dict,
) -> dict:
    """Bounded read-only exploration: resolve technical/theory gaps against evidence."""
    gaps = analysis.get("gaps", [])
    if not gaps:
        return {"resolved_facts": [], "assumptions": [], "clarifications_needed": []}

    sys = (
        "Resolve the gaps for this task using EVIDENCE, not imagination.\n"
        "- technical gaps: READ the codebase (code_search, read_file, list_symbols) "
        "and answer with the evidence (file:line). If you cannot find evidence, record "
        "it as an ASSUMPTION, never invent a fact.\n"
        "- theory gaps: answer from knowledge or a web search; cite the source.\n"
        "- product_intent gaps: do NOT answer — put them in clarifications_needed.\n"
        f"You may make at most {settings.SPEC_ELABORATION_MAX_EVIDENCE_CALLS} read "
        "calls, then call emit_grounding."
    )
    gaps_str = "\n".join(
        f"  - [{g.get('kind', '?')}] {g.get('question', '')}" for g in gaps
    )
    user = (
        f"Task: {analysis.get('deliverable', request)}\n\nGaps to resolve:\n{gaps_str}\n\n"
        "Explore if needed, then call emit_grounding."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
    args = _exploration_call(
        messages,
        read_schemas=read_schemas,
        dispatch=dispatch,
        terminator=_GROUNDING_TOOL,
        terminator_name="emit_grounding",
        max_exploration_calls=settings.SPEC_ELABORATION_MAX_EVIDENCE_CALLS,
    )
    if not args:
        # Degrade: every gap becomes an explicit clarification/assumption.
        prod = [g["question"] for g in gaps if g.get("kind") == "product_intent"]
        assum = [
            {"statement": f"Unresolved: {g['question']}", "why_no_evidence": "grounding pass failed"}
            for g in gaps if g.get("kind") != "product_intent"
        ]
        return {"resolved_facts": [], "assumptions": assum, "clarifications_needed": prod}
    return args


# ── Pass C: critique (generate candidates → deterministic discard) ────────

_CANDIDATES_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_candidates",
        "description": "Emit candidate design directions to be checked.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "One sentence: the design direction."},
                            "referenced_files": {"type": "array", "items": {"type": "string"}, "description": "Existing files this would modify (relative paths)."},
                            "referenced_symbols": {"type": "array", "items": {"type": "string"}, "description": "Existing functions/classes this relies on."},
                        },
                        "required": ["summary"],
                    },
                },
            },
            "required": ["candidates"],
        },
    },
}


def _pass_critique(request: str, analysis: dict, grounding: dict) -> dict:
    """Generate N candidate directions. The DISCARD is done deterministically after."""
    n = max(2, settings.SPEC_ELABORATION_CANDIDATES)
    facts = grounding.get("resolved_facts", [])
    facts_str = "\n".join(
        f"  - {f.get('question', '')} → {f.get('answer', '')}" for f in facts
    ) or "  (none)"
    sys = (
        f"Propose {n} DISTINCT candidate design directions for this task. Each must "
        "name the EXISTING files it would modify and the EXISTING symbols it relies "
        "on (these will be checked against the real codebase — do not invent paths). "
        "Make them genuinely different approaches, not rewordings. Call emit_candidates."
    )
    user = (
        f"Task: {analysis.get('deliverable', request)}\n"
        f"In scope: {', '.join(analysis.get('in_scope', [])) or '(unspecified)'}\n"
        f"Grounded facts:\n{facts_str}\n\nCall emit_candidates with {n} candidates."
    )
    args = _structured_call(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        terminator=_CANDIDATES_TOOL,
        terminator_name="emit_candidates",
    )
    return args or {"candidates": []}


def _deterministic_discard(
    candidates: list[dict],
    workspace_path: str,
    project_id: Optional[int],
) -> tuple[dict | None, list[RejectedAlternative], list[str]]:
    """Discard candidates by EXECUTABLE checks, not the model's self-judgement.

    A candidate referencing files/symbols that do not exist in the real
    codebase is a hallucination — the verifier (filesystem + code-index), not
    a second LLM, kills it. Returns (winner, rejected[], risks[]).
    """
    if not candidates:
        return None, [], []

    scored: list[tuple[int, dict, list[str]]] = []
    for cand in candidates:
        problems: list[str] = []
        for rel in cand.get("referenced_files", []) or []:
            if _looks_like_existing_path(rel) and not _file_exists(workspace_path, rel):
                problems.append(f"file not found: {rel}")
        for sym in cand.get("referenced_symbols", []) or []:
            if project_id is not None and not _symbol_exists(project_id, sym):
                problems.append(f"symbol not found: {sym}")
        scored.append((len(problems), cand, problems))

    # Lowest problem count wins; ties keep input order (stable sort).
    scored.sort(key=lambda t: t[0])
    _, winner, winner_problems = scored[0]

    rejected: list[RejectedAlternative] = []
    for n_problems, cand, problems in scored[1:]:
        why = "; ".join(problems) if problems else "viable alternative, not chosen"
        rejected.append(
            RejectedAlternative(alternative=cand.get("summary", "?"), why_rejected=why)
        )
    # The winner's own unresolved references become risks (honest residue).
    risks = list(winner_problems)
    return winner, rejected, risks


# ── Assembly ──────────────────────────────────────────────────────────────


def _assemble(
    request: str,
    understanding: str,
    analysis: dict,
    grounding: dict,
    winner: dict | None,
    rejected: list[RejectedAlternative],
    risks: list[str],
) -> GroundedSpec:
    facts = [
        ResolvedFact(
            question=f.get("question", ""),
            answer=f.get("answer", ""),
            evidence=f.get("evidence", "") or "",
            confidence=f.get("confidence", "medium") or "medium",
        )
        for f in grounding.get("resolved_facts", [])
        if f.get("question")
    ]
    assumptions = [
        Assumption(
            statement=a.get("statement", ""),
            why_no_evidence=a.get("why_no_evidence", "") or "",
        )
        for a in grounding.get("assumptions", [])
        if a.get("statement")
    ]
    clarifications = [c for c in grounding.get("clarifications_needed", []) if c]
    in_scope = analysis.get("in_scope", []) or []
    out_of_scope = analysis.get("out_of_scope", []) or []
    deliverable = analysis.get("deliverable", "") or request
    design_direction = (winner or {}).get("summary", "")

    signature = " | ".join(
        [deliverable] + in_scope + [f.question for f in facts]
    )[:500]

    # Surface theory gaps that are NOT already answered by a resolved fact.
    # (Previously this used `and not facts`, which dropped ALL theory
    # open-questions the moment a single — possibly unrelated — fact was
    # resolved. Over-surfacing is safe; silently dropping unresolved
    # questions violates the surface-don't-invent contract.)
    resolved_questions = {f.question for f in facts}
    open_questions = [
        g.get("question", "")
        for g in analysis.get("gaps", [])
        if g.get("kind") == "theory" and g.get("question", "") not in resolved_questions
    ]

    return GroundedSpec(
        deliverable=deliverable,
        in_scope=in_scope,
        out_of_scope=out_of_scope,
        resolved_facts=facts,
        assumptions=assumptions,
        clarifications_needed=clarifications,
        design_direction=design_direction,
        alternatives_rejected=rejected,
        risks=risks,
        open_questions=open_questions,
        signature_text=signature,
    )


# ── LLM call helpers ──────────────────────────────────────────────────────


def _structured_call(
    messages: list[dict[str, Any]],
    *,
    terminator: dict,
    terminator_name: str,
) -> dict | None:
    """One forced terminator call; returns parsed args or None."""
    import litellm

    call_kwargs = dict(get_litellm_params())
    call_kwargs["messages"] = messages
    call_kwargs["tools"] = [terminator]
    call_kwargs.setdefault("temperature", 0.2)
    call_kwargs.setdefault("max_tokens", 2000)
    call_kwargs["stream"] = False
    # Force the terminator when the provider supports it; harmless otherwise
    # (we still parse defensively below).
    try:
        call_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": terminator_name},
        }
        response = litellm.completion(**call_kwargs)
    except Exception:
        call_kwargs.pop("tool_choice", None)
        response = litellm.completion(**call_kwargs)

    return _extract_tool_args(response, terminator_name)


def _exploration_call(
    messages: list[dict[str, Any]],
    *,
    read_schemas: list[dict],
    dispatch: dict,
    terminator: dict,
    terminator_name: str,
    max_exploration_calls: int,
) -> dict | None:
    """Planner-style bounded loop: read tools + terminator. Returns parsed args."""
    import litellm

    base_kwargs = get_litellm_params()
    tools = list(read_schemas) + [terminator]
    exploration_calls = 0

    for _ in range(max_exploration_calls + 2):
        call_kwargs = dict(base_kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["tools"] = tools
        call_kwargs.setdefault("temperature", 0.2)
        call_kwargs.setdefault("max_tokens", 2000)
        call_kwargs["stream"] = False

        response = litellm.completion(**call_kwargs)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return _parse_json_content(getattr(message, "content", None))

        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
        })

        for tc in tool_calls:
            if tc.function.name == terminator_name:
                return _parse_args(tc.function.arguments)

        for tc in tool_calls:
            exploration_calls += 1
            result = execute_tool_call(dispatch, tc.function.name, tc.function.arguments)
            if len(result) > _MAX_RESULT_CHARS:
                result = result[:_MAX_RESULT_CHARS] + "\n...[truncated]"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if exploration_calls >= max_exploration_calls:
            messages.append({
                "role": "user",
                "content": f"Budget used. Call {terminator_name} now with what you have.",
            })

    return None


def _extract_tool_args(response: Any, name: str) -> dict | None:
    try:
        message = response.choices[0].message
        for tc in getattr(message, "tool_calls", None) or []:
            if tc.function.name == name:
                return _parse_args(tc.function.arguments)
        return _parse_json_content(getattr(message, "content", None))
    except Exception:
        return None


def _parse_args(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _parse_json_content(content: Any) -> dict | None:
    """Last-resort: pull a JSON object out of a text response."""
    if not isinstance(content, str) or "{" not in content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if end <= start:
        return None
    try:
        parsed = json.loads(content[start:end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


# ── Deterministic check helpers ───────────────────────────────────────────


def _looks_like_existing_path(rel: str) -> bool:
    """Heuristic: a relative path with a file extension that a candidate claims
    to MODIFY (so it should already exist). Avoids false-killing new files —
    a path with no extension or an obvious 'new' marker is not checked."""
    if not isinstance(rel, str) or not rel.strip():
        return False
    rel = rel.strip()
    base = os.path.basename(rel)
    return "." in base and not rel.endswith("/")


def _file_exists(workspace_path: str, rel: str) -> bool:
    candidate = rel if os.path.isabs(rel) else os.path.join(workspace_path, rel)
    return os.path.exists(candidate)


def _symbol_exists(project_id: int, qualified_name: str) -> bool:
    """Best-effort code-index lookup. Returns True (don't reject) if the index
    is unavailable or the name is too vague to check."""
    name = (qualified_name or "").strip()
    if not name or len(name) < 3:
        return True
    try:
        from infinidev.code_intel._db import execute_with_retry

        def _q(conn):
            # Match the bare symbol name (last segment) to tolerate the model's
            # qualified-vs-bare naming. A hit anywhere means "exists".
            bare = name.split(".")[-1].split("(")[0]
            row = conn.execute(
                "SELECT 1 FROM ci_symbols WHERE project_id = ? AND "
                "(qualified_name = ? OR name = ?) LIMIT 1",
                (project_id, name, bare),
            ).fetchone()
            return row is not None

        return bool(execute_with_retry(_q))
    except Exception:
        return True  # index unavailable → never reject on symbol grounds
