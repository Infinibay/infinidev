"""run_council — orchestrates the whole deliberation.

Flow:

    seed (moderator) ──► open threads on a fresh Channel
        │
        ▼
    for round in 1..MAX_ROUNDS:
        digest = channel.snapshot().render_digest()      # frozen view
        turns  = parallel(run_member_round, members)      # barrier
        apply turns to the channel in deterministic order # commit
        if all concluded: break
        if moderator.judge(channel) == converged: break
        │
        ▼
    synthesize (moderator) ──► DesignBrief

Returns the :class:`DesignBrief`, or ``None`` if the council is disabled
or fails before producing anything — the pipeline then proceeds straight
to the planner, exactly as it did before the council existed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Type

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from infinidev.config.secrets import redact
from infinidev.config.settings import settings
from infinidev.engine._best_effort import best_effort
from infinidev.engine.command_output_store import (
    COMMAND_OUTPUT_MAX_READ_BYTES,
    CommandOutputHandle,
    CommandOutputStore,
)
from infinidev.engine.council import moderator as MOD
from infinidev.engine.council.agent_loop import LoopResult, run_terminating_loop
from infinidev.engine.council.brief import DesignBrief
from infinidev.engine.council.channel import Channel
from infinidev.engine.council.member import MemberTurn, run_member_round
from infinidev.engine.working_memory import (
    NoteCitation,
    TraceableNoteEnvelope,
    create_traceable_note,
    get_working_memory,
)
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)

_COMMAND_ANALYSIS_MAX_ITERATIONS = 8
_COMMAND_ANALYSIS_MAX_READ_CALLS = 8
_COMMAND_ANALYSIS_MAX_TOTAL_BYTES = 256 * 1024
_COMMAND_ANALYSIS_MAX_QUESTION_CHARS = 2000
_COMMAND_ANALYSIS_MAX_CLAIMS = 6
_COMMAND_ANALYSIS_MAX_CITATIONS_PER_CLAIM = 4
_ANALYSIS_ID_RE = re.compile(r"^[a-zA-Z0-9:._-]{1,200}$")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?(?:key|token)|token|secret|password|passwd|credential|"
    r"private[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(sk-(?:proj-|ant-|or-)?[A-Za-z0-9_-]{12,}|"
    r"gh[op]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|"
    r"hf_[A-Za-z0-9_]{12,}|AIza[A-Za-z0-9_-]{12,})\b"
)


def _redact_private_text(text: str) -> str:
    """Remove configured and common inline credentials before any LLM sees them."""
    masked = redact(text)
    masked = _SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", masked)
    return _SECRET_VALUE_RE.sub("[REDACTED]", masked)


class _OutputRangeInput(BaseModel):
    """A byte range within the one output fixed by the caller."""

    model_config = ConfigDict(extra="forbid")

    offset: int = Field(0, ge=0)
    limit: int = Field(
        16_384,
        ge=1,
        le=COMMAND_OUTPUT_MAX_READ_BYTES,
        description="Maximum UTF-8 source bytes to return.",
    )


class _ScopedCommandOutputReader(InfinibayBaseTool):
    """Read only one prevalidated handle, under a finite per-agent budget."""

    is_read_only: bool = True
    name: str = "read_output_range"
    description: str = (
        "Read a bounded UTF-8 byte range from the single private command output "
        "assigned to this analysis. Offsets are bytes; no path or other artifact "
        "can be selected."
    )
    args_schema: Type[BaseModel] = _OutputRangeInput

    _store: CommandOutputStore = PrivateAttr()
    _handle: CommandOutputHandle = PrivateAttr()
    _project_id: int = PrivateAttr()
    _session_id: str = PrivateAttr()
    _calls: int = PrivateAttr(default=0)
    _bytes_returned: int = PrivateAttr(default=0)
    _ranges: list[tuple[int, int]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        *,
        store: CommandOutputStore,
        handle: CommandOutputHandle,
        project_id: int,
        session_id: str,
    ) -> None:
        super().__init__()
        self._store = store
        self._handle = handle
        self._project_id = project_id
        self._session_id = session_id

    @property
    def ranges_read(self) -> tuple[tuple[int, int], ...]:
        """Exact source intervals returned to the isolated subagent."""
        return tuple(self._ranges)

    def _run(self, offset: int = 0, limit: int = 16_384) -> str:
        if self._calls >= _COMMAND_ANALYSIS_MAX_READ_CALLS:
            return self._error("command-output analysis read-call budget exhausted")
        remaining = _COMMAND_ANALYSIS_MAX_TOTAL_BYTES - self._bytes_returned
        if remaining <= 0:
            return self._error("command-output analysis byte budget exhausted")
        bounded_limit = min(limit, remaining, COMMAND_OUTPUT_MAX_READ_BYTES)
        self._calls += 1
        try:
            content, start, end, has_more = self._store.read_range(
                self._handle,
                project_id=self._project_id,
                session_id=self._session_id,
                offset=offset,
                limit=bounded_limit,
            )
        except Exception as exc:
            logger.debug("Isolated command-output range read failed", exc_info=True)
            return self._error(f"command-output range rejected: {type(exc).__name__}")
        self._bytes_returned += end - start
        self._ranges.append((start, end))
        return self._success({
            "content": _redact_private_text(content),
            "start_offset": start,
            "end_offset": end,
            "returned_bytes": end - start,
            "has_more": has_more,
            "next_offset": end if has_more else None,
        })


class _AnalysisCitationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., gt=0)


class _AnalysisClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["fact", "inference"]
    statement: str = Field(..., min_length=1, max_length=1000)
    citations: list[_AnalysisCitationInput] = Field(
        ...,
        min_length=1,
        max_length=_COMMAND_ANALYSIS_MAX_CITATIONS_PER_CLAIM,
    )


class _SubmitAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=1, max_length=2000)
    claims: list[_AnalysisClaimInput] = Field(
        ...,
        min_length=1,
        max_length=_COMMAND_ANALYSIS_MAX_CLAIMS,
    )


class _SubmitAnalysisTool(InfinibayBaseTool):
    """Schema-only terminator for the isolated analysis loop."""

    is_read_only: bool = True
    name: str = "submit_artifact_analysis"
    description: str = (
        "Finish the analysis with a short summary and grounded claims. Every "
        "claim must be marked fact or inference and cite byte ranges returned by "
        "read_output_range."
    )
    args_schema: Type[BaseModel] = _SubmitAnalysisInput

    def _run(self, summary: str, claims: list) -> str:
        return self._success({"summary": summary, "claims": claims})


_COMMAND_ANALYSIS_SYSTEM_PROMPT = """\
You are an isolated command-output analyst. You can inspect exactly one opaque
output through read_output_range and cannot access paths, commands, files, web,
network, prior conversation, or project history. Read only the ranges needed.

Call submit_artifact_analysis exactly once. Every claim must:
- be marked `fact` when directly supported, or `inference` when reasoned;
- cite one or more complete [start_offset, end_offset) ranges exactly as returned;
- avoid copying credentials or long verbatim output into the statement.
Do not invent citations. The caller validates every range against what you read.
"""


def analyze_command_output(
    handle: CommandOutputHandle | dict[str, Any],
    question: str,
    *,
    project_id: int,
    session_id: str,
    step_index: int = 0,
    tool_call_id: str | None = None,
    max_iterations: int = 6,
    store: CommandOutputStore | None = None,
) -> TraceableNoteEnvelope | None:
    """Explicitly analyse one private command-output handle in isolation.

    This API is additive and never runs from :func:`run_council` implicitly.
    The subagent receives only a fixed-handle range reader and a structured
    terminator. It has no workspace/session context and no developer, web,
    knowledge, shell, or history tools. Invalid handles, budgets, citations,
    model failures, and persistence failures all fail closed with ``None``.
    """
    try:
        scoped_handle = _coerce_command_output_handle(handle)
        if type(project_id) is not int or project_id <= 0:
            raise ValueError("project_id must be a positive integer")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id is required")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("analysis question is required")
        if len(question) > _COMMAND_ANALYSIS_MAX_QUESTION_CHARS:
            raise ValueError("analysis question exceeds its bounded size")
        if (
            type(max_iterations) is not int
            or not 1 <= max_iterations <= _COMMAND_ANALYSIS_MAX_ITERATIONS
        ):
            raise ValueError("analysis iteration budget is invalid")

        private_store = store or CommandOutputStore()
        _validate_command_output_handle(
            private_store, scoped_handle, project_id=project_id, session_id=session_id
        )
        reader = _ScopedCommandOutputReader(
            store=private_store,
            handle=scoped_handle,
            project_id=project_id,
            session_id=session_id,
        )
        tools: list[InfinibayBaseTool] = [reader, _SubmitAnalysisTool()]
        result = run_terminating_loop(
            system_prompt=_COMMAND_ANALYSIS_SYSTEM_PROMPT,
            user_content=(
                "Question: " + _redact_private_text(question.strip()) + "\n"
                f"Opaque source length: {scoped_handle.byte_count} bytes."
            ),
            tools=tools,
            terminator_names={"submit_artifact_analysis"},
            max_iterations=max_iterations,
            agent_id_prefix="command-output-analysis",
            # Deliberately withhold all ambient scope. The fixed reader carries
            # only the two identities needed for its private store check.
            project_id=None,
            session_id=None,
            workspace_path=None,
            temperature=0.1,
            max_tokens=1600,
        )
        payload = _validate_analysis_result(
            result,
            reader=reader,
            store=private_store,
            handle=scoped_handle,
            project_id=project_id,
            session_id=session_id,
        )
        note = _analysis_note(
            payload,
            handle=scoped_handle,
            question=question,
            step_index=step_index,
            tool_call_id=tool_call_id,
        )
        memory = get_working_memory(session_id)
        if memory.remember_traceable(note):
            return note
        existing = next(
            (
                item for item in memory.load_traceable_notes(
                    kinds=("artifact_analysis",)
                )
                if item.occurrence_id == note.occurrence_id
            ),
            None,
        )
        return existing
    except Exception:
        logger.debug("Command-output analysis failed closed", exc_info=True)
        return None


def _coerce_command_output_handle(
    value: CommandOutputHandle | dict[str, Any],
) -> CommandOutputHandle:
    if isinstance(value, CommandOutputHandle):
        return value
    if not isinstance(value, dict) or set(value) != {
        "artifact_id", "type", "stream", "char_count", "byte_count",
    }:
        raise ValueError("invalid command-output handle")
    return CommandOutputHandle(
        artifact_id=value["artifact_id"],
        artifact_type=value["type"],
        stream=value["stream"],
        char_count=value["char_count"],
        byte_count=value["byte_count"],
    )


def _validate_command_output_handle(
    store: CommandOutputStore,
    handle: CommandOutputHandle,
    *,
    project_id: int,
    session_id: str,
) -> None:
    # Validation happens before any model call. A four-byte read can represent
    # every valid UTF-8 scalar while keeping the preflight bounded.
    limit = min(max(handle.byte_count, 1), 4)
    store.read_range(
        handle,
        project_id=project_id,
        session_id=session_id,
        offset=0,
        limit=limit,
    )


def _validate_analysis_result(
    result: LoopResult,
    *,
    reader: _ScopedCommandOutputReader,
    store: CommandOutputStore,
    handle: CommandOutputHandle,
    project_id: int,
    session_id: str,
) -> _SubmitAnalysisInput:
    if result.terminator != "submit_artifact_analysis":
        raise ValueError("analysis ended without its terminator")
    payload = _SubmitAnalysisInput.model_validate(result.args)
    observed = reader.ranges_read
    if not observed:
        raise ValueError("analysis cited output without reading it")
    for claim in payload.claims:
        for citation in claim.citations:
            start = citation.start_offset
            end = citation.end_offset
            if start >= end or end > handle.byte_count:
                raise ValueError("analysis citation is outside the artifact")
            if (start, end) not in observed:
                raise ValueError("analysis citation was not an exact returned range")
            # This also proves both cited boundaries are valid UTF-8 boundaries
            # against the still-scoped, hash-verified artifact.
            _, exact_start, exact_end, _ = store.read_range(
                handle,
                project_id=project_id,
                session_id=session_id,
                offset=start,
                limit=end - start,
            )
            if exact_start != start or exact_end != end:
                raise ValueError("analysis citation is not an exact UTF-8 range")
    return payload


def _analysis_note(
    payload: _SubmitAnalysisInput,
    *,
    handle: CommandOutputHandle,
    question: str,
    step_index: int,
    tool_call_id: str | None,
) -> TraceableNoteEnvelope:
    lines = [_redact_private_text(payload.summary.strip())]
    canonical_claims: list[dict[str, Any]] = []
    for claim in payload.claims:
        ranges = [
            [citation.start_offset, citation.end_offset]
            for citation in claim.citations
        ]
        canonical_claims.append({
            "kind": claim.kind,
            "statement": _redact_private_text(claim.statement.strip()),
            "ranges": ranges,
        })
        rendered_ranges = ", ".join(f"bytes {start}:{end}" for start, end in ranges)
        lines.append(
            f"{claim.kind.upper()} [{rendered_ranges}]: "
            f"{_redact_private_text(claim.statement.strip())}"
        )
    identity = json.dumps(
        {
            "artifact_id": handle.artifact_id,
            "question": _redact_private_text(question.strip()),
            "claims": canonical_claims,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    occurrence_id = "artifact-analysis:" + hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()
    if not _ANALYSIS_ID_RE.fullmatch(occurrence_id):
        raise ValueError("invalid analysis occurrence identity")
    source_occurrence = f"command-output:{handle.artifact_id}"
    citation = NoteCitation(
        occurrence_id=source_occurrence,
        source_artifact_id=handle.artifact_id,
        step_index=step_index,
        tool_call_id=tool_call_id,
    )
    return create_traceable_note(
        "artifact_analysis",
        "\n".join(lines),
        source_artifact_id=handle.artifact_id,
        step_index=step_index,
        tool_call_id=tool_call_id,
        occurrence_id=occurrence_id,
        citations=(citation,),
    )


def run_council(
    handoff: str,
    *,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
    hooks: Any | None = None,
) -> DesignBrief | None:
    """Deliberate on ``handoff`` and return a synthesised DesignBrief.

    ``handoff`` is free text describing the escalated request (the
    pipeline builds it from the EscalationPacket). ``hooks`` is an
    optional :class:`OrchestrationHooks` used purely to surface the
    debate to the user (each post → ``notify('Council:<id>', ...)``).
    """
    if not settings.COUNCIL_ENABLED:
        return None

    ctx = dict(
        session_id=session_id, project_id=project_id, workspace_path=workspace_path,
    )

    def _status(level: str, msg: str) -> None:
        if hooks is not None:
            with best_effort("council on_status hook failed"):
                hooks.on_status(level, msg)

    def _say(speaker: str, msg: str) -> None:
        if hooks is not None and msg:
            with best_effort("council notify hook failed"):
                hooks.notify(speaker, msg, "agent")

    try:
        # ── Seed ─────────────────────────────────────────────────────────
        _status("info", "Convening council — assigning roles...")
        roster = MOD.seed_council(handoff, **ctx)
        if not roster.members:
            logger.warning("Council seeded with no members; skipping")
            return None

        _say(
            "Council",
            "Convening a council of "
            f"{len(roster.members)} agents to debate:\n"
            f"  {roster.question}\n\n"
            + "\n".join(
                f"  - {m.member_id}: {m.objective}" for m in roster.members
            ),
        )

        channel = Channel(roster.question)
        for ot in roster.opening_threads:
            channel.open_thread(
                author="moderator", title=ot.title,
                opening_text=ot.prompt, round=0,
            )

        # ── Rounds ───────────────────────────────────────────────────────
        concluded: set[str] = set()
        final_positions: dict[str, str] = {}
        max_rounds = max(1, settings.COUNCIL_MAX_ROUNDS)

        for round_num in range(1, max_rounds + 1):
            active = [m for m in roster.members if m.member_id not in concluded]
            if not active:
                break
            _status("info", f"Council round {round_num}/{max_rounds}...")

            snapshot = channel.snapshot()
            turns = _run_round_parallel(
                active, question=roster.question, snapshot=snapshot,
                round_num=round_num, ctx=ctx,
            )

            # Commit in deterministic order (by member_id) so the channel
            # is reproducible regardless of thread completion order.
            for turn in sorted(turns, key=lambda t: t.member_id):
                _commit_turn(turn, channel, round_num, concluded, final_positions, _say)

            if len(concluded) >= len(roster.members):
                _status("info", "All members concluded — closing council.")
                break
            if round_num >= max_rounds:
                break

            converged, reason = MOD.judge_convergence(
                channel.snapshot().render_digest(current_round=round_num),
                round_num, **ctx,
            )
            if converged:
                _status("info", f"Council converged: {reason}")
                break

        # ── Synthesize ───────────────────────────────────────────────────
        _status("info", "Synthesising design brief...")
        full_digest = channel.snapshot().render_digest(
            current_round=max_rounds + 1, recent_rounds=max_rounds + 1,
        )
        if final_positions:
            full_digest += "\n\nFINAL POSITIONS:\n" + "\n".join(
                f"  - {mid}: {pos}" for mid, pos in sorted(final_positions.items())
            )
        brief = MOD.synthesize(full_digest, roster.question, **ctx)
        return brief
    except Exception:
        logger.exception("Council failed; proceeding without a design brief")
        _status("warn", "Council failed — proceeding without a design brief.")
        return None


def _run_round_parallel(
    members: list, *, question: str, snapshot: Channel, round_num: int,
    ctx: dict,
) -> list[MemberTurn]:
    """Fan members out across a thread pool; barrier on all results.

    On a single local LLM server these serialise at the GPU — the
    parallelism buys deliberation quality, not wall-clock. With a cloud
    COUNCIL_MODEL it is genuinely concurrent.
    """
    max_workers = max(1, min(settings.COUNCIL_MAX_CONCURRENCY, len(members)))

    def _one(assignment) -> MemberTurn:
        digest = snapshot.render_digest(
            for_author=assignment.member_id, current_round=round_num,
        )
        return run_member_round(
            assignment, question=question, digest=digest, round_num=round_num,
            **ctx,
        )

    if max_workers == 1:
        return [_one(m) for m in members]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, members))


def _commit_turn(
    turn: MemberTurn, channel: Channel, round_num: int,
    concluded: set[str], final_positions: dict[str, str], say,
) -> None:
    """Apply one member's decided action to the live channel."""
    if turn.action == "post":
        if turn.new_thread_title:
            channel.open_thread(
                author=turn.member_id, title=turn.new_thread_title,
                opening_text=turn.message, round=round_num,
                refs=list(turn.refs),
            )
        else:
            tid = turn.thread_id or _first_thread_id(channel)
            posted = channel.post(
                author=turn.member_id, thread_id=tid, text=turn.message,
                round=round_num, parent_id=turn.parent_id or None,
                refs=list(turn.refs),
            )
            if posted is None and tid != _first_thread_id(channel):
                # Stale thread id — retry into the first thread so the
                # contribution isn't silently lost.
                channel.post(
                    author=turn.member_id, thread_id=_first_thread_id(channel),
                    text=turn.message, round=round_num, refs=list(turn.refs),
                )
        say(f"Council:{turn.member_id}", turn.message)
    elif turn.action == "conclude":
        concluded.add(turn.member_id)
        if turn.final_position:
            final_positions[turn.member_id] = turn.final_position
            say(
                f"Council:{turn.member_id}",
                f"[concluye] {turn.final_position}",
            )


def _first_thread_id(channel: Channel) -> str:
    titles = channel.thread_titles
    return titles[0][0] if titles else ""


__all__ = ["analyze_command_output", "run_council"]
