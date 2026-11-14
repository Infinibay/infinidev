# Agent-owned Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the default user-facing agent ownership of interpretation and action,
with a mechanical runtime and a separate conversational close.

**Architecture:** Add `LoopEngine.run_turn` alongside the compatible `execute` API.
The default pipeline calls it directly, bypassing semantic routing and forced Task
creation. Reuse provider calls, tool dispatch, cancellation and context infrastructure.

**Tech Stack:** Existing Python, Pydantic, LiteLLM and SQLite services; uv tooling.

**Spec:** `docs/superpowers/specs/2026-11-13-agent-owned-conversation.md`.

## Global Constraints

- No new intent regex, classifier inference, public engine mode or dependency.
- Current user text remains literal; injected context is separately marked.
- Host permissions and explicit limits remain authoritative; exhaustion is not success.
- Preserve legacy engines, model capability probes and manual tool-call compatibility.
- Keep the Bruma coordinator's report and active process intact.

## Review Focus

- A response or question must never imply Task completion.
- A malformed tool payload must not become an apparently successful final answer.
- A late user message or cancellation must preempt a terminal reply correctly.
- A new request may use old evidence without inheriting stale execution obligations.
- A tool batch must not execute mutations after a terminal call or beyond its budget.

## Task 1: Conversational state and explicit tools

Files: create `engine/turns/models.py`, `engine/turns/tools.py` and focused tests.

Interfaces: `PlanItem(id, title, status)` with statuses pending/in_progress/completed/
blocked; `TurnState(plan=[], notes=[])`; `TurnResult(message, status, state)` with
statuses answered/cancelled/exhausted/failed. Runtime metadata may include engine,
metrics and tool counts. `finish_turn(message)` closes a turn only; `update_plan`
atomically supplies the plan and validates unique stable IDs. Build bound tools
with `build_turn_tools(state)`; runtime consumes the validated terminal result.

- [x] Write failing tests for schema errors, duplicate IDs, explicit plan updates,
      opaque titles and conversational close without Task mutation.
- [x] Implement tools using existing BaseTool and validation contracts.
- [x] Run focused tests and review interfaces with Task 2 (18 passed; review approved).

## Task 2: Mechanical conversational runtime

Files: create `engine/turns/runtime.py`, add `LoopEngine.run_turn`, focused tests.

Consumes Task 1 interfaces. Produces `run_turn(agent, user_input, *, turn_context='',
initial_attachments=None, task_tools=None, max_iterations=None,
max_total_tool_calls=None, prompt_configuration=None) -> TurnResult`.

- [x] Write failing tests for plain final replies, tools then final, terminal-only
      batches, malformed payloads, cancellation, budget exhaustion and repeated reads.
- [x] Build full host-authorized tools plus turn tools, excluding legacy Step/plan
      terminators. Reuse mechanical provider/dispatch/transcript primitives and
      pressure detection, archival and compaction, with a conversational wrapper; bypass
      Step guards, phase inference, automatic review and tool suppression.
- [x] Preserve provider/manual formats, usage counters and callback events.
- [x] Persist bounded per-session state through existing DB services, load it as
      context rather than replay authority, and retain archival result references.
- [x] Run focused runtime, protocol, context and cancellation tests.

## Task 3: Default pipeline integration and migration

Files: pipeline.py, principal prompt, session context as needed, pipeline tests.

- [x] Write failing tests proving default short requests reach `run_turn` without
      chat/spec/profile/council classification or Task events.
- [x] Route configured task mode directly after common context resolution. Supply
      literal input, prior session context, attachments and explicit budgets.
- [x] Finalize turn transcript, hooks, streaming and Ken events without fake work
      completion. Hook continuations consume remaining limits without clearing
      cancellation. Preserve legacy branches for explicit --think and /auto workflows.
- [x] Verify greeting, research, tests-only edit and intermediate delivery cases
      with scripted model boundaries; test actual tool effects where meaningful.
- [x] Run combined affected tests, then the full suite; report baseline failures
      separately. Review the complete diff before integration into the live checkout.

No commits, installation or live-session restart are implicit in this plan.
