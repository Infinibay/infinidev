# Agent-owned conversation

The user authorizes changing fundamental Infinidev architecture and rejects regular
expressions as a way to interpret requests or impose execution policy. Bruma's real
session demonstrates the cost of interpreting the same request in several stages.

## Contract

The default `task` entry is a conversation owned by one capable agent. The literal
current request, conversation context and host-authorized tools reach that agent
directly. It can answer, investigate, edit or delegate through its available tools.
There is no chat routing decision, mandatory specification elaboration, inferred
task kind, inferred Step phase, or text-based tool selection before it can act.

A conversation turn and a work Task are distinct. `finish_turn(message)` returns
control to the user. A clean text-only final reply is the same conversational
outcome. Neither creates, completes, blocks or advances a Task. Optional work plans
are updated explicitly by stable item IDs and declared statuses; titles never
control execution. The agent sees prior state as context and decides what the
current user message changes. No textual request-identity matching replays work.

The runtime owns protocol schemas, host permissions, cancellation, real resource
budgets, tool/result ordering, streaming, context pressure, archival and persistence.
Budget exhaustion returns an interrupted outcome with retained evidence. It must
not reset a budget by inventing another Step or claim success.

Existing high-level caller/runner/context entry points contain legacy policy. Reuse
their boundary primitives, not their Step vetoes, discovery suppression, inferred
argument repairs or Step-specific compaction wrappers. Hook continuations share the
remaining budget and cancellation scope. Hook outcome labels are conversational
(`answered`, `exhausted`, `cancelled`, `failed`), not Task completion labels.

Existing `LoopEngine.execute` and non-default engine adapters retain their explicit
contracts, including configured objective verification. The new conversational
method reuses mechanical caller/tool/context infrastructure without entering legacy
Step reconciliation, no-diff restrictions or completion classifiers. User-configured
turn hooks remain effective. No new public engine mode or dependency is introduced.
Explicit `--think` and `/auto` compatibility workflows retain their current adapter
contracts during this migration; ordinary default interactive turns use the new
owner. This is explicit host configuration, never a classification of user prose.

## Approach

Adding another intent classifier still duplicates ownership. Broadening heuristics
preserves the same defect. Reusing the existing read-only orchestrator would force
delegation even for a small edit. Instead add a first-class conversational method to
the execution runtime and route the default entry to it. Keep legacy execution as a
compatibility API rather than rewriting every adapter simultaneously.

## Validation

Cover greeting and clarification without tools, research without edits, tests-only
edits, explicit intermediate delivery without forced validation, opaque plan titles,
all granted tools remaining available across repeated reads, native/manual tool
calls, malformed protocol, cancellation during inference and tools, global budgets,
context compaction, attachments, session continuation and absence of fake Task events.
Bruma validation is coordinated at a safe process restart; no competing local model
inference is launched during its active work.
