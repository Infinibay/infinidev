# Conversation turns

With `TASK_ENGINE_MODE=task`, an ordinary interactive request reaches the principal
agent directly through `LoopEngine.run_turn`. The agent receives the literal current
message, separate session and host context, attachments, and its full granted tool
catalog. It decides whether to answer, ask, investigate, edit, or delegate. Chat
routing, specification elaboration, inferred task profiles, and Step phases do not
run before it can act. Gather runs only when explicitly enabled or requested.

## Returning control

A plain final response can end a turn before tools are attempted. After any tool
attempt, `finish_turn(message)` is required; plain text is a continuation and receives
a protocol reminder, bounded to three consecutive reminders. Neither form creates,
completes, blocks, or advances a work Task. `update_plan` supplies optional
work items with stable IDs and explicit statuses; item titles carry no execution
policy. Prior plans and notes are context for the current request.

The conversational outcomes are `answered`, `exhausted`, `cancelled`, and `failed`.
`answered` means the agent returned control to the user. Exhaustion preserves evidence
and reports interruption. Host permissions, cancellation, and resource limits remain
authoritative. An end-instruction hook can continue once within the same cancellation
scope and remaining model/tool budgets.

## Rollout paths

| Entry | Runtime contract |
| --- | --- |
| Ordinary interactive `task` | Direct principal conversation; no synthetic Task events |
| Explicit `--think` | Existing PhaseEngine adapter |
| Explicit `/auto` | Existing autonomous adapter and chain budgets |
| Configured `react`, `staged`, `orchestrator`, `auto`, or `graph_beta` | Existing selected-engine adapter |
| Direct flows such as `/init` and `/explore` | Existing flow entry points |

`LoopEngine.execute` remains the compatibility API for Task/Step execution and its
configured verification. The pipeline owns the final reply, transcript close, user
hooks, and Ken turn-end notification for the conversational path.

The historical measurements in [Engine Task closure analysis](ENGINE_TASK_CLOSURE_ANALYSIS.md)
compare legacy adapters. They do not establish the performance of `run_turn`.

## Prompt migration

Legacy `loop.*` prompt-fragment overrides apply to the `execute` compatibility
path. They do not customize `run_turn`. The principal honors the agent identity
and backstory, repository `AGENTS.md` instructions, and enabled optional
capabilities through its conversational prompt.

## Execution provenance

The host archives `turn_provenance` records in working memory with a unique run ID.
Records distinguish model responses (native or parsed calls), dispatcher results, and
turn closure. They contain tool names, hashed call IDs, content lengths and hashes,
without argument values or reasoning. Dispatcher results may include permission or
validation rejection; dispatch does not itself prove a side effect. Full tool outputs
continue to use the existing evidence archive.

The public host record reports dispatch count, whether the workspace tracker observed
changes, closure method and trace ID, separately from the model-authored reply. A
`finish_turn` message is still model-authored; accepting it does not certify its claims
or convert embedded tool-looking text into a real tool result. Trace archival failure
is logged and is not evidence that the recorded operation succeeded.

After tool attempts, the runtime requests `tool_choice="required"` while retaining the
full granted catalog, so the agent can continue work or close explicitly. Providers
that lack this option retain their existing downgrade/omission compatibility path.
The three-response protocol fuse is unchanged.

`runtime_request` records the requested protocol, roles and presence of finish_turn;
`protocol_reminder` records actual reminders appended to the conversation.
`provider_request` records tool names, roles and effective tool_choice at the input to
LiteLLM after controller adaptation and hooks, including retries. This is not a wire
capture and does not prove that a backend honored the option. No message content,
argument values, credentials or reasoning is included in these request records.

In manual mode, each parsed action is retained as an assistant JSON `tool_calls`
envelope before its host observation, including rejected calls. Markup stripped for
presentation must not erase that action from the next provider request. This history
is distinct from sanitized provenance records, which continue to omit arguments.

The workspace comes from the agent's bound ToolContext, with the agent attribute as
fallback for direct callers. The production InfinidevAgent stores its workspace in
ToolContext, not on the agent object. The same resolved root feeds project instructions
and the baseline that reconciles changes made outside file-edit tools.

OpenAI-compatible endpoints are probed for native tool support, using the same
bounded capability probes as llama.cpp and vLLM. Positive evidence enables native
function calling and supported tool_choice options; absent or failed support keeps
manual compatibility. Results are cached by the existing capability lifecycle.
The generic provider name alone no longer forces manual transport. Startup probes
advertise inert functions and inspect responses; they do not execute tools.

Set `INFINIDEV_TASK_POST_TOOL_CHOICE=auto` to keep automatic native tool
selection after a tool attempt in the principal turn. The default is
`required` when supported. The first request uses `auto` in either mode.
The tool catalog, grants and explicit `finish_turn` protocol still apply.
If optional readback requests a named tool and the endpoint rejects that
selector, its fallback honors the configured choice and provider capabilities.

Set `INFINIDEV_TASK_TEST_REVISION_FEEDBACK_ENABLED=true` to observe test
revisions in the principal engine. A recognised test command with an
unambiguous zero exit is associated with the observed workspace when the
executor records a launch directory inside it and the before and after
snapshots match. The executor omits this evidence for compound shell syntax.
The dispatcher can normalize a leading literal `cd … &&` into `cwd`, which
is checked against the workspace like any explicit `cwd`. A later workspace
change can add a host observation before the next request. The receipt and terminal show
`current`, `stale`, or `unconfirmed` for the latest attempt. The receipt
also records the time spent taking verification snapshots.

The principal saves a matched observation and its available archive locator
in the work checkpoint. At the next turn it compares the saved revision
against a fresh workspace snapshot. Results from earlier turns are marked
as such in the host receipt. A changed or unavailable revision, or a launch
directory no longer inside the workspace, clears the saved match. Older
observations without a recorded launch directory are discarded. The model
receives the archive locator when recall is granted.

The option is off by default. The model chooses whether to run another
authorised check, and `finish_turn` remains available. The status describes
the revision associated with the command result. The launch directory
identifies where the shell starts. Wrappers and runner arguments can select
tests elsewhere. Test execution and coverage require evidence from the
runner itself.

## Action channels and continuity

Native action authority comes only from the provider's tool_calls field. Native
content remains data even when it contains literal JSON/XML tool examples. Neither
prose examples nor discarded reasoning is promoted into execution. Manual turns
accept a complete visible JSON tool_calls envelope or complete Qwen JSON envelopes;
legacy execute format parsers remain separate. Tool arguments are opaque to prose
and thinking-tag transformations. Conversational calls do not force JSON response
format, so plain replies remain valid on JSON-capable manual backends.

Provenance records use kind runtime_trace and do not consume the forty evidence
references retained between turns or enter semantic embedding/retrieval. They remain
durable and available through explicit trace inspection. Historical rows retain
their original classification. Evidence titles have unique occurrence identifiers;
normal recall selects recent records by creation time across turns. An exact archive
title is resolved directly, including beyond the bounded semantic recall window and
without waiting for embeddings. This preserves known observations without deciding
which new user requests need investigation.

The trace records run_turn setup/baseline timings and per-model-call elapsed time
using a monotonic clock. Observed token deltas are reported counters for that call,
not cumulative context size, pricing or a complete account of failed transport
attempts. Model-call elapsed time includes transport, streaming and client handling;
setup timing excludes pipeline work before run_turn. These measurements support
latency diagnosis without recording arguments, prompt content or private reasoning.

## Public delivery

Completed assistant prose is a public channel. Native content accompanying tool calls
(including finish_turn), and nonterminal prose after tools, is handed to the pipeline
through on_public_message. Reasoning, tool observations and parsed manual action
envelopes do not use that channel. The final finish_turn message supplements the
preceding public messages rather than replacing their details.

The pipeline records preceding public parts in conversation_turns. It also mirrors
them into session_messages unless the renderer explicitly declares that notify already
persists them (TUI and web renderer). Existing frontend ownership of the returned final
reply remains unchanged. A first-pass reply is published once a hook continuation is
confirmed, before the next pass can refer to it. Cancellation/exhaustion preserve prior
public parts and produce one final interruption message.

TurnResult.public_messages contains the preceding public channel outputs for direct
runtime consumers; message remains the terminal reply. Callers without a callback
must consume both fields. Public-message trace entries contain hashes, sizes and a
callback handoff flag, not message bodies or proof that a person read the display.

Classic mirrored intermediate messages carry turn_complete=false. Supervisors must
use the terminal reply/host closure or actual idle state to detect readiness, rather
than treating every new assistant message as the end of an agent turn. This delivery
contract fixes lost public content; it does not certify arbitrary prose claims or
references to artifacts the agent has not verified.
