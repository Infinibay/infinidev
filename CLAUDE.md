# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infinidev is a terminal-based AI programming tool that uses an agent loop to execute tasks autonomously. It's a CLI-focused evolution of the Infinibay multi-agent system, designed to work with local open-weight LLMs (via Ollama/LiteLLM) on consumer hardware. The `research_copy/` directory contains the original web-based Infinibay prototype (backend + React frontend) and is not part of the active CLI tool.

## Build & Run

```bash
# Install (creates venv in ~/.infinidev/venv, wrapper in ~/.local/bin/infinidev)
./install.sh

# Or develop locally with uv
uv sync
uv run infinidev          # launch CLI (TUI mode by default)
uv run infinidev --classic # legacy text-only mode

# Run tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_foo.py::test_bar -v
```

Entry point: `infinidev.cli.main:main` (defined in `pyproject.toml [project.scripts]`).

Settings live at `~/.infinidev/settings.json` and are reloaded on each interaction. DB is SQLite at `~/.infinidev/infinidev.db`.

## Architecture

### Pipeline (`engine/orchestration/pipeline.py`)

Every user turn runs through a **chat-agent-first** pipeline:

```
user message
  ↓
ChatAgent (read-only, default)         ← orchestration/chat_agent.py
  ↓ respond? return reply, done.
  ↓ escalate → EscalationPacket
AnalystPlanner                         ← analysis/planner.py
  ↓ emits Plan(overview, steps[])
Gather (optional, develop flow)        ← gather/runner.py
  ↓
LoopEngine.execute(initial_plan=plan)  ← loop/engine.py
  ↓
Review (runs if files changed)         ← analysis/review_engine.py
```

The `ChatAgent` owns a short (~20 iteration) read-only LLM loop with
the `respond`/`escalate` terminator tools. The `AnalystPlanner` owns a
budgeted loop with the `emit_plan` terminator. Neither uses the
LoopEngine — they call litellm directly; the LoopEngine is reserved
for the developer's heavy plan-execute-summarize loop.

### Loop Engine (`engine/loop/engine.py`)

The developer stage is a **plan-execute-summarize** cycle, not a ReAct loop:

1. **Plan** — either seeded upfront from `initial_plan` (chat-agent-first path, steps marked `user_approved=True`) or bootstrapped by the LLM via `add_step` calls (legacy PhaseEngine path).
2. **Execute** — One step at a time, up to 4 tool calls per step.
3. **Summarize** — LLM produces a ~50-token summary; raw tool output is *archived, not discarded* (see Working memory).
4. **Repeat** — Prompt is rebuilt from scratch each iteration using only compact summaries.

### Working memory (`engine/working_memory.py`)

Eviction from the model's context is recoverable rather than destructive.
When a step closes, `StepManager._archive_evicted_context` writes every
tool call + result of that step into the `working_memory` SQLite table and
queues an embedding on a background worker. The model gets it back with
the `recall_context` tool (cosine search, keyword fallback), so it can
retrieve an error message or a file listing from six steps ago instead of
re-running the command.

Retention in the prompt is explicit: the newest `WORKING_MEMORY_VERBATIM_STEPS`
summaries render in full, older ones collapse to one line, and the block
tells the model that the detail is one `recall_context` call away. The
*chat transcript* is never touched by any of this — it is the model's
memory that gets compacted, not the user's conversation.

Two gotchas worth knowing: `execute_with_retry` does **not** commit, so
every writer commits inside its own callback (otherwise the embedding
worker's connection cannot see the rows), and each record carries the
`db_path` it was written to because the embed worker is process-wide.

The LLM signals step completion via a `step_complete` tool call with `status` (continue/done/blocked), `summary`, and optional plan modifications (add/modify/remove steps). `user_approved` steps are protected — `apply_operations` rejects remove/modify on them so the LLM cannot rewrite an analyst-produced plan mid-execution.

**`engine.py` holds the shape of a run; the work lives in collaborators.**
Each is a module in `loop/`, and knowing which one owns a concern is
usually enough to find the code:

| module | owns |
|---|---|
| `context_builder.py` | what a run needs (once) and what each iteration says (per step) |
| `llm_caller.py` | one model call: retries, streaming, manual-mode parsing |
| `tool_processor.py` | classifying calls into real tools vs pseudo-tools |
| `tool_runner.py` | executing them and writing results into `messages` |
| `critic_liaison.py` | the pair-programming critic: advisory, except one veto |
| `step_complete_gate.py` | the four reasons a step may not close |
| `loop_guard.py` | repetition, error circuits, text-only stalls |
| `step_manager.py` | advancing the plan, summarising, finishing |
| `run_report.py` | what the finished run tells the reviewer |

`step_complete` is the model's *claim* that a step is done, and
`StepCompleteGate` is the chain that can override it — notes, a user
message that arrived mid-generation, the critic, then the step's own
deterministic verification. Ordered cheapest-first, so a step that fails
the note check never pays for an LLM call. Every gate refuses the same way:
by overwriting the `step_complete` tool result, which the model reads as
"your close was overridden" — far more reliable than a user-role message,
because following a tool result is its natural mode after a tool call.

**Dual tool-calling modes:** The engine auto-detects whether the LLM supports native function calling (FC mode) or falls back to parsing tool calls from text JSON (manual mode). Detection happens at startup via `config/model_capabilities.py`.

### Prompt Construction

Every iteration builds an XML-structured prompt: `<task>`, `<plan-overview>` (stable prose, set once by the planner), `<plan>` (step list), `<previous-actions>`, `<current-action>` (active step's full `detail`), `<next-actions>`, `<expected-output>`. The protocol rules are in `engine/loop/prompt/text.py` as `LOOP_PROTOCOL`. Per-step `detail` renders ONLY for the active step to keep context compact.

### Tools (`tools/`)

All tools inherit from `InfinibayBaseTool` (extends CrewAI's `BaseTool`). Tools are bound to agents via `bind_tools_to_agent()` and resolve context (project_id, task_id, workspace_path) from a process-global dict.

Categories:
- **file**: `read_file`, `partial_read`, `create_file`, `replace_lines`, `list_directory`, `code_search`, `glob`
- **code_intel**: `get_symbol_code`, `list_symbols`, `search_symbols`, `find_references`, `edit_symbol`, `add_symbol`, `remove_symbol`, `project_structure`
- **git**: `git_branch`, `git_commit`, `git_diff`, `git_status`
- **shell**: `execute_command`, `code_interpreter`
- **knowledge**: `record_finding`, `read_findings`, `search_findings` (with semantic dedup)
- **meta**: `help` (dynamic tool documentation), `recall_context` (working-memory retrieval)
- **mcp** (`tools/mcp_bridge.py`): every tool a configured MCP server publishes, under the server's own name — `ken_rank`, `ken_recall`, `ken_callgraph`, … Discovered at runtime from `tools/list`, never hand-written.
- **chat_agent** (tier-exclusive): `respond`, `escalate` — terminators for the chat agent's loop; never bound to the developer.
- **planner** (tier-exclusive): `emit_plan` — the planner's single-shot terminator that produces the `Plan` artifact.

The base class exposes `is_read_only: bool = False`. The 18 pure-read tools (file reads, code-intel lookups, git diff/status, findings reads) override it to `True`. `get_tools_for_role("chat_agent")` and `get_tools_for_role("planner")` filter the full toolset by this attribute — the schema passed to LiteLLM is the security boundary, not prompt rules.

Key tool design: `read_file` auto-indexes files via tree-sitter for code intelligence. `replace_lines` uses deterministic line-range replacement (no text matching). Symbol tools (`edit_symbol`, `add_symbol`, `remove_symbol`) use the code index to locate symbols by qualified name.

Tool schemas are validated at runtime — hallucinated parameters are rejected before execution. Old tool names (`edit_method`, `add_method`, `remove_method`, `write_file`, `find_definition`) are aliased to new names in `engine/loop_tools.py`.

### MCP and Ken (`engine/mcp_client.py`, `engine/ken_client.py`)

Infinidev is an MCP **host**. Servers are declared in `.mcp.json`
(workspace) or `~/.infinidev/mcp.json`; with no config at all, `ken mcp`
is registered as the default server. `McpServerClient` speaks real
stdio JSON-RPC: `initialize` + `notifications/initialized` handshake,
responses matched by `id` (servers interleave notifications), one reader
thread per stream so a full pipe can never deadlock the child, real
per-request deadlines, and exponential backoff (cap 8 s) on crashes.

**Server tools reach the model under the server's own names.**
`tools/mcp_bridge.py` reads `tools/list` and generates one
`InfinibayBaseTool` subclass per remote tool, keeping its name, its
description and its JSON Schema. Nothing in the tool layer knows what Ken
is — point the host at another MCP server and its tools appear too. Three
things the raw listing needs first:

- **Descriptions are compressed to the first paragraph.** Python-SDK
  servers publish the whole docstring; Ken's 30 tools cost ~6 100 tokens
  of schema that way, ~2 500 after. The parameter walk-through in those
  docstrings duplicates the schema sitting beside it.
- **Writers are separated from readers.** The spec's
  `annotations.readOnlyHint` wins when a server sends one (Ken sends
  none); otherwise the name is matched against a mutating-verb pattern and
  a match means "assume it writes". This is what keeps `ken_remember` out
  of the read-only tiers, which is a security boundary, not a hint.
- **A local tool always wins a name collision** — a remote server must not
  be able to shadow `read_file`.

`MCP_TOOL_FILTER` (comma-separated globs, default `*`) narrows what
reaches the schema; small models get MCP tools only once it is set,
because a 90-tool schema hurts their selection more than the extra
capability helps.

`KenClient` (`engine/ken_client.py`) is a separate, *internal* path: it
backs the deterministic tools, not the model's toolbox. **Ken augments,
never gates** — every method falls back to a local implementation when the
server is missing or the workspace has no `.ken` index, so `code_search`
and `glob` stay deterministic (git grep / pathlib).

`/mcp` shows per-server health (running / idle / failed + stderr tail) and
takes `start`/`stop`/`restart`.

**Ken owns the project index.** The TUI no longer runs a full tree-sitter
sweep at startup — that indexed the same tree Ken already indexes, cost
seconds before the user could type, and narrated itself in the transcript.
The local index has not gone away: the symbol *writer* tools
(`edit_symbol`, `add_symbol`, `get_symbol_code`) need line-accurate
positions Ken does not provide, and every tool that needs it indexes the
file it is about to touch on demand. `/reindex` still forces a full sweep.

### User hooks (`engine/user_hooks/`)

Shell commands the user binds to six lifecycle points in
`.infinidev/hooks.json` (workspace) or `~/.infinidev/hooks.json` (global).
Not to be confused with `engine/hooks/`, which is the *in-process* hook
manager the engine registers Python callbacks on — that one can rewrite a
tool call, this one can only contribute text. Nothing ships enabled: with
no config file the subsystem is inert.

A hook declares `command` (stdout is the output) **or** `prompt` (fixed
text, no subprocess). Config merges **per event**: the first file to
declare an event owns it outright, so `"task_start": []` in a project is
how you switch a global hook off there.

```json
{"hooks": {
  "step_end_instruction": [{"prompt": "Review the diff and fix what is wrong."}],
  "task_end_summary":     [{"command": "git diff --stat", "timeout": 30}]
}}
```

**The start / end / end triad is the design.** A task and a step each get
one start hook and *two* end hooks, and the pair differs in one thing —
which side of the summarisation boundary the output lands on:

| event | fires | output |
|---|---|---|
| `task_start` | turn opens | into the chat agent's input *and* the developer's task prompt |
| `step_start` | each iteration | appended to the step's user message |
| `step_end_instruction` | model calls `step_complete` | overwrites the tool result, step stays open — **dies with the step** |
| `step_end_summary` | step really closes | `ActionRecord.hook_notes` — **survives every later prompt** |
| `task_end_instruction` | after Review | re-enters `run_task` for one more pass — **not written to history** |
| `task_end_summary` | turn really closes | stored as a hidden `work_summary` turn — **the next turn reads it** |

Four things that are load-bearing rather than incidental:

- **The instruction hooks fire once.** Per step index, per turn. The
  model's second `step_complete` is indistinguishable from its first, so
  a hook that re-fired would hold the step forever. A hook that printed
  *nothing* still burns its one shot, or a sometimes-silent hook could
  fire twice inside one step.
- **`step_end_instruction` is the last gate, after the engine's own four**
  (see `loop/step_complete_gate.py`). The user's command is never asked to
  comment on a step that failed its own verification, and it does not
  re-run for each correction turn a failing check costs.
- **`task_end_instruction` re-enters `run_task`, not the loop.** The
  follow-up is classified, planned, executed and reviewed like any turn —
  a hook asking for a deep review gets the whole pipeline. `_hook_reentry`
  suppresses `task_start` and the instruction hook on that pass, which is
  what bounds it to exactly one.
- **Hooks are fail-open and deadlined.** Non-zero exit, timeout, missing
  binary, malformed JSON: all cost the hook's output and a log line,
  never the run. Output is capped at 8 000 chars — it goes straight into
  the prompt.

Payload reaches the command twice: full JSON on stdin, scalars as
`INFINIDEV_HOOK_*` env vars (`INFINIDEV_HOOK_STEP_INDEX`,
`INFINIDEV_HOOK_FILES_CHANGED`, …) for the one-liners people actually
write. cwd is the workspace. `HOOKS_ENABLED` / `HOOKS_TIMEOUT` in
settings; config reloads on mtime change, no restart.

### Agent (`agents/base.py`)

`InfinidevAgent` holds role metadata, binds tools based on role, and manages execution context. The CLI creates one agent with role="developer" per user instruction.

### TUI (`ui/`)

prompt_toolkit, not Textual. The layout is **transcript-first**
(`ui/layout.py`): the conversation owns the full terminal width, the
composer (`ui/controls/composer.py`) sits under it in a rounded frame with
an inline ghost placeholder, and one status line (`ui/controls/status_line.py`)
closes the screen with model · branch · context-left on the left and key
hints on the right. The explorer (Ctrl+B) and the sidebar with its five
panels (Alt+.) still exist in full — they are toggles, closed by default.

Message rendering (`ui/controls/message_widgets.py`) is deliberately
undecorated: one `>` opening the user's turn, *nothing* on an assistant
reply beyond a two-space indent, `·` for system notices, and no
per-message background so the terminal theme shows through. Headers are
drawn only for named speakers (Reviewer, critic verdicts) — see
`_BORDERED_CONFIG` and `_GENERIC_SENDERS`. Conversation turns are never
grouped (`NEVER_GROUP_TYPES`); consecutive *tool calls* fold into one
collapsible `✓ Ran N tools ▸` group (`build_tool_group`).

The pair-programming critic gets the same treatment
(`controls/critic_widget.py`, message type `critic`): `◇ Critic · N notes ▸`,
collapsed by default, expanding to one line per verdict and then to the
full body. It talks on most steps, so rendered as full system messages it
buried the assistant's reply — and its severity colours used to be amber,
which won every contest for the eye. Two things differ from the tool
group: **one** verdict also collapses (a single critic paragraph is just
as interruptive as three), and the summary line counts rejects
separately, since a reject changed what the model did and a
recommendation usually did not. Severity/model/source travel as fields on
the message dict (`add_message(**fields)`), not baked into the body —
the renderer counts rejects without parsing prose, and a body prefixed
with `(model)` would poison the preview line.

Modals share one hand-rolled frame (`ui/dialogs/base.py`): rounded
corners, title inlined into the top border, key hints in the bottom
border, one column of interior padding. `prompt_toolkit.widgets.Frame`
cannot do any of those three, which is why it is not used.

Two tests guard this: `tests/test_tui_render.py` draws the app into an
in-memory `Screen` (fast, no terminal), and `tests/test_tui_smoke.py`
boots the real `Application` and feeds it keystrokes — use the second
when touching key bindings or startup.

### Config (`config/`)

- `settings.py` — All settings use `INFINIBAY_` env var prefix. Key: `LLM_MODEL` (LiteLLM format like `ollama/qwen2.5-coder:7b`), `LLM_BASE_URL`, `SANDBOX_ENABLED`, loop limits.
- `llm.py` — `get_litellm_params()` builds the dict for `litellm.completion()`.
- `model_capabilities.py` — Runtime probing of FC support, JSON mode, schema sanitization needs.
- `openai_oauth.py`, `codex_catalog.py` — the ChatGPT-subscription provider (below).

### ChatGPT subscription (`openai_subscription`)

`LLM_PROVIDER=openai_subscription` bills against the user's ChatGPT plan
instead of a metered API key. Three things make that work, and each one is
somewhere a naive implementation goes wrong:

**The protocol is the Responses API, not chat completions.** The Codex
backend serves no `/chat/completions`. The provider's prefix is therefore
`openai/responses/` — `openai/` picks LiteLLM's OpenAI transport and
`responses/` flips it onto `POST {api_base}/responses`, with LiteLLM's
`completion_extras/litellm_responses_transformation` bridge translating
messages, tools and streaming in both directions. **The engine is untouched:
it still calls `litellm.completion()`.** `_normalize_subscription_model`
repairs a bare `gpt-5.5` or a copied `openai/gpt-5.5`, because the resulting
404 explains nothing.

**The credential is an OAuth token owned by another tool.** There is no
second login: `codex login` writes `~/.codex/auth.json` (honouring
`CODEX_HOME`) and `openai_oauth.py` reads, refreshes and *writes back* that
same file. Writing back is mandatory, not tidiness — OpenAI rotates refresh
tokens, so a refresh kept in memory would leave a dead one on disk and
silently log the user out of the Codex CLI. Refreshes are atomic
(`os.replace` onto a file created 0600), taken under an advisory `flock`
shared with any other writer, and re-read inside the lock so a concurrent
`codex` run that already refreshed is used instead of burning a second
token. `_apply_chatgpt_subscription` in `llm.py` resolves the token on every
request and is the single place that overrides `api_key`/`api_base` for all
four `get_litellm_params*` builders.

**The limits are not the API's limits.** litellm's cost map says `gpt-5.5`
takes 1 050 000 input tokens; on the subscription it takes 272 000.
`codex_catalog.py` reads the catalog the Codex CLI caches at
`~/.codex/models_cache.json` — never writes it — for the model list, the
per-model context window (scaled by `effective_context_window_percent`, the
share left for input) and the reasoning levels each model accepts, so
`THINKING_BUDGET=ultra` can reach `xhigh` where it exists instead of being
flattened to `high`. Trusting the cost map would hand the loop ~800 000
tokens of headroom that do not exist.

Two smaller traps, both covered by tests: `is_native` must stay False or the
`openai/` prefix routes the request to api.openai.com where the OAuth token
is not a valid key; and the loop's pinned `temperature=0.2` has to be
dropped, since GPT-5.x reasoning models reject it.

### DB (`db/service.py`)

SQLite with tables: `projects`, `tasks`, `findings`, `artifacts`. All access goes through `execute_with_retry()` with exponential backoff for WAL contention.

### Flows (`flows/event_listeners.py`)

Currently a stub — `EventBus` with no-op `emit()`. Designed for future WebSocket/external event support.

### Embeddings (`tools/base/dedup.py`, `tools/base/mnn_embedder.py`)

`all-MiniLM-L6-v2` is used for finding dedup (threshold 0.82), ContextRank fuzzy symbol search, and predictive/historical scoring. Two backends:

- **Default**: ChromaDB's bundled ONNX Runtime — no setup, ~115 ms per query on CPU.
- **Optional MNN**: install-and-forget via `uv sync --extra mnn`. On first call the embedder auto-converts ChromaDB's cached ONNX model (~30 s, logged), caches it under `~/.infinidev/models/minilm.mnn`, and takes over. Same 384-dim vectors (cosine 1.0000 vs ONNX — stored BLOBs compatible), ~11 ms per query, CPU-only in the current pip wheel. Auto-patches MNN's `.so` for hardened kernels. `INFINIDEV_MNN_MODEL_PATH` is an optional override for the model path, not a required toggle.

`dedup._get_embed_fn()` probes MNN first when the env var is set; otherwise returns the ChromaDB default. Never swap embedding *models* (dim mismatch corrupts stored BLOBs) — only swap *runtimes*.

## Key Constraints

- Loop limits: max 50 iterations, max 4 tool calls per step, max 200 total tool calls per task
- History window: configurable via `LOOP_HISTORY_WINDOW` (0 = keep all summaries)
- Semantic dedup threshold: 0.82 cosine similarity for findings
- File size limit: 5MB for read operations
- Git branches must follow `task-{task_id}-<slug>` naming
