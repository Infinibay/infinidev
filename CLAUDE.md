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

The `ChatAgent` owns a short (~5 iteration) read-only LLM loop with
the `respond`/`escalate` terminator tools. The `AnalystPlanner` owns a
budgeted loop with the `emit_plan` terminator. Neither uses the
LoopEngine — they call litellm directly; the LoopEngine is reserved
for the developer's heavy plan-execute-summarize loop.

### Loop Engine (`engine/loop/engine.py`)

The developer stage is a **plan-execute-summarize** cycle, not a ReAct loop:

1. **Plan** — either seeded upfront from `initial_plan` (chat-agent-first path, steps marked `user_approved=True`) or bootstrapped by the LLM via `add_step` calls (legacy PhaseEngine path).
2. **Execute** — One step at a time, up to 4 tool calls per step.
3. **Summarize** — LLM produces a ~50-token summary; raw tool output is discarded.
4. **Repeat** — Prompt is rebuilt from scratch each iteration using only compact summaries.

The LLM signals step completion via a `step_complete` tool call with `status` (continue/done/blocked), `summary`, and optional plan modifications (add/modify/remove steps). `user_approved` steps are protected — `apply_operations` rejects remove/modify on them so the LLM cannot rewrite an analyst-produced plan mid-execution.

**Dual tool-calling modes:** The engine auto-detects whether the LLM supports native function calling (FC mode) or falls back to parsing tool calls from text JSON (manual mode). Detection happens at startup via `config/model_capabilities.py`.

### Prompt Construction

Every iteration builds an XML-structured prompt: `<task>`, `<plan-overview>` (stable prose, set once by the planner), `<plan>` (step list), `<previous-actions>`, `<current-action>` (active step's full `detail`), `<next-actions>`, `<expected-output>`. The protocol rules are in `prompts/shared.py` as `LOOP_PROTOCOL`. Per-step `detail` renders ONLY for the active step to keep context compact.

### Tools (`tools/`)

All tools inherit from `InfinibayBaseTool` (extends CrewAI's `BaseTool`). Tools are bound to agents via `bind_tools_to_agent()` and resolve context (project_id, task_id, workspace_path) from a process-global dict.

Categories:
- **file**: `read_file`, `partial_read`, `create_file`, `replace_lines`, `list_directory`, `code_search`, `glob`
- **code_intel**: `get_symbol_code`, `list_symbols`, `search_symbols`, `find_references`, `edit_symbol`, `add_symbol`, `remove_symbol`, `project_structure`
- **git**: `git_branch`, `git_commit`, `git_diff`, `git_status`
- **shell**: `execute_command`, `code_interpreter`
- **knowledge**: `record_finding`, `read_findings`, `search_findings` (with semantic dedup)
- **meta**: `help` (dynamic tool documentation)
- **chat_agent** (tier-exclusive): `respond`, `escalate` — terminators for the chat agent's loop; never bound to the developer.
- **planner** (tier-exclusive): `emit_plan` — the planner's single-shot terminator that produces the `Plan` artifact.

The base class exposes `is_read_only: bool = False`. The 18 pure-read tools (file reads, code-intel lookups, git diff/status, findings reads) override it to `True`. `get_tools_for_role("chat_agent")` and `get_tools_for_role("planner")` filter the full toolset by this attribute — the schema passed to LiteLLM is the security boundary, not prompt rules.

Key tool design: `read_file` auto-indexes files via tree-sitter for code intelligence. `replace_lines` uses deterministic line-range replacement (no text matching). Symbol tools (`edit_symbol`, `add_symbol`, `remove_symbol`) use the code index to locate symbols by qualified name.

Tool schemas are validated at runtime — hallucinated parameters are rejected before execution. Old tool names (`edit_method`, `add_method`, `remove_method`, `write_file`, `find_definition`) are aliased to new names in `engine/loop_tools.py`.

### Agent (`agents/base.py`)

`InfinidevAgent` holds role metadata, binds tools based on role, and manages execution context. The CLI creates one agent with role="developer" per user instruction.

### Config (`config/`)

- `settings.py` — All settings use `INFINIBAY_` env var prefix. Key: `LLM_MODEL` (LiteLLM format like `ollama/qwen2.5-coder:7b`), `LLM_BASE_URL`, `SANDBOX_ENABLED`, loop limits.
- `llm.py` — `get_litellm_params()` builds the dict for `litellm.completion()`.
- `model_capabilities.py` — Runtime probing of FC support, JSON mode, schema sanitization needs.

### DB (`db/service.py`)

SQLite with tables: `projects`, `tasks`, `findings`, `artifacts`. All access goes through `execute_with_retry()` with exponential backoff for WAL contention.

### Flows (`flows/event_listeners.py`)

Currently a stub — `EventBus` with no-op `emit()`. Designed for future WebSocket/external event support.

## Key Constraints

- Loop limits: max 50 iterations, max 4 tool calls per step, max 200 total tool calls per task
- History window: configurable via `LOOP_HISTORY_WINDOW` (0 = keep all summaries)
- Semantic dedup threshold: 0.82 cosine similarity for findings
- File size limit: 5MB for read operations
- Git branches must follow `task-{task_id}-<slug>` naming
