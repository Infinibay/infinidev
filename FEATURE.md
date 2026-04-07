# Infinidev — Feature Reference

This document is a complete map of what Infinidev does today. It is
**not** a tutorial or a getting-started guide (see `README.md` for
that) — it is the reference an experienced operator wants to have on
hand when planning a new run, debugging a stuck task, or extending
the system. Each section includes the *why* behind a feature, not
just the *what*, because most of the design decisions in this
codebase are motivated by behaviour we observed in real runs against
real models.

The features are organised by layer, from the outermost (the user
interface) to the innermost (the code intelligence index). If you
are looking for "what changed recently", read `git log` — this file
is a snapshot of the *current* state, not a history.

> **Audience**: developers extending Infinidev, or operators running
> it day-to-day. If you are a model running inside Infinidev, the
> tool descriptions are richer and more focused than this file —
> use `help` instead.

---

## Table of contents

1. [What Infinidev is](#what-infinidev-is)
2. [User interfaces](#user-interfaces)
3. [The orchestration pipeline](#the-orchestration-pipeline)
4. [The loop engine (plan-execute-summarize)](#the-loop-engine)
5. [Tools — the model's hands](#tools)
6. [Code intelligence — the model's eyes](#code-intelligence)
7. [The guidance system — coaching for stuck models](#the-guidance-system)
8. [The phase engine — for deeper reasoning](#the-phase-engine)
9. [The tree engine — for explore / brainstorm](#the-tree-engine)
10. [Multi-provider LLM support](#multi-provider-llm-support)
11. [Performance instrumentation](#performance-instrumentation)
12. [Database & persistence](#database--persistence)
13. [Settings, env vars, and operational tooling](#settings-env-vars-and-operational-tooling)
14. [Things that exist on purpose to NOT do](#things-that-exist-on-purpose-to-not-do)

---

## What Infinidev is

Infinidev is a terminal-based AI programming agent. Its design centre
is a single goal: **make local open-weight models on consumer hardware
useful for real software-engineering tasks**. Every architectural
decision flows from this constraint.

It is the CLI evolution of the original Infinibay multi-agent system.
The web prototype lives in `research_copy/` and is no longer part of
the active product.

The two things that make Infinidev different from a generic
"LLM-with-tools" wrapper:

  1. **It assumes the model is small and easily confused**, and takes
     active steps to compensate — proactive guidance, structured
     skeletons of large files, auto-extracted test failures,
     duplicate-code detection, plan validation. Big-model wrappers
     trust the model; Infinidev coaches it.
  2. **The pipeline is unified across all entry points** — the TUI,
     the classic CLI, and the one-shot `--prompt` mode all run the
     same `engine.orchestration.run_task`. Improvements land in one
     place and reach every user.

---

## User interfaces

Three entry points, all backed by the same orchestration pipeline:

### TUI mode (default)

```bash
infinidev
```

Full prompt_toolkit-based terminal UI with chat history, status bar,
context-token meter, file-diff widgets, an autocomplete dropdown for
commands, and a STEPS panel that follows along as the agent executes
its plan.

The TUI is the recommended mode for interactive sessions. It supports:

  * **Question-answer flow** — the analyst can ask clarifying
    questions and you reply inline.
  * **Spec confirmation** — for the develop flow, the analyst shows
    you a specification (summary, requirements, hidden requirements,
    assumptions, out-of-scope) and asks "proceed? y/n/feedback".
  * **/plan mode** — review the generated plan before execution and
    approve, cancel, or feed back changes.
  * **Live STEPS panel** — watch which step is active, which are
    done (`v`), and which are upcoming (`o`).
  * **File diff viewer** — every file the agent modifies is captured
    and shown as a unified diff in a side panel.
  * **Permission prompts** — when a tool requires explicit approval
    (sensitive shell commands, writes outside the workspace), the UI
    pauses and asks.
  * **Message injection** — type while the agent is running and your
    message is queued; the agent sees it on the next step.

### Classic CLI mode

```bash
infinidev --classic
# or  infinidev --no-tui
```

A traditional read-eval-print loop on top of the same engine.
Everything the TUI can do, this can do — same analysis flow, same
question loop, same review phase. The output is `click`-coloured but
linear; no panels, no live updates. Use this when:

  * You are SSHed into a host without TUI capabilities.
  * You want a clean transcript for later review.
  * Your terminal is too small for the TUI.

The classic mode also supports the `/init`, `/explore`, `/brainstorm`,
`/refactor`, `/think`, and `/settings` commands described below.

### One-shot `--prompt` mode

```bash
infinidev --classic --prompt "Refactor the auth middleware to use JWT"
infinidev --classic --prompt "/explore <problem>"
```

Non-interactive: the prompt is read from the command line, the agent
runs to completion, and the result is printed. No question loop, no
spec confirmation — those steps are skipped because there is no human
to answer.

Special handling for **imperative tasks**: if the prompt starts with
an action verb (`create`, `add`, `fix`, `implement`, `refactor`,
`rename`, `delete`, `update`, ...), the analysis phase is bypassed
entirely. This was added because the analyst was wrapping imperative
tasks in a "do NOT write files, only analyze" envelope, leaving the
loop in an impossible state.

`--prompt` mode is the right choice for benchmarking, scripting,
batch processing, or any case where the human is offline.

### Slash commands

| Command | Effect |
|---|---|
| `/help` | List available commands. |
| `/models list` | List models available in the configured provider (Ollama). |
| `/models set <name>` | Switch the active model for the next task. |
| `/settings`, `/settings <key>`, `/settings <key> <val>` | Read/edit Infinidev settings (persisted to `~/.infinidev/settings.json`). |
| `/settings reset` / `export` / `import` | Settings file management. |
| `/init` | Run the document flow: explore the project, write a `CLAUDE.md`-style overview. |
| `/explore <problem>` | Branch out from a problem statement, find facts, return a synthesis. |
| `/brainstorm <problem>` | Like `/explore` but with creative-ideation prompting. |
| `/refactor [scope]` | Refactor code (modularize, clean up, restructure) — auto-scoped if no argument. |
| `/think` | Switch the next task to the phase engine (ANALYZE → PLAN → EXECUTE). |
| `/plan <task>` | (TUI only) Run the phase engine with an interactive plan-review checkpoint. |
| `/exit`, `/quit` | Exit. |

---

## The orchestration pipeline

The single source of truth for what happens to a user request, from
"the user typed something" to "we have a result to display".

Located at `engine/orchestration/`. Every UI entry point is a thin
adapter that constructs an `OrchestrationHooks` implementation and
calls `run_task(...)`. The pipeline imports nothing from `click`,
`prompt_toolkit`, `threading`, or any UI module.

### Pipeline phases

1. **Analysis**. The analyst classifies the task (develop / document /
   research / explore / brainstorm / sysadmin / done), optionally asks
   clarifying questions, and produces a specification. For the
   develop flow it also asks the user to confirm the spec before
   executing. Can be skipped per-call (`skip_analysis=True`) or
   globally (`ANALYSIS_ENABLED=False`).

2. **Gather**. Collects relevant codebase context (the
   `infinidev.gather` package). Renders a "brief" of files, symbols,
   and notes that get prepended to the task prompt. Soft-fails: if
   gather raises, the pipeline continues with the original task
   prompt and reports the failure.

3. **Execute**. Dispatches to one of three engines:
   * **LoopEngine** for normal tasks (the most common path).
   * **TreeEngine** for the explore and brainstorm flows.
   * **PhaseEngine** when the user opts in via `--think` or `/plan`.

4. **Review**. Runs the review-rework loop if review is enabled and
   the task produced file changes. The reviewer can approve, request
   changes (the agent re-runs with feedback), or fail verification
   (the agent re-runs to fix the broken tests).

### `OrchestrationHooks` Protocol

The only contract between pipeline and UI. Every UI implements:

```python
class OrchestrationHooks(Protocol):
    def on_phase(self, phase: str) -> None: ...
    def on_status(self, level: str, msg: str) -> None: ...
    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None: ...
    def ask_user(self, prompt: str, kind: str = "text") -> str | None: ...
    def on_step_start(self, step_num, total, all_steps, completed) -> None: ...
    def on_file_change(self, path: str) -> None: ...
```

`@runtime_checkable` so it works with duck-typed implementations.
`ask_user` returns `None` for non-interactive callers — branches that
receive `None` MUST proceed with sensible defaults rather than
failing.

### Default hook implementations

| Class | Where | Behaviour |
|---|---|---|
| `NoOpHooks` | tests, base class | Drops every call. `ask_user` returns `None`. |
| `ClickHooks` | classic CLI interactive | Coloured output via `click.echo`, blocks on `input()` or `prompt_toolkit.PromptSession`. |
| `NonInteractiveHooks` | `--prompt` mode | Inherits `ClickHooks`, overrides `ask_user` to never block (always returns `None`). |
| `TUIHooks` | TUI (`ui/hooks_tui.py`) | Marshals to the prompt_toolkit event loop, blocks on `threading.Event` for user input, updates panels via `app.invalidate()`. |

---

## The loop engine

`engine/loop/engine.py` — the heart of the system. NOT a ReAct loop;
it is a **plan-execute-summarize** cycle:

1. **Plan**. The model produces 2-3 initial steps from the user
   instruction.
2. **Execute**. The model works on one step at a time, with up to 4
   tool calls per step (configurable).
3. **Summarize**. After each step, a smaller LLM call produces a
   ~50-token summary; the raw tool output is then discarded.
4. **Repeat**. The next iteration's prompt is rebuilt from scratch
   using only the compact summaries — the working set never grows.

The model signals step completion via a `step_complete` tool call
with `status` (continue / done / blocked), `summary`, and optional
plan modifications (add / modify / remove steps).

### Dual tool-calling modes

The engine auto-detects whether the LLM supports native function
calling (FC mode) or falls back to parsing tool calls from text JSON
(manual mode). Detection happens at startup via
`config/model_capabilities.py`. The two modes share all the same
guidance, instrumentation, and tool dispatch — the only difference
is how tool calls cross the wire.

### Prompt construction

Every iteration builds an XML-structured prompt with these sections:

  * `<task>` — the user's request.
  * `<plan>` — the current plan with `[done]`, `[active]`, `[pending]`
    markers.
  * `<previous-actions>` — compact summaries of the last N steps.
  * `<current-action>` — the active step with its expected output.
  * `<next-actions>` — what's coming up after the current step.
  * `<expected-output>` — the high-level success criterion.
  * `<context-budget>` — how many tokens have been used vs the max.
  * `<behavior-summary>` — when present, coaching from the behavior
    tracker.

The protocol rules are in `prompts/shared.py` as `LOOP_PROTOCOL`.

### Loop limits (configurable)

| Setting | Default | Purpose |
|---|---|---|
| `LOOP_MAX_ITERATIONS` | 50 | Outer iteration cap. Stops infinite plan-replan loops. |
| `LOOP_MAX_TOOL_CALLS_PER_ACTION` | 4 (small) / 8 (large) | Per-step budget. Forces the model to call `step_complete`. |
| `LOOP_MAX_TOTAL_TOOL_CALLS` | 200 | Total per-task cap. Hard ceiling against runaway. |
| `LOOP_HISTORY_WINDOW` | 0 (= keep all) | How many step summaries to retain. |
| `LOOP_STEP_NUDGE_THRESHOLD` | 6 | When to inject a "you've used N tool calls, finish or replan" nudge. |

### Auto-extracted test failures (A4)

When `execute_command` finishes and the command was a test runner
(pytest, jest, cargo, go, mocha, node:test, rspec — auto-detected),
the engine parses the output through the per-runner parser package
and **appends a structured failures block** to the tool result. The
model gets the parsed list inline with the raw output, so it never
has to discover `tail_test_output` on its own:

```
[auto-extracted structured_failures (3 total):]
[
  {"runner":"pytest","test_name":"test_create_table","file":"...","line":42,
   "error_type":"KeyError","message":"..."},
  ...
]
```

Capped at 8 failures per response to keep the prompt small.

### Behavior tracker

`engine/behavior/` — a separate scoring system that watches what the
model does (and doesn't do) over multiple iterations and injects
short corrections into the next prompt:

  * "WARNING: You have read multiple files but saved no notes."
  * "STOP DOING: You have called the same tool 4 times with the same
    arguments."
  * "Behavior score: -2"

Independent from the guidance system. Where guidance fires on
specific failure patterns and produces detailed how-to advice, the
behavior tracker fires continuously and produces short nudges.

---

## Tools

All tools inherit from `InfinibayBaseTool` (which wraps CrewAI's
`BaseTool`) and live in `tools/`. Each tool has:

  * A schema (Pydantic model) — invalid arguments are rejected at
    runtime so hallucinated parameters never reach the implementation.
  * A description — surfaced to the model in the system prompt and
    via the `help` tool.
  * Auto-context binding — `project_id`, `agent_id`, `session_id`,
    `workspace_path` are resolved from a process-global thread-local
    so tools never need them as explicit parameters.

### File tools

| Tool | What it does |
|---|---|
| `read_file` | Read a file with line numbers. Auto-indexes for code intelligence. **For files >800 lines, returns a structured tree-sitter skeleton instead of the raw content** (see [Code intelligence](#code-intelligence)). |
| `partial_read` | Read a specific line range. Same as `read_file(start_line, end_line)` but explicit. |
| `create_file` | Create a new file. Pre-write syntax check (Python, JS/TS) refuses writes that would leave the file in a broken state. |
| `replace_lines` | Deterministic line-range replacement. NO text matching — you give exact line numbers. |
| `add_content_after_line` / `add_content_before_line` | Insert without replacing. |
| `apply_patch` | Apply a unified diff. |
| `multi_edit_file` | Multiple replace_lines operations on the same file in one call. |
| `list_directory` | List files in a directory with sizes and types. |
| `code_search` | ripgrep-backed full-text search across the project. |
| `glob` | Glob pattern matching for file paths. |

### Pre-write safety

`create_file`, `replace_lines`, `edit_file`, `multi_edit_file`,
`apply_patch` all run two checks before writing to disk:

  1. **Syntax check** (`code_intel/syntax_check.py`). Tree-sitter
     parses the proposed content; any ERROR or missing-token nodes
     cause the write to be rejected with a line-numbered error
     message. Supports Python and JavaScript today.
  2. **Silent symbol deletion detection**. Compares the set of
     top-level symbols in the old vs new content. If the edit
     accidentally drops a function or class the model probably
     didn't mean to delete, the write is rejected with a warning
     listing the lost symbols.

Both checks have a per-task latency budget under 20ms (instrumented
via the static analysis timer).

### Code intelligence tools

| Tool | What it does |
|---|---|
| `list_symbols` | List all symbols in a file or matching a pattern. |
| `search_symbols` | Fuzzy search across symbol names (FTS5 + prefix matching). |
| `search_by_docstring` | **Intent-based** search: find symbols by what they DO, not what they're CALLED. Uses BM25 ranking over docstrings + signatures. |
| `find_similar_methods` | **Body-based** search: find methods elsewhere in the project that look like a given method. Catches copy-paste duplicates and near-duplicates via normalized-token Jaccard. |
| `find_references` | Find every usage of a symbol across the project. |
| `find_definition` | Jump to where a symbol is defined. |
| `get_symbol_code` | Read the full source of one symbol by qualified name. Cheaper than `partial_read` when you only need one method. |
| `project_structure` | Tree view of the project's directory layout. |
| `edit_symbol` (alias `edit_method`) | Replace a symbol's body by qualified name. No line numbers needed. |
| `add_symbol` (alias `add_method`) | Insert a new symbol into a file. |
| `remove_symbol` (alias `remove_method`) | Delete a symbol cleanly. |
| `rename_symbol` | Rename across all references. |
| `move_symbol` | Move a symbol to a different file, fixing imports. |
| `analyze_code` | Static-analysis report for a file. |

The "find existing code" trilogy — `search_symbols`,
`search_by_docstring`, `find_similar_methods` — is one of the most
important features for refactoring tasks. They cover the three axes
of "I want to reuse code that already exists":

  * **search_symbols** → I know what it's CALLED.
  * **search_by_docstring** → I know what it DOES (in words).
  * **find_similar_methods** → I know what it LOOKS LIKE (in code).

### Git tools

| Tool | What it does |
|---|---|
| `git_branch` | Create / list / delete branches. Naming convention: `task-{task_id}-<slug>`. |
| `git_commit` | Stage and commit. |
| `git_diff` | Diff working tree, index, or arbitrary refs. |
| `git_status` | Working tree status. |
| `git_push` | Push to remote (requires explicit user permission). |

### Shell tools

| Tool | What it does |
|---|---|
| `execute_command` | Run an arbitrary shell command in the workspace. Captures stdout/stderr, exit code, runtime. Test commands trigger the auto-extracted failures hook. |
| `code_interpreter` | Run a Python snippet in a sandboxed interpreter. |

### Knowledge tools

| Tool | What it does |
|---|---|
| `record_finding` | Persist a key fact the agent discovered (with semantic dedup at threshold 0.82). |
| `read_findings` | Retrieve all findings for the current session. |
| `search_findings` | Vector + FTS hybrid search over findings. |
| `validate_finding` / `reject_finding` / `update_finding` / `delete_finding` | Lifecycle management. |
| `write_report` / `read_report` / `delete_report` | Long-form structured outputs. |
| `search_knowledge` | Cross-session knowledge search. |
| `summarize_findings` | Compact a finding cluster into one entry. |

### Meta tools

| Tool | What it does |
|---|---|
| `help` | Dynamic tool documentation — the model can ask "what does X do?" and receive the full schema + description. |
| `add_step` / `modify_step` / `remove_step` | Plan management. The model evolves its plan as it learns more. |
| `declare_test_command` | Tell the system "the test command for this project is X" so the test detection works for non-standard runners. |
| `tail_test_output` | Re-read structured failures from the last test run without re-running the tests. Three modes: `tail`, `failures`, `structured`. |

### Web tools

| Tool | What it does |
|---|---|
| `web_search` | Query a search engine (rate-limited via `WEB_RPM_LIMIT`). |
| `web_fetch` | Fetch a URL with caching (`WEB_CACHE_TTL_SECONDS`) and robots.txt enforcement. |
| `code_search_web` | Search across public code-hosting platforms. |

### Small-model tool subset

A curated subset of ~22 tools is exposed to small models (<25 B
parameters). These are the tools whose schemas are simple enough that
small models can use them reliably. The full list includes:

  * File I/O (read, create, replace_lines, add_content_*, list_dir,
    code_search, glob)
  * Git (commit, diff, status)
  * Shell (execute_command)
  * Knowledge (record_finding, search_findings)
  * Code intelligence (search_symbols, search_by_docstring,
    find_similar_methods, find_references, get_symbol_code,
    edit_symbol)
  * Plan management (add_step, modify_step, remove_step)
  * Project introspection (declare_test_command, tail_test_output)

Switching is automatic — `LLM_MODEL` is matched against a
small-model registry at startup.

---

## Code intelligence

The system that gives the model "eyes" over the codebase. Backed by
tree-sitter and a SQLite-based symbol index.

### Supported languages

The `extract_file_skeleton` extractor and the indexer support these
languages out of the box, all via tree-sitter packages already in the
venv:

| Language | Extension(s) | Skeleton support | Index support |
|---|---|---|---|
| Python | `.py`, `.pyi` | ✓ | ✓ |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | ✓ | ✓ |
| TypeScript | `.ts` | ✓ | ✓ |
| TSX | `.tsx` | ✓ | ✓ |
| Go | `.go` | ✓ | ✓ |
| Rust | `.rs` | ✓ | ✓ |
| Java | `.java` | ✓ | ✓ |
| C | `.c`, `.h` | ✓ | ✓ |
| C++ | `.cc`, `.cpp`, `.cxx`, `.hpp`, `.hh` | ✓ | – |
| Ruby | `.rb` | ✓ | – |
| C# | `.cs` | ✓ | – |
| PHP | `.php` | ✓ | – |
| Kotlin | `.kt`, `.kts` | ✓ | – |
| Bash / Shell | `.sh`, `.bash`, `.zsh` | ✓ | – |

For unsupported languages, `read_file` falls back to a head+tail
preview (60 + 30 lines) with the same "use partial_read" hint at the
end.

### Large-file skeleton mode

When `read_file` is called on a file larger than 800 lines (default,
configurable) without an explicit line range, the tool returns a
structured skeleton instead of the raw content:

```
⚠ FILE TOO LARGE TO READ IN FULL — returning structured skeleton.
  file:     app/services/VirtioSocketWatcherService.ts
  size:     4245 lines, 160611 bytes (typescript)
  symbols:  41 classes, 1 functions, 72 methods, 5 globals, 13 imports

── imports (13) ──
  L10-10  import prisma from '@utils/database'
  ...

── classes (41) ──
  L42-45  class BaseMessage  — Message types from InfiniService
  L47-51  class ErrorMessage
  L78-986  class VirtioSocketWatcherService
      L142-198  .start  — Begins polling sockets and dispatching events
      L772-986  .connectToVm  — Establishes VirtIO socket connection
      ...

── How to read this file ──
  This file is too large to load in full. To inspect specific parts,
  use one of these tools:

  • partial_read(file_path=..., start_line=N, end_line=M)
  • get_symbol_code(file_path=..., name='ClassName.methodName')
  • search_symbols(query=...)
```

The skeleton typically reduces a 4000-line file from ~40k tokens to
~2.4k tokens (94% reduction) while preserving navigation information.
The "How to read this file" hint at the end is critical — small
models don't discover specialized tools on their own, so the reply
itself names the next tools to call.

The walker honours four invariants:

  1. **Globals are top-level only** — function-local `const`
     declarations never get mis-reported as module globals.
  2. **Methods are class-body-only** — same node type at module level
     vs inside a class is correctly classified by position.
  3. **Passthrough nodes** (`export_statement`, `namespace_declaration`,
     `mod_item`) are descended into without creating an entry.
  4. **ERROR nodes are skipped** — tree-sitter's error tolerance is
     great for parsing but terrible for skeleton extraction.

Adding a new language is one config dict entry in
`code_intel/syntax_check.py::_LANGUAGE_SKELETON_CONFIG`, no new
walker code.

### Symbol index (`ci_symbols`)

Every file the agent reads or edits is indexed in the background.
The indexer extracts symbols (classes, functions, methods, variables,
imports) and stores them in `ci_symbols` along with their:

  * Qualified name (`ClassName.methodName`)
  * Kind (`function`, `method`, `class`, `variable`, `constant`,
    `import`)
  * Line range (`line_start`, `line_end`)
  * Signature
  * Docstring
  * Type annotations
  * Visibility (public / private / protected)
  * `is_async`, `is_static`, `is_abstract` flags
  * Language

Indexes:
  * `idx_ci_symbols_name` — fast lookup by bare name
  * `idx_ci_symbols_qualified` — exact qualified name
  * `idx_ci_symbols_kind` — filter by kind + name
  * `idx_ci_symbols_file` — all symbols in a file
  * `idx_ci_symbols_parent` — methods of a class

Plus an FTS5 virtual table `ci_symbols_fts` indexing `name`,
`qualified_name`, `signature`, and `docstring` for full-text
queries with BM25 ranking. Triggers keep the FTS in sync on every
INSERT / DELETE / UPDATE.

### Method body fingerprints (`ci_method_bodies`)

Per-method normalized fingerprints for fuzzy similarity search across
the entire project. Populated by the indexer immediately after
`store_file_symbols` runs, so it stays in sync without a separate
background pass.

For each function/method we store:

  * **body_hash** — sha256 of the normalized body (16 hex chars).
    Catches exact copy-paste even when whitespace, comments, or
    local identifiers have been changed.
  * **body_norm** — space-separated normalized tokens. Used for
    Jaccard similarity over token sets.
  * **body_size** — line count (after stripping comments and blanks).
    Used as a cheap pre-filter.

The normalization pipeline:

```
source body
  → strip comments (// /* */ # — language-aware)
  → strip blank lines
  → replace user identifiers with V1, V2, V3 placeholders
    (keywords kept as-is)
  → collapse whitespace → lowercase
```

Methods smaller than 6 normalized lines are skipped (trivial getters
always look like every other trivial getter).

The `find_similar_methods` tool runs a size-bounded scan (±40% line
count) and ranks candidates by Jaccard, with exact `body_hash`
matches surfaced first at similarity=1.0. Real-world validation on
the infinibay/backend repo: 1235 method fingerprints indexed in
7.6s, 44 exact-duplicate methods detected on the first pass.

### Background indexing

`code_intel/background_indexer.py` is a process-global registry for
the `IndexQueue` worker thread. The CLI starts one at boot and tools
push paths to it via `enqueue_or_sync(project_id, path)`:

  * If the queue is running and matches the project, the path is
    enqueued in microseconds and the worker thread does the actual
    parse off the hot path.
  * If no queue exists (tests, scripts, isolated tool calls), it
    falls back to a synchronous `ensure_indexed` call.

Result: the `file_indexing` latency category dropped from ~478ms
total per task to ~0.18ms (a 2650x reduction) without changing tool
behaviour.

### Smart reindexing

`smart_index.ensure_indexed` skips reindexing when the file's content
hash hasn't changed. The hash comparison happens in microseconds, so
re-reading the same file 100 times costs only the cost of the first
parse.

---

## The guidance system

The system that watches the loop in real time and injects pre-baked
how-to advice when specific failure patterns are detected. Located
at `engine/guidance/`. Designed to coach small models toward the
right action without ever making an LLM call itself — every detector
runs in microseconds against the message history and loop state.

Each detector is a function that returns `True` when its trigger
condition is met. The dispatcher walks `_DETECTORS` after each loop
iteration, fires the first one that matches, and queues a guidance
entry to be rendered on the next prompt build.

### Detectors

| Key | Fires when | Purpose |
|---|---|---|
| `stuck_on_planning` | Manually surfaced for now (auto-fire detector pending). | "You've planned long enough, start opening files." |
| `stuck_on_edit` | Repeated edit failures on the same file. | Suggests using `replace_lines` with explicit line numbers. |
| `stuck_on_tests` | Test command run 3+ times with similar exit codes. | Points at `tail_test_output` for structured failures. |
| `same_test_output_loop` | Identical test fingerprint 3+ runs in a row. | "Re-running the same command won't fix it." |
| `reread_loop` | Same file read 3+ times without an edit between. | "You already have this file in context — make a decision." |
| `unknown_tool` | Model called a non-existent tool name. | Lists the actual tool names available. |
| `vague_steps` | Plan steps with vague titles like "explore the project". | Asks for concrete files / symbols. |
| `text_only_iters` | 2+ iterations with no tool calls (just thinking). | "Stop narrating, start acting." |
| `stuck_on_search` | Code search produced 50+ hits and the model is paging. | Suggests narrowing the query or using `search_by_docstring`. |
| `malformed_tool_call` | Model emitted `{"tool_calls": [...]}` as text in `content`. | Shows the correct FC envelope. |
| `regression_after_edit` | A test that was passing now fails after an edit. | "Your last edit broke a previously passing test." |
| `first_test_run` | First test command of the task. | **Proactive** — introduces `tail_test_output` BEFORE the model gets stuck. |
| `duplicate_steps` | Two steps with very similar wording in the plan. | "You're replanning the same thing." |

### Guidance library

`engine/guidance/library.py` contains the `GuidanceEntry` definitions
— each one is a structured (key, message, severity) tuple. The
dispatcher in `hooks.py` queues these entries via
`maybe_queue_guidance` and renders them via `drain_pending_guidance`
into the next prompt's `<behavior-summary>` block.

Guidance is **rate-limited** per task via `LOOP_GUIDANCE_MAX_PER_TASK`
(default 12) so the model never gets flooded.

### Test runner detection

`engine/guidance/test_runners.py` provides `is_test_command`,
`normalize_test_command`, and `test_outcome_fingerprint`. Test
detection covers 16+ runners via the per-parser flag tables (each
`TestParser` subclass owns its own command tokens and flag list to
avoid the substring trap where `cargo test` would naively match
`go test`).

### Per-runner test parsers

`engine/test_parsers/` — one class per runner:

| Parser | Runner | Detects |
|---|---|---|
| `PytestParser` | pytest, py.test | Python test failures with file:line + traceback |
| `JestParser` | jest, vitest | JS/TS test failures (jest-style and vitest-style) |
| `MochaParser` | mocha | JS test failures with stack traces |
| `NodeTestParser` | node:test | Node 20+ TAP-format output |
| `GoTestParser` | go test | Go test failures with `--- FAIL:` markers |
| `CargoTestParser` | cargo test | Rust test failures + panic locations |
| `RSpecParser` | rspec | Ruby test failures |

Adding a new runner is one new file with a `TestParser` subclass +
one entry in `_PARSERS`. No other module needs changes — the
dispatch loop picks it up automatically.

---

## The phase engine

`engine/phases/phase_engine.py` — an alternative execution path that
explicitly separates **CLASSIFY → INVESTIGATE → PLAN → EXECUTE**
phases. Used by `--think` and the TUI's `/plan` command.

### When to use it

The phase engine is heavier than the loop engine (it makes more LLM
calls per task) but produces better results on complex tasks where
the model needs to investigate before planning. Use it when:

  * The task has multiple unknowns ("how does X integrate with Y?").
  * The model needs to read 5+ files before knowing what to do.
  * You want a human-reviewable plan before any code is touched.

Skip it for simple tasks ("create a file with this content") where
the loop engine is faster.

### Phases

1. **CLASSIFY**. Determines task type (feature / bug / refactor / etc)
   and depth (minimal / standard / deep). Skipped for small models
   (they get it wrong) and replaced with sensible defaults.

2. **INVESTIGATE**. Iterative question-answer loop where the model
   asks itself what it needs to know and answers by reading the
   codebase. Bounded by `investigate_max_tool_calls` from the depth
   config.

3. **PLAN**. Generates a structured plan with files, steps, and
   expected outcomes. Has a re-plan loop driven by test progress.

4. **EXECUTE**. Runs the plan step by step via a normal LoopEngine
   instance. Each step gets its own bounded loop run.

### `execute_with_plan_review`

The public API for the `/plan` flow. Replaces the four private
function calls the TUI worker used to make. Takes a callback
(`on_plan_ready(plan_steps)`) that the UI uses to show the plan to
the user and collect their verdict (`approve` / `cancel` / `feedback`).
On feedback the plan is regenerated with the user's notes appended;
on approve the executor runs.

### Test checkpoint integration

Each phase engine run holds a `TestCheckpoint` that runs the test
command after each step and tracks `passed/total`. When tests are
green, the loop exits early; when they're red, the re-plan loop
fires.

---

## The tree engine

`engine/tree.py` — an exploration-oriented engine for `/explore` and
`/brainstorm`. Maintains a **tree of facts, questions, blockers, and
ideas** and uses a ranking function to pick the next node to expand.

### When to use it

  * **`/explore <problem>`** — depth-first investigation of an
    unfamiliar topic. The tree starts with the problem and grows
    sub-questions and facts as the model discovers them.
  * **`/brainstorm <problem>`** — creative ideation with forced
    perspective shifts. Same engine, different prompting.
  * **`/init`** — uses the loop engine, not the tree engine, but
    follows a similar exploration pattern via the `document` flow.

### Output

The tree engine returns a synthesised report at the end, drawn from
the highest-confidence facts and the resolved questions. Files
produced (if any) get tracked the same way as loop engine outputs.

---

## Multi-provider LLM support

`config/providers.py` is a registry of LLM provider configurations.
Eight providers are supported out of the box:

| Provider | Prefix | Notes |
|---|---|---|
| `ollama` | `ollama_chat/` | Local models via Ollama. The default. |
| `openai_compatible` | `custom_openai/` | Anything speaking the OpenAI API — llama-server, vLLM, LM Studio, ... |
| `openai` | `openai/` | The actual OpenAI API. |
| `anthropic` | `anthropic/` | Claude API. |
| `gemini` | `gemini/` | Google Gemini. |
| `groq` | `groq/` | Groq cloud. |
| `together` | `together_ai/` | Together AI. |
| `mistral` | `mistral/` | Mistral cloud. |

Switching providers is one command:

```bash
infinidev --provider openai_compatible --model "custom_openai/gemma4-26b" \
  --prompt "..." 
```

Or via env vars (`INFINIDEV_LLM_PROVIDER`, `INFINIDEV_LLM_MODEL`,
`INFINIDEV_LLM_BASE_URL`, `INFINIDEV_LLM_API_KEY`).

### Capability detection

`config/model_capabilities.py` probes the model at startup to
determine:

  * Native function-calling support (FC mode vs manual mode)
  * JSON mode support
  * Schema sanitization needs (some providers reject `additionalProperties`)
  * Streaming support

The result is cached for the session. On reconfiguration (`/models
set`, `--model` override), the cache is reset and capabilities are
re-detected.

---

## Performance instrumentation

`engine/static_analysis_timer.py` — a per-category latency
accumulator that tracks how long each part of the engine spends on
real work. Off by default; enable with `INFINIDEV_ENABLE_SA_TIMER=1`.

### Categories

| Category | What it measures |
|---|---|
| `between_llm_calls` | Wall-clock GAP between consecutive LLM invocations (the GPU-idle gap a user notices). |
| `prompt_build` | Time spent building the next iteration's prompt. |
| `summarizer_llm` | Time inside the summarizer LLM call (separate from the main one). |
| `hook_dispatch` | Behavior + UI hook callbacks. |
| `subprocess_exec` | Time inside `execute_command` subprocess runs. |
| `db_write` | SQLite write operations. |
| `tool_io` | File I/O inside tool implementations. |
| `trace_log` | The trace file writer. |
| `file_indexing` | Time spent indexing files for code intelligence. |
| `syntax_check` | Pre-write syntax validation. |
| `silent_deletion` | Top-level symbol diff for silent-deletion detection. |
| `guidance` | Detector dispatch and library lookups. |
| `plan_validate` | Plan structure validation. |

Sample output at the end of an opt-in run:

```
Static analysis accumulated latency:
  between_llm_calls   71.78 ms total      4 calls   17.95 ms avg
  prompt_build        12.08 ms total      2 calls    6.04 ms avg
  syntax_check        11.97 ms total      1 calls   11.97 ms avg
  db_write            28.23 ms total      5 calls    5.65 ms avg
  ...
  TOTAL              136.44 ms total     55 calls
```

The timer was used during development to identify and kill two
preexisting bottlenecks:

  * **DB connection setup** — fixed via thread-local pooling in
    `tools/base/db.py`. `db_write` dropped from 1812 ms to 12 ms
    per task (a 642x reduction).
  * **Synchronous file indexing** — fixed by routing through the
    background `IndexQueue`. `file_indexing` dropped from 478 ms
    to 0.18 ms per task.

Together these cut `between_llm_calls` from ~4754 ms to ~1704 ms
on a typical small task.

---

## Database & persistence

SQLite database at `~/.infinidev/infinidev.db`. WAL mode, busy
timeout configured, exponential-backoff retry on contention via
`tools/base/db.py::execute_with_retry`. Connection pooling is
thread-local: each worker thread gets its own connection that lives
for the thread's lifetime.

### Tables

| Table | Purpose |
|---|---|
| `projects` | Project registry. |
| `findings` | Knowledge tool entries with embeddings, FTS5, semantic dedup. |
| `findings_fts` | FTS5 virtual table for findings. |
| `artifacts` | Tracked file artifacts. |
| `artifact_changes` | Per-edit change history. |
| `web_cache` | HTTP cache for `web_fetch`. |
| `conversation_turns` | Chat history per session. |
| `library_docs` | Cached library documentation. |
| `exploration_trees` | Persisted tree-engine state. |
| `status_updates` | Long-running operation status. |
| `branches` | Git branch metadata per task. |
| `ci_files` | Indexed file metadata + content hash. |
| `ci_symbols` | All extracted symbols. |
| `ci_symbols_fts` | FTS5 index over symbol names, signatures, docstrings. |
| `ci_references` | All symbol usages (for `find_references`). |
| `ci_imports` | Import declarations. |
| `ci_diagnostics` | Heuristic analysis results. |
| `ci_method_bodies` | **Per-method normalized fingerprints for fuzzy similarity search** (the `find_similar_methods` backend). |

### Schema migrations

`db/service.py::init_db()` is idempotent — every `CREATE TABLE` uses
`IF NOT EXISTS` and column additions go through `_migrate_add_column`
which catches the duplicate-column error. Running `init_db()` on an
existing database is safe and applies any new schema.

---

## Settings, env vars, and operational tooling

### Settings file

`~/.infinidev/settings.json`. Loaded at startup, reloadable between
turns via `reload_all()`. Override via `INFINIDEV_*` env vars or via
`/settings` slash commands. Per-call overrides via `--model` /
`--provider` apply in-memory only and are NOT persisted.

### Key settings

| Key | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `ollama_chat/qwen3.5:9b` | The model to use. |
| `LLM_BASE_URL` | `http://localhost:11434` | Provider endpoint. |
| `LLM_PROVIDER` | `ollama` | Provider name. |
| `LLM_API_KEY` | (empty) | Required for hosted providers. |
| `LOOP_MAX_ITERATIONS` | 50 | Outer iteration cap. |
| `LOOP_MAX_TOTAL_TOOL_CALLS` | 200 | Total tool-call budget. |
| `LOOP_HISTORY_WINDOW` | 0 | History window (0 = keep all summaries). |
| `LOOP_GUIDANCE_ENABLED` | True | Guidance system on/off. |
| `LOOP_GUIDANCE_MAX_PER_TASK` | 12 | Per-task guidance rate limit. |
| `LOOP_VALIDATE_SYNTAX_BEFORE_WRITE` | True | Pre-write syntax check. |
| `LOOP_REQUIRE_NOTE_BEFORE_COMPLETE` | False | Force the agent to add a note before completing a step. |
| `LOOP_CUSTOM_TEST_COMMANDS` | (empty) | Add project-specific test command substrings. |
| `ANALYSIS_ENABLED` | True | Run the analyst phase. |
| `REVIEW_ENABLED` | True | Run the review phase. |
| `GATHER_ENABLED` | False | Run the gather phase by default. |
| `SANDBOX_ENABLED` | True | Sandbox tool execution. |
| `EXECUTE_COMMANDS_PERMISSION` | `prompt` | `allow` / `deny` / `prompt`. |
| `FILE_OPERATIONS_PERMISSION` | `allow` | Same shape for file ops. |
| `MAX_FILE_SIZE_BYTES` | 5 MB | Read-file size limit. |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.82 | Cosine threshold for finding dedup. |
| `CODE_INTERPRETER_TIMEOUT` | 30 s | Sandbox interpreter timeout. |
| `COMMAND_TIMEOUT` | 600 s | Default shell command timeout. |
| `WEB_RPM_LIMIT` | 30 | Web fetch rate limit. |

### Env vars

| Env var | Purpose |
|---|---|
| `INFINIDEV_<KEY>` | Override any setting (e.g. `INFINIDEV_LLM_MODEL=...`). |
| `INFINIDEV_ENABLE_SA_TIMER=1` | Turn on the static-analysis latency timer + final report. |
| `INFINIDEV_TRACE_FILE=<path>` | Append a per-iteration trace (prompt, thinking, tool calls, plan) to this file. **MUST be outside the workdir** or the model will read its own trace mid-task. |
| `INFINIDEV_FORCE_MANUAL_TC=1` | Force manual tool-calling mode even if the model claims FC support. |

### Profiling

`infinidev --profile ...` saves a session profile to
`~/.infinidev/profiles/`. Useful for finding hot paths in the
engine itself, complementing the SA timer (which is bound to
specific categories).

---

## Things that exist on purpose to NOT do

This section is here because some absences are by design and re-
introducing them would break things we already validated.

  * **No automatic git push.** The agent never pushes to a remote
    unless the user explicitly invokes `git_push` or grants permission
    via the prompt. Workdirs in `~/swe/runs/` should have their `origin`
    remote stripped to make accidents physically impossible.
  * **No editing of `git config`.** The agent must not change global
    git settings. Hooks that try to do this are blocked.
  * **No `--no-verify` on commits.** Pre-commit hooks exist for a
    reason. If a hook fails, the agent fixes the underlying issue
    and re-stages.
  * **No "force" or "destructive" git operations** unless the user
    explicitly asks. `git reset --hard`, `git push --force`, branch
    deletion are all behind explicit confirmation.
  * **No automatic file uploads to third-party services.** The agent
    never sends file content to external pastebins, gist services,
    or diagram renderers without explicit user permission.
  * **No reaching from `engine/` into `ui/`.** The orchestration
    package imports nothing UI-shaped. UIs implement
    `OrchestrationHooks` and pass an instance in. This direction of
    dependency is enforced by convention and reviewed in PRs.
  * **No silent default for the static-analysis timer.** Always
    opt-in via `INFINIDEV_ENABLE_SA_TIMER=1`. Two toggles for the
    same thing was confusing in an earlier iteration; we collapsed
    them into one.
  * **No tree-sitter walking inside ERROR nodes.** Tree-sitter is
    error-tolerant for parsing but produces garbage when used to
    extract structure from broken regions. The skeleton walker
    skips ERROR subtrees entirely.
  * **No new abstractions for one-time operations.** Three similar
    lines are better than a premature helper. Helpers earn their
    existence by being called from at least three places.
  * **No `.md` files except the user-facing ones in the repo root.**
    `README.md`, `TODO.md`, `FEATURE.md`, `CLAUDE.md`,
    `CHANGELOG.md` are the canonical user-facing docs. Internal
    notes go in commit messages and code comments, not new
    markdown files.

---

## Appendix: where things live

```
src/infinidev/
├── agents/                   InfinidevAgent — role + tool binding
├── cli/                      Classic CLI entry point + index queue
│   ├── main.py               Thin adapter over orchestration.run_task
│   ├── index_queue.py        Background indexing worker thread
│   └── initial_index.py      First-pass project indexer
├── code_intel/               Symbol extraction + similarity + skeleton
│   ├── parsers/              Per-language tree-sitter wrappers
│   ├── indexer.py            File indexing pipeline
│   ├── index.py              SQLite read/write for ci_symbols
│   ├── query.py              search_symbols, search_by_docstring,
│   │                         find_definition, find_references
│   ├── method_index.py       Per-method fingerprints (find_similar_methods)
│   ├── syntax_check.py       Pre-write validation + skeleton extractor
│   ├── smart_index.py        Hash-based skip-if-unchanged logic
│   └── background_indexer.py Process-global IndexQueue registry
├── config/                   Settings, providers, model capabilities
│   ├── settings.py           All settings + INFINIDEV_ env var loader
│   ├── providers.py          Multi-provider registry (8 providers)
│   ├── llm.py                LiteLLM parameter builder
│   └── model_capabilities.py Runtime FC / JSON / schema sniffing
├── db/                       SQLite layer
│   └── service.py            Schema migrations + DB helpers
├── engine/                   Core agent engine
│   ├── orchestration/        Unified pipeline (TUI/classic/--prompt)
│   │   ├── pipeline.py       run_task + run_flow_task + Protocol
│   │   └── hooks.py          NoOp / Click / NonInteractive defaults
│   ├── loop/                 The plan-execute-summarize cycle
│   │   ├── engine.py         The main loop
│   │   ├── llm_caller.py     LLM dispatch (FC + manual modes)
│   │   ├── tool_executor.py  Tool dispatch + result handling
│   │   └── ...
│   ├── guidance/             Coaching system for stuck models
│   │   ├── library.py        13 GuidanceEntry definitions
│   │   ├── detectors.py      All _has_* functions + _DETECTORS list
│   │   ├── hooks.py          maybe_queue_guidance / drain_pending_guidance
│   │   └── test_runners.py   is_test_command + normalize_test_command
│   ├── test_parsers/         7 per-runner parsers (pytest/jest/...)
│   ├── phases/               PhaseEngine — CLASSIFY/INVESTIGATE/PLAN/EXECUTE
│   ├── tree/                 TreeEngine — exploration / brainstorm
│   ├── analysis/             AnalysisEngine + ReviewEngine
│   ├── behavior/             Behavior tracker + scoring
│   ├── hooks/                UI event listeners
│   ├── formats/              Tool-call parsing for manual TC mode
│   ├── static_analysis_timer.py  Per-category latency accumulator
│   └── llm_client.py         LiteLLM wrapper with streaming
├── tools/                    All agent tools
│   ├── base/                 BaseTool, context binding, DB helpers
│   ├── file/                 read_file, create_file, replace_lines, ...
│   ├── code_intel/           list_symbols, search_*, find_*, edit_symbol, ...
│   ├── git/                  branch / commit / diff / status / push
│   ├── shell/                execute_command, code_interpreter
│   ├── knowledge/            findings + reports
│   ├── meta/                 help, plan management, tail_test_output, ...
│   ├── docs/                 Documentation tools
│   ├── chat/                 Inter-agent messaging
│   ├── web/                  search / fetch / code search
│   └── permission.py         Permission gate for sensitive ops
├── ui/                       prompt_toolkit TUI
│   ├── app.py                InfinidevApp — Application root
│   ├── workers.py            Background pipeline runner (thin adapter)
│   ├── hooks_tui.py          TUIHooks — implements OrchestrationHooks
│   ├── widgets/              Chat, file diff, autocomplete, status bar
│   ├── handlers/             Command + dialog handlers
│   └── dialogs/              Settings editor, model picker
├── flows/                    Event listeners (currently a stub)
├── gather/                   Codebase context gathering
└── prompts/                  All prompt templates + system prompts
    ├── shared.py             LOOP_PROTOCOL — the agent contract
    ├── flows.py              Per-flow identity + backstory
    └── phases.py             PhaseStrategy definitions
```
