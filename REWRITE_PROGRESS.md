# Infinidev Rewrite Progress

Updated: 2026-07-30

Three workstreams: the TUI, the loop's memory model, and Ken/MCP as the
default context backend.

## Provider compatibility

- [x] Add MiniMax M3 to the provider catalog.
- [x] Confirm the installed LiteLLM catalog supports `minimax/MiniMax-M3`.
- [x] Add Kimi K3 to the Kimi static catalog.
- [x] Add GLM 5.2 to Z.AI and Z.AI Coding catalogs.
- [x] Add context-window fallbacks for MiniMax M3, Kimi K3, and GLM 5.2.
- [x] Add provider catalog regression tests.
- [ ] Validate Kimi K3 against the live Moonshot account catalog when the model is enabled for the account.
- [ ] Validate GLM 5.2 against the live Z.AI account catalog when the model is enabled for the account.

## Loop memory rewrite

The loop keeps its plan-execute-summarize shape; what changed is what
happens to context when a step closes.

- [x] Explicit task lifecycle types: pending, active, blocked, completed, failed, cancelled.
- [x] Runtime state object for tasks, chat history, tool counters, working memory.
- [x] Public chat transcript stays immutable and separate from compactable model memory.
- [x] **Recoverable eviction** — `engine/working_memory.py` archives every tool
      call + result of a closing step into SQLite with an embedding, on a
      background worker so archiving costs the loop nothing.
- [x] **`recall_context` tool** — cosine search over the archive (keyword
      fallback), so the model retrieves an old error or file listing instead
      of re-running the command.
- [x] **Explicit prompt retention** — newest `WORKING_MEMORY_VERBATIM_STEPS`
      summaries render in full, older ones collapse to one line, and the block
      says where the detail went.
- [x] End-of-task summary lands in the same conversation history *and* the
      searchable archive.
- [x] Objective verification, review, permissions, guidance, retries, and
      cancellation preserved through the compatibility pipeline.
- [x] Runtime task state and event history persisted (`runtime_events`).
- [x] Migration and compatibility tests for existing loop consumers.
- [x] Fixed: `execute_with_retry` never commits — every writer now commits in
      its own callback, and each record carries the `db_path` it belongs to.
      (Without this the embed worker's connection saw no rows and semantic
      recall silently degraded to keyword matching.)
- [ ] Replace plan-execute-summarize control flow with event-driven scheduling.
      *Deliberately deferred: the memory model was the actual complaint, and
      the scheduler rewrite would churn objective verification, the critic,
      and the guidance system with no user-visible gain.*

## MCP and Ken integration

- [x] `.mcp.json` with `ken mcp`; Ken is the default server when no config exists.
- [x] Generic MCP manager that loads config and dispatches JSON-RPC.
- [x] **Real protocol** — `initialize` + `notifications/initialized` handshake,
      responses matched by `id`, reader threads on both pipes, real per-request
      deadlines, `content`/`structuredContent`/`isError` parsing.
      (The previous client skipped the handshake, read the first line as the
      answer, ignored its own timeout, and expected a `hits` key no MCP server
      emits — every Ken call returned empty.)
- [x] **Correct Ken tool names** — `ken_search_files`, `ken_search_symbols`,
      `ken_grep`, `ken_recall`, `ken_remember`, `ken_rank`, `ken_callgraph`.
      (Previously `search` / `memory_search` / `index_status`, which do not exist.)
- [x] Ken rank/index history available as a context-routing source.
- [x] Fallback behaviour when a server is unavailable — every Ken method
      degrades locally; no tool can fail because MCP is down.
- [x] `semantic_search`, `memory_search`, `remember` in the user-facing tool list.
- [x] Multiple MCP servers in one config file; workspace config extends user config.
- [x] Per-server `timeout`, `startup_timeout`, `tool_ttl`; exponential backoff (cap 8 s).
- [x] `McpRuntimeBridge` records MCP events in working memory without leaking to chat.
- [x] `/mcp` panel: per-server health, tool counts, stderr tail, start/stop/restart.
- [x] Warm-up at startup so the first semantic search doesn't pay for Ken's model load.
- [x] Tests for startup, tool discovery, timeout, crash, and graceful degradation —
      against a protocol-conformant fake server (`tests/mcp_fake_server.py`).
- [x] **Reverted the tool hijack** — `code_search`, `glob`, `find_references`,
      `list_symbols`, `search_symbols` are deterministic again. Ken is now a
      *fallback* on those (semantic symbols, call graph, literal worktree
      grep), never a replacement for exact search.

## User interface

- [x] Transcript-first layout: full-width conversation, framed composer,
      one status line, one column of side margin. Explorer and sidebar keep
      every panel — both are toggles, closed by default.
- [x] Message rendering stripped to the content: `>` opens the user's turn,
      the assistant's reply carries no mark or name at all, no coloured
      backgrounds. Headers only for named speakers.
- [x] Conversation turns are never folded into groups (they were collapsing
      under "▼ Responses (2)").
- [x] Copy affordance reduced from " [⧉] " on every message to one dim glyph
      on assistant replies; the click target is unchanged.
- [x] Opening banner (name, version, workspace, model) replaces the
      "Welcome to Infinidev!" system line; startup no longer narrates
      indexing at all — Ken owns the index.
- [x] Animated working indicator with the current phase and elapsed time,
      replacing a static two-line "thinking...".
- [x] Short conversations are bottom-anchored so they sit on the composer.
- [x] Modals rebuilt: rounded frame, inlined title, hints in the border,
      interior padding, shadow.
- [x] Sidebar: fixed 30-column width (was ~30% of the terminal), quiet
      section rules instead of solid colour bars, sections sized to their
      content, empty sections hidden, and a new FILES CHANGED section fed
      by the previously no-op `on_file_change` hook.
- [x] `?` opens the help — the status line advertised it with nothing
      behind it. Sidebar reachable via Alt+. , F4, or `/sidebar`.
- [x] Help rewritten and covered by a test that fails if it advertises a
      command with no handler.
- [x] Compact, expandable tool groups (`✓ Ran N tools ▸`) with per-tool detail.
- [x] Context usage, model, and git branch on the status line without
      stealing transcript space.
- [x] Palette rebuilt: no per-message backgrounds, chrome close to the
      terminal's own colours.
- [x] Inline ghost placeholder on the cursor's line (was a separate row above it).
- [x] Momentum scrolling in the transcript.
- [x] Responsive: everything drops in priority order as the terminal narrows;
      the composer frame tracks its container, not the terminal.
- [x] Existing file picker, dialogs, autocomplete, copy, diff, image, and
      cancellation features preserved.
- [x] Deterministic renderer tests (`tests/test_tui_render.py`) — the layout is
      drawn into an in-memory screen and asserted as text.
- [ ] Persistent task rail with dependencies. *Not built: the sidebar's STEPS
      panel already shows plan state, and a second always-on rail contradicts
      the transcript-first goal.*
- [ ] Command palette. *Not built: `/` autocomplete covers the same ground.*
- [ ] Provider/model setup flow with masked API-key editing. *Existing
      `/models manage` + settings editor unchanged.*

## Verification

- [x] Provider regression tests.
- [x] Context and ranking regression tests.
- [x] Full suite green.
- [x] MCP client verified against the real `ken mcp` (handshake, 30 tools,
      `ken_search_files` / `ken_recall` / `ken_grep` / `ken_rank` payloads).
- [x] `semantic_search`, `memory_search`, and the `search_symbols` semantic
      fallback verified against this repo's live Ken index.
- [x] Semantic recall verified end-to-end with real embeddings (queries that
      share no vocabulary with the archived text still retrieve it).
- [ ] Interactive smoke test with a local model.
- [ ] Interactive smoke test with MiniMax M3 / Kimi K3 / GLM 5.2.
- [ ] Remove the legacy code-intel implementations once Ken is verified on a
      second, non-Python workspace.
