# Infinidev runtime code audit — 2026-08-05

## Scope and method

This audit targets the production runtime paths most likely to explain a TUI stuck on
`Working...`, repeated `infinidev`/`ken mcp` entries, or rapid memory growth: MCP lifecycle,
subprocesses, parallel tool execution, councils, background workers, global registries, queues,
SQLite connections, and shutdown. It combines source inspection, architecture/call-graph review,
focused concurrency tests, and the complete pytest suite. It is not a claim that every one of the
823 indexed files received a line-by-line formal verification.

## Executive result

One critical race directly explains the reported symptoms. Parallel tool calls were allowed to
perform concurrent JSON-RPC exchanges over one MCP stdio stream. A caller could consume another
caller's response, leave that caller asleep until timeout, and initiate repeated Ken teardown and
restart cycles while the UI remained in `Working...`. The fix serializes a complete request and
response per server while preserving concurrency between different servers.

The same patch now starts each MCP server in its own process group, terminates the whole group,
and closes the default manager at interpreter shutdown. A regression test sends 24 calls from
eight threads and verifies that all complete through exactly one server PID.

## Findings

### P0 — Concurrent MCP RPC could strand responses and cause restart storms — fixed

- Location: `src/infinidev/engine/mcp_client.py`, `McpServerClient._request` / `_rpc`.
- Cause: `_rpc` read from one shared inbox outside the lifecycle lock. `_pending` tried to route
  mismatched IDs, but a caller already blocked in `Queue.get()` did not wake when another thread
  placed its response in `_pending`. The final response ordering could therefore strand a caller
  until the 30-second timeout.
- Amplifier: regular tool calls can run in a pool of up to eight workers in
  `src/infinidev/engine/tool_executor.py`. MCP tools participate in that pool.
- Effect: apparent freeze, timeout latency, repeated server teardown/restart, excess Ken processes
  over time, and tool failures unrelated to the actual request.
- Correction: one `_request_lock` per MCP server protects process initialization plus the complete
  request/response exchange. Different MCP servers can still run concurrently.
- Regression: `test_parallel_calls_share_one_process_without_stranding_responses`.

### P1 — MCP termination did not own the whole process tree — fixed

- Location: `src/infinidev/engine/mcp_client.py`, `_spawn` and `_teardown`.
- Cause: the server was launched in the application's process group and teardown called
  `terminate()`/`kill()` only on the immediate child.
- Effect: an MCP server that starts helpers can leave descendants behind after restart or exit.
- Correction: POSIX servers now use `start_new_session=True`; teardown signals their process group,
  escalating from `SIGTERM` to `SIGKILL`.

### P1 — Default MCP manager had no explicit process-exit cleanup — fixed

- Location: `src/infinidev/engine/mcp_client.py`, default-manager singleton.
- Cause: normal TUI shutdown ended Ken's ranking session but did not explicitly close MCP children.
  It relied on pipe EOF and child behaviour.
- Correction: `reset_default_mcp_manager` is registered with `atexit` and closes every server.

### P1 — Council observer retains every council transcript forever — open

- Location: `src/infinidev/engine/council/observer.py`, module-global `_sessions`.
- Cause: completed and failed councils are never evicted in production. `clear_councils()` is only
  used as a test helper. Every message is stored both in the combined transcript and in its member
  transcript (the same message object is referenced twice).
- Effect: a long-lived TUI with automatic or repeated councils grows monotonically. Opening the
  agents browser can additionally `deepcopy` every retained transcript.
- Recommendation: retain running councils plus a configurable bounded number of completed
  summaries; persist full transcripts in SQLite if historical inspection is required. Avoid full
  transcript copies in list views.

### P1 — Working-memory embedding backlog is unbounded — open

- Location: `src/infinidev/engine/working_memory.py`, class-level `_embed_queue`.
- Cause: `queue.Queue()` has no `maxsize`, and `_enqueue_embed()` always performs an unrestricted
  `put`. One worker embeds batches of at most 16.
- Effect: if embedding is slower than archival, or the embedder stalls, queued text and metadata
  grow without a memory bound. `_inflight` also remains elevated until processing succeeds or the
  worker settles the batch.
- Recommendation: bound the queue, coalesce/deduplicate pending records, expose backlog health, and
  degrade to keyword-only recall rather than accepting unlimited work.

### P2 — Changing SQLite database paths leaks the previous cached connection — open

- Location: `src/infinidev/code_intel/_db.py`, `get_pooled_connection`.
- Cause: when a thread-local cached connection exists for a different path, the function opens and
  replaces it without closing the old connection.
- Effect: workspace/database switching can retain file descriptors, WAL readers, mmap state, and
  SQLite memory until garbage collection.
- Recommendation: close the mismatched cached connection before replacing it and add a regression
  test that switches paths repeatedly.

### P2 — Runtime MCP setting/config changes do not rebuild MCP singletons — open

- Locations: `src/infinidev/config/settings.py::reload_all`,
  `src/infinidev/engine/mcp_client.py::get_default_mcp_manager`, and the MCP bridge cache.
- Cause: settings reload mutates the settings object, but does not reset the manager, Ken facade, or
  discovered tool-class cache.
- Effect: changing `MCP_ENABLED`, filters, commands, or `.mcp.json` can leave the old processes and
  tool catalog active until restart. UI state and runtime behaviour can disagree.
- Recommendation: introduce one explicit `reload_mcp_runtime()` transaction that closes the old
  manager and clears all three dependent caches.

### P2 — Tool execution creates a fresh pool of up to eight threads per parallel batch — open

- Location: `src/infinidev/engine/tool_executor.py`.
- This does not create additional OS processes, but Linux process viewers configured to show
  threads can display them as many `infinidev` entries. Ken and numerical/embedding libraries may
  also create native worker threads.
- Risk: repeated pool construction adds churn; simultaneous council, critic, UI, embedding, and
  tool pools can temporarily produce dozens of threads and amplify per-thread SQLite/model state.
- Recommendation: use a bounded process-wide tool executor with task cancellation and telemetry.
  This is a resource-efficiency issue, not the root MCP process-spawn bug.

### P3 — Several process-global/UI lifecycles have no bounded shutdown — open

- `InfinidevApp._start_animation_timer` starts an infinite daemon thread with no stop event.
- `InfinidevAgent.deactivate()` is a no-op, so the main agent's process-global tool context is not
  cleared there (short-lived specialist agents do clear theirs explicitly).
- `EventBus.subscribe()` allows duplicate callbacks; the TUI subscription is not paired with an
  unsubscribe on shutdown.
- These are low-impact in the normal one-app-per-process CLI, but matter for embedded use, repeated
  app construction in one interpreter, and deterministic teardown tests.

## What is not the root cause

- The configured `.mcp.json` correctly launches `ken mcp`; it does not recursively launch
  Infinidev.
- The `ken` and `infinidev` executables resolve to separate uv tool installations.
- Qwen provider changes do not touch process or MCP creation.
- Seeing many rows named `infinidev` in `htop` is not by itself proof of many processes when
  `Show custom thread names` / thread display is enabled. Confirm using PID/PPID and process-only
  mode. The repeated Ken lifecycle, however, was a real code defect regardless of presentation.

## Validation

- Focused MCP/process/concurrency suite: 89 tests passed.
- Added stress regression: 24 calls, eight caller threads, one `started` event/PID.
- Full suite: 2,977 passed, 1 skipped (8 pre-existing Pydantic serializer warnings).

## Recommended order for remaining fixes

1. Bound and persist/evict council observer state.
2. Bound the embedding queue and add backlog telemetry.
3. Close replaced SQLite connections.
4. Add transactional MCP runtime reload.
5. Consolidate transient thread pools and add a runtime diagnostics panel showing processes,
   threads, queue depths, active councils, and MCP restart counts.
