# Phase 1 discovery: command-output continuity

This note records the contracts and integration points confirmed before the
command-output continuity work. It is discovery evidence, not a runtime design
change.

## Compatibility contracts preserved

The existing model-facing shell API remains unchanged:

- `infinidev.tools.shell.execute_command_input.ExecuteCommandInput` remains the
  argument schema.
- `infinidev.tools.shell.execute_command_tool.ExecuteCommandTool._run` keeps the
  public call contract:
  ```python
  _run(
      command: str,
      timeout: int | None = None,
      cwd: str | None = None,
      env: dict[str, str] | None = None,
      rationale: str = "",
  ) -> str
  ```
- The legacy result remains a JSON string containing `exit_code`, truncated
  `stdout`, truncated `stderr`, and `success`, plus `killed_reason` for an
  interrupted command. Capture is additive and best-effort: when disabled or
  unsuccessful, no command-output handle is announced and the legacy result
  shape is preserved.
- Existing stdout/stderr transcript limits remain 10,000 and 5,000 characters,
  respectively. Durable capture, when enabled, stores the decoded text before
  those slices are applied.

## Both `ExecuteCommandTool` execution routes

`src/infinidev/tools/shell/execute_command_tool.py` has two foreground execution
strategies, and both converge on the same pre-truncation capture boundary:

1. **Sealed/classic route:** `_run()` selects `_run_sealed()` when no stdin UI
   handler is registered. The subprocess uses `stdin=DEVNULL`, obtains its
   complete stdout/stderr through `communicate()`, and then calls
   `_capture_before_truncation()`.
2. **TUI/live-feedback route:** `_run()` selects `_run_with_stdin_detection()`
   when an stdin handler is registered. That route accumulates streamed byte
   buffers while emitting live output, drains the process, decodes the complete
   buffers, and then calls `_capture_before_truncation()` (including timeout,
   cancellation, and user-kill results).

Therefore capture must be verified through both the normal sealed path and the
TUI/live-feedback path; changing only one route would violate the contract.

## Private catalog and bounded read API

The implementation boundary is
`src/infinidev/engine/command_output_store.py`:

- `CommandOutputHandle` is the opaque catalog identity; `to_dict()` is the
  descriptor attached to a tool result.
- `CommandOutputStore.store_streams()` atomically stores and catalogs complete
  decoded streams.
- `CommandOutputStore.read_text()` validates and reconstructs a complete stream
  for trusted internal consumers.
- `CommandOutputStore.read_range()` provides a bounded UTF-8-safe range read.
- `CommandOutputStore.sweep()` applies retention cleanup.

The public `artifacts` catalog stores only opaque command-output metadata and a
private storage key; raw output is not placed in searchable artifact content.
Model access is through the scope-bound
`infinidev.tools.knowledge.read_command_output.ReadCommandOutputTool`, rather
than through filesystem paths or a general artifact-content API.

## Configuration, registration, and prompt integration

Concrete integration files are:

- `src/infinidev/config/settings.py`: the `COMMAND_OUTPUT_*` feature flag,
  finite artifact/session/project quotas, storage timeout, retention and sweep
  grace, plus independent auto-note and note-compaction opt-ins.
- `src/infinidev/tools/__init__.py`: imports and registers
  `ReadCommandOutputTool`; registration is gated by
  `COMMAND_OUTPUT_CAPTURE_ENABLED` for both normal and small-model tool sets.
- `src/infinidev/tools/knowledge/__init__.py` and
  `src/infinidev/tools/knowledge/read_command_output.py`: export the bounded
  reader and its input schema.
- `src/infinidev/prompts/tool_hints.py`: documents how a model follows an
  opaque descriptor with `read_command_output`.
- `src/infinidev/tools/meta/help_content.py`: includes the reader in tool help.
- `src/infinidev/engine/loop/step_manager.py` and
  `src/infinidev/engine/working_memory.py`: consume descriptors for optional,
  traceable notes without copying raw command output into canonical memory.

## Regression coverage identified for expansion

The affected regression suites are:

- `tests/test_tools_shell.py` — sealed execution, exact legacy shape,
  pre-truncation capture, and soft-disable/failure behavior.
- `tests/test_live_command_feedback.py` — TUI/live-feedback capture path.
- `tests/test_command_output_store.py` — private storage, catalog integrity,
  scope checks, quotas, atomicity, tamper resistance, and retention.
- `tests/test_read_command_output_tool.py` — bounded reads, scope enforcement,
  UTF-8 ranges, feature gating, and registration.
- `tests/test_tool_runner_transcript.py` — opaque descriptor propagation from a
  tool result into loop context.
- `tests/test_step_manager_summary.py` — independent auto-note and compaction
  flags plus descriptor-only note creation.
- `tests/test_working_memory.py` — immutable traceable-note envelopes and
  deterministic compaction.
- `tests/test_council.py` — isolated internal analysis of private command
  output without leaking it into shared context.
- `tests/test_tool_docs_complete.py` — registration/help/documentation
  completeness when capture is enabled.

## Reviewed delivery scope

The authoritative extraction for the reviewed delivery contains **33 changed
files**, including **11 test files** and **4 documentation-oriented files**.
These counts describe the complete review surface; they do not imply a
chronology, authorship boundary, or separation between phases.

The extracted changes cover the command-output continuity implementation and
its surrounding integration:

- private, quota- and retention-bounded command-output storage;
- pre-truncation capture in both foreground shell execution routes;
- the scope-bound range reader, tool registration, configuration, prompt help,
  loop descriptor propagation, traceable working-memory notes, and isolated
  council consumption;
- regression coverage for storage integrity, bounded reads, shell and live
  capture, loop propagation, note compaction, council isolation, registration,
  and tool documentation; and
- documentation of the capture contract, rollout controls, security boundary,
  and discovery map.

The delivery also contains supporting changes outside that central path. In
particular, `index_queue.py` now wakes its worker with a sentinel for prompt,
restart-safe shutdown; `smart_index.py` recognizes successfully indexed files
that legitimately contain zero symbols; and the general `help` implementation
now discovers registered, pseudo-, MCP, aliased, and retired tools from live
schemas while retaining the hand-written code-interpreter reference. Their
corresponding regression updates are part of the 11-file test surface above.

Other runtime, UI, tool, database, session, and code-intelligence modifications
visible in the extracted delivery remain part of the same 33-file review
surface. This report makes no claim that any listed path predated another,
belonged to a separate corrective phase, or was the only intentionally created
artifact.

## Full-suite verification evidence

The complete configured test suite was executed from the repository root after
the delivery changes, using the locked environment:

```text
command: uv run pytest -q
exit_code: 0
summary: 2468 passed, 20 skipped, 1 warning in 107.07s (0:01:47)
stderr: warning: No `requires-python` value found in the workspace. Defaulting to `>=3.13`.
warning_source: tests/test_chat_agent.py::TestRespondTerminator::test_respond_first_iteration
warning_type: RequestsDependencyWarning
```

The progress output reached `[100%]`. The single warning came from
`requests/__init__.py`, reporting that the installed urllib3/chardet or
charset-normalizer versions do not match a supported combination; it did not
fail the suite. This block records the actual command result rather than a
result inferred from test collection or static inspection.

## Final Git inspection evidence

The following non-destructive inspection was run after the suite:

```text
commands:
  git rev-parse HEAD
  git branch --show-current
  git status --short
  git diff --stat
  git ls-files --others --exclude-standard
command_exit_code: 0
head: d51cf2aa2d70629604911230079839724c0ffef4
branch: main
status_count: 28 paths (24 tracked modifications, 4 untracked paths)
status_breakdown: 19 runtime/source paths, 7 test paths, 2 documentation-oriented paths
tracked_diff_stat: 24 files changed, 872 insertions(+), 266 deletions(-)
untracked:
  HARNESS.md
  docs/PHASE_1_DISCOVERY.md
  tests/test_foreground_tool_cancel.py
  tests/test_permission_modal.py
```

The 28-path Git status snapshot is intentionally reported separately from the
reviewer's authoritative 33-file extraction above. They are different scopes:
the former is the worktree relative to `HEAD` at verification time, while the
latter is the review surface supplied by extraction. No chronology or
provenance is inferred from either count.

The exact final `git status --short` snapshot was:

```text
 M src/infinidev/cli/index_queue.py
 M src/infinidev/cli/session_resume.py
 M src/infinidev/code_intel/interpreter_api.py
 M src/infinidev/code_intel/smart_index.py
 M src/infinidev/db/service.py
 M src/infinidev/engine/guidance/similarity_detector.py
 M src/infinidev/engine/loop/engine.py
 M src/infinidev/engine/loop/tool_runner.py
 M src/infinidev/engine/tool_dispatch.py
 M src/infinidev/prompts/tool_hints.py
 M src/infinidev/tools/code_intel/iter_symbols_tool.py
 M src/infinidev/tools/file/read_file_tool.py
 M src/infinidev/tools/meta/help_content.py
 M src/infinidev/tools/meta/help_tool.py
 M src/infinidev/ui/app.py
 M src/infinidev/ui/controls/chat_history.py
 M src/infinidev/ui/dialogs/permission_detail.py
 M src/infinidev/ui/handlers/commands.py
 M src/infinidev/ui/keybindings.py
 M tests/test_chat_scroll_momentum.py
 M tests/test_help_tool.py
 M tests/test_index_queue.py
 M tests/test_session_resume.py
 M tests/test_tool_runner_transcript.py
?? HARNESS.md
?? docs/PHASE_1_DISCOVERY.md
?? tests/test_foreground_tool_cancel.py
?? tests/test_permission_modal.py
```
