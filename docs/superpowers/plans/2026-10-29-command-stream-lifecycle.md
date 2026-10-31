# Command Stream Lifecycle Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task.

**Goal:** Avoid busy waiting, orphan processes and incomplete output artifacts.

**Architecture:** Preserve the interactive executor and its public result format.
Track live pipes and stream completeness inside its existing read loop.

**Tech Stack:** Python 3.13+, subprocess, select, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-29-command-stream-lifecycle-design.md`

## Global Constraints

- Run Python tools through `uv run`.
- Preserve permissions, stdin prompts, exit codes and public JSON.
- Leave pre-existing local changes intact; keep changes in the current checkout.
- Do not commit or publish as part of this task.

## Review Focus

- EOF on one stream must not lose output from the other.
- EOF on both streams must retain timeout and cancellation.
- Reader exceptions, including KeyboardInterrupt, must reap the subprocess.
- An incomplete stream must not prevent capture of a complete peer stream.
- A complete stream exactly at the memory limit remains eligible for capture.

### Task 1: Remove EOF pipes and guarantee exception cleanup

**Files:** Modify `src/infinidev/tools/shell/execute_command_tool.py`;
create `tests/test_command_stream_lifecycle.py`.

**Interfaces:** Preserve `_run_with_stdin_detection(command, cwd, run_env, timeout)`.

- [x] Add real-process regressions for EOF busy waiting and reader exceptions.
- [x] Run the new tests and confirm failures expose busy waiting and live children.
- [x] Track live descriptors outside the loop, remove EOF descriptors, wait when
      none remain, and terminate the process group when the reader raises.
- [x] Check timeout, cancellation, prompts and output after one stream closes.
- [x] Run `uv run pytest tests/test_command_stream_lifecycle.py -q`.

### Task 2: Preserve capture integrity under bounded buffering

**Files:** Same production and test files.

**Interfaces:** Add optional `incomplete_streams` to `_capture_before_truncation`;
existing callers omit it and retain their result shape.

- [x] Add regressions for overflow on stdout/stderr, including a complete peer
      stream and an exact-boundary stream; confirm overflow fails before the fix.
- [x] Track discarded bytes in the reader and exclude incomplete streams from
      artifact storage, preserving the legacy output tail.
- [x] Run related shell, live feedback and output-store tests.
- [x] Run `uv run pytest`, inspect failures and review the diff.

## Verification record

- Baseline: 59 shell and live-feedback tests passed.
- Task 1 RED: four failures reproduced EOF busy polling (69,843/80,894 polls
  in approximately 0.35 seconds) and live processes after reader exceptions.
- Task 1 GREEN: eight tests passed, including timeout/cancellation after EOF
  and accepting/stopping stdin prompts.
- Task 2 RED: both stdout and stderr overflow produced incorrect full-capture
  handles; the exact-boundary control passed.
- Related tests: 94 passed, including command-output storage and retrieval.
- Post-fix real-process measurement: four select calls with stderr closed and
  four with both streams closed, versus roughly 70–80k before the fix.
- Independent read-only review: no introduced defects found. Optional future
  coverage: force overflow during final drain/interruption and verify a real
  descendant process is also stopped after a reader exception. The reviewer
  confirmed the existing group termination and completeness propagation paths.
- Full suite: `uv run pytest` completed with exit code 0: 5,357 passed,
  13 skipped, 11 Pydantic serialization warnings in 194.32 seconds.
- Final diff review and `git diff --check` passed. Four task files changed;
  pre-existing local changes were preserved. No commit was created.
