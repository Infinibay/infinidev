# Incremental Index Freshness Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Refresh obsolete parser output and bound reads of discarded source files.

**Architecture:** Preserve both public indexing APIs and the SQLite schema. Use
the existing versioned index-state query and source-size limit in both paths.

**Tech Stack:** Python, tree-sitter, SQLite, pytest; existing dependencies only.

**Spec:** `docs/superpowers/specs/2026-10-29-incremental-index-freshness-design.md`

## Global Constraints

- Use `uv run --frozen` to preserve the dependency lockfile.
- Preserve existing local changes; do not commit or publish.
- Preserve `MAX_FILE_SIZE = 1_000_000` for source files and existing config policy.
- Keep public signatures, DB schema and `notify_integrity` behavior.

## Review Focus

- Parser version changes and legacy version zero must bypass the hash-only skip.
- Zero symbols and config tracking must still count as successful updates.
- An unavailable grammar must not cause false success on an old matching hash.
- The exact source-size boundary remains indexable; boundary plus one is skipped.
- Config files above the source limit remain tracked; unsupported types avoid I/O.

### Task 1: Honor parser versions throughout smart indexing

**Files:** Modify `src/infinidev/code_intel/smart_index.py`;
create `tests/test_incremental_index_freshness.py`.

**Interfaces:** Keep `ensure_indexed(...) -> bool`; use
`get_file_index_state(...) -> tuple[str, int] | None` for both comparisons.

- [x] Add real-index regressions for parser bumps and legacy rows, including
      zero-symbol source/config files, plus a missing-grammar control.
- [x] Confirm stale-version regressions fail before implementation.
- [x] Compare expected `(content_hash, parser_version)` before and after indexing.
- [x] Run focused tests and preserve baseline-integrity behavior.

### Task 2: Bound discarded-file reads

**Files:** Modify `src/infinidev/code_intel/smart_index.py` and
`src/infinidev/code_intel/indexer.py`; extend the same test module.

**Interfaces:** Keep `index_file(...) -> int`; use its existing source size limit.

- [x] Add byte-observed real-file tests for oversized source and unsupported types;
      include exact-boundary, boundary-plus-one and large-config controls.
- [x] Confirm excess bytes are read before the fix.
- [x] Detect unsupported languages before content reads; use bounded reads for
      source files in both entry points and preserve unbounded config tracking.
- [x] Run related index, parser, queue and integrity tests.
- [x] Obtain independent review, run `uv run --frozen pytest`, and inspect the diff.

## Verification record

- Initial baseline: 53 tests passed.
- Task 1 RED: six stale-version cases failed (parser bumps and legacy version
  zero across normal Python, symbol-free Python and TOML).
- Task 1 GREEN: 33 focused tests passed, including unavailable grammar.
- Task 2 RED: oversized source reads consumed 16,777,216 bytes through
  `ensure_indexed` and 8,388,608 through `index_file` for one 8 MiB source;
  the unsupported-type regression also failed.
- Related suite: 207 passed in 34.48 seconds, including real indexing, queue
  lifecycle, parser versions and integrity notifications. Sixteen new cases.
- Implementation deliberately retains the existing second read for valid changed
  files; sharing snapshots would require a separate interface design.
- Independent review found no blocking defects and independently ran the sixteen
  new cases successfully. Optional coverage suggestion: also exercise the
  existing unsupported-type early return through direct `index_file`, beyond
  the `ensure_indexed` regression and existing integration test.
- Full suite: `uv run --frozen --no-sync pytest` exited 0 with 5,373 passed,
  13 skipped and 11 Pydantic serialization warnings in 198.83 seconds.
- Final diff review and `git diff --check` passed. This round modified two
  production files and added one test module plus its spec/plan. Prior changes
  and the dependency lockfile were preserved; no commit was created.
