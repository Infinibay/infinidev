# Opt-in Tool-Result Status Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans.

**Goal:** Observe repeated retained tool-result fingerprints for a named child run.

**Architecture:** Add a separate optional CLI JSON member backed by bounded SQL
metadata projections. Use an explicit-path read-only branch of the shared DB
helper; preserve writable callers and the existing file-only status contract.

**Tech Stack:** Python, SQLite, argparse, pytest.

**Spec:** Root authorized offline opt-in observations through status: session/run
selection, finite windows, metadata only, no inference, feedback, cancellation or
live installation. Missing/busy/corrupt data must not become a zero count.

## Constraints and review focus

- No raw arguments, commands, reasoning, messages or result bodies in output.
- Default status JSON and named/unscoped liveness semantics stay unchanged.
- 4096 retained trace rows, 512 result rows, 16KiB per trace JSON; partial history.
- Separate mode=ro snapshot, 50ms busy timeout, 2M SQL steps, 250ms cooperative
  elapsed budget, 128KiB SQLite value limit; no absolute filesystem deadline.
- Review invalid primitive/event types, foreign sessions/runs, retained gaps,
  WAL visibility, schema/budget failures and settings/model-free CLI imports.

## Task 1: Reader, opt-in CLI and compatibility

Files: `src/infinidev/engine/result_observation.py`, `src/infinidev/code_intel/_db.py`,
`src/infinidev/cli/status.py`, `tests/test_result_observation.py`,
`tests/test_session_status_cli.py`, `docs/SESSION_STATUS.md`.

Interface: `read_tool_results(workspace: Path, session_id: str, run_id: str) -> dict`;
CLI `--tool-results` requires `--run`; DB helper adds keyword-only `read_only=False`.

- [x] Observe RED before implementation (18 failures; baseline CLI3 PASS).
- [x] Implement metadata projection, bounded shared read-only connection and CLI.
- [x] Pin malformed types/deadline with 4 RED and invalid event with 3 RED; fix.
- [x] Verify 95 focused PASS, including writable DB/status compatibility.
- [x] Independent Ken source review: no blockers across six exact file hashes.
- [x] Verify full final suite and source equality:6072PASS/14SKIP/0FAIL,1144files.
- [x] Freeze reviewed source for commit/push; publication receipt kept outside repo.

## Decisions

Use retained worktree7e28 without resetting it. Original helper settings imports
become lazy so the observer remains bootstrap-free. Fingerprints describe recorded
UTF-8 returned text; they do not certify content completeness, work quality or idle.
A child closure remains a child observation, never parent delivery or restart authority.

## Evidence

Logs/exit codes outside the repo in the tool-result-status-2026-11-16 validation
folder. Preliminary full:6069PASS/14SKIP before the final event-label guard.
Final source review is separately hashed; reviewer did not run tests or a model.

Final review: independent Ken review, six exact hashes, no blocking findings.
Design-only busy-wait wording aligned with the implementation/docs50ms timeout.
Validation final:95focusedPASS; full6072PASS/14SKIP/0FAIL/11existing warnings139.06s.
