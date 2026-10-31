# Guardrail Code Quality Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Separate final acceptance policy from correction mechanics while
preserving the engine's observable behavior.

**Architecture:** The existing public evaluator owns validation; one private
CorrectionSession per evaluation owns model/tool interaction and counters. Typed
private correction results represent proposals or stops, never Task success.

**Tech Stack:** Existing Python, dataclasses and pytest only.

**Spec:** `docs/superpowers/specs/2026-10-30-guardrail-code-quality-design.md`

## Global Constraints

- Work in the existing authorized checkout; preserve all earlier local changes.
- Use `uv run --frozen --no-sync`; no dependencies, commits, pushes or remote calls.
- Keep public evaluator/legacy signatures, monkeypatch seams and status semantics.
- Keep 8 rounds, cumulative 16-tool unbounded fuse, global counters and final revalidation.
- No prompt, profile, router, schema, main-loop or persistence changes.

## Review Focus

- Tool quotas must accumulate across retries but reset for a new evaluation.
- Per-action quotas must reset while LoopState accounting continues.
- A final accepted candidate can follow a tool-budget stop in the same batch.
- Exceptions/cancellation after a partial correction must preserve its text and counters.
- Native/manual messages, hooks, cache refresh and skipped call IDs keep their order.

### Task 1: Characterize correction lifecycle contracts

**Files:** extend `tests/test_guardrail_outcomes.py` using its actual LoopState
fixture and public evaluator. Save before/after fixtures outside the worktree.

**Interfaces:** Existing `evaluate_guardrail` and `GuardrailOutcome` only.

- [x] Add cumulative unbounded budget coverage across several corrections in
      FC and manual mode; 16 executions, 3 contexts, 5 model calls and exhausted.
- [x] Cover a new evaluation renewing that fuse while runtime accounting persists.
- [x] Cover per-action budget reset across two corrections (one tool each).
- [x] Cover accepted final completion after an over-budget call and model-round
      exhaustion after exactly 8 requests/tools.
- [x] Cover a later tool exception retaining the previous candidate and accounting.
- [x] Run characterization against existing implementation. Expected: all pass.
      Prove the cumulative/round checks catch a mutant in a separate process.

### Task 2: Give correction mechanics one owner

**Files:** `src/infinidev/engine/loop/guardrail_runner.py` only.

**Interfaces:** private immutable `_CorrectionResult(text, status, reason)`;
`_CorrectionSession(ctx, *, hooks, build_context_messages, run_tool, is_cancelled)`;
`run(result: str, feedback: Any, *, attempt: int) -> _CorrectionResult`.
Public evaluator and `apply_guardrail` remain unchanged.

- [x] Implement the private session with context building, manual/native call
      normalization, batch handling, tool execution/tracking/cache and quotas.
- [x] Move correction mechanics out of the public policy; use early returns for
      bool/invalid input and explicit candidate/blocked/exhausted/cancelled handling.
- [x] Preserve existing helper patch points; remove redundant state/config args
      from the now-owned private helpers rather than add compatibility wrappers.
- [x] Run the full characterization and existing guardrail/finalization tests.
- [x] Compare deterministic scenario outputs against saved baseline, and record
      AST line/control-flow/nesting changes as structural evidence only.
- [x] Run independent review and `uv run --frozen --no-sync pytest`.

## Execution ledger

- Builder check: nothing important left.
- Baseline: 175 related tests passed in 3.05s.
- Before snapshot: `/tmp/infinidev-guardrail-quality/before.py`.
- Pre-flight: Task 1 exercises unchanged public contracts consumed by Task 2.
- Ruling: this is a behavior-preserving refactor. Characterization tests must
  pass before code moves; mutation checks establish sensitivity. No source-shape
  assertion or artificial new behavior is introduced solely to create RED.

- Task 1 complete: seven added characterization cases pass before refactoring.
  Cumulative-budget mutation produced three expected failures; a 9-round mutant
  produced one expected failure. No source mutation persisted in the checkout.
- Task 2 implementation complete: correction session owns mechanics; no existing
  tests/mocks migrated or weakened. Public aliases and legacy adapter preserved.
- Ruling: add private not_started result for prompt budget denied before an
  attempt. A new regression caught an extra validation that could falsely accept
  unchanged output; watched FAIL(done) then PASS(exhausted). This preserves the
  previous policy rather than introduce a new retry behavior.
- Direct suites after refactor: 183 passed in 3.23s.
- Differential comparison: 648 deterministic scenarios, zero differences in
  outcomes, validator/tool sequence, counters, contexts, schemas and transcripts.
- Structural result: module 406 to 382 lines; evaluator 253 to 68 lines, control
  flow nodes 45 to 19, nesting 9 to 4. Largest correction method is 59 lines.
  These are structure measurements, not performance or correctness proofs.
- Broader related suites: 932 passed in 15.21s.
- Independent review: no Critical, Important or Minor findings. Reviewer ran
  40 outcome/protocol tests and independently compared 624 exceptional/cancellation
  scenarios, finding zero differences in outcomes, callback sequence, cancellation
  checks and tool counts.
- Final: Ruling: live context reads versus prior snapshots respect the documented
  immutable-after-setup config and actual engine ownership; speculative rebinding
  callbacks do not justify extra snapshot state.
- Final: Ruling: exceptions after terminal parsing retain the previous candidate,
  and cancellation aborts remaining acknowledgements as before. Preserve those
  semantics in a structural refactor rather than silently changing stop policy.
- Final: Ruling: broader protocol/loop consolidation and older dirty changes are
  outside this round; private session/result already isolate the diagnosed concern.
- Final full suite: 5485 passed, 13 skipped, 11 pre-existing Pydantic warnings
  in 186.71s; process exit 0. Eight added regression/characterization cases.
- Final diff/line-length checks passed. No new dependencies, prompt changes,
  commits or pushes. Only the guardrail implementation and focused tests changed
  in production/test code for this round.
