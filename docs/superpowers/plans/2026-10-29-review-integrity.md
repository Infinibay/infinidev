# Review Integrity Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for implementation.

**Goal:** Preserve honest review outcomes and stop further work after cancellation or failure.

**Architecture:** Validate the existing evidence-review JSON contract, share the
review stop policy in the existing analysis module, and propagate verdicts at
the pipeline boundary. Existing tuple and string return interfaces stay stable.

**Tech Stack:** Python, pytest, existing LiteLLM and engine collaborators.

**Spec:** `docs/superpowers/specs/2026-10-29-review-integrity-design.md`

## Global constraints

- Preserve earlier uncommitted work and use `uv run --frozen`.
- No new dependency, settings, persistence schema, push, or release.
- Keep missing reviewer availability distinct from an approval.

## Review focus

- Malformed issue objects must not disappear into an approval.
- Cancellation during an operation stops the next operation.
- Reviewer verdicts must not replace failed or cancelled rework outcomes.
- A generated report must not let failed evidence review fall into approval.
- Zero and negative Task budgets remain unlimited.

## Task 1: Validate evidence-review input

Files: `analysis/evidence_review.py`, `tests/test_evidence_review.py`.

- [x] Add failing tests using the real reviewer and controlled provider output.
- [x] Validate required verdict, summary, issues, and issue fields before grounding.
- [x] Verify valid approvals, grounded rejection, and allegation demotion remain functional.

## Task 2: Respect stopped rework

Files: `analysis/review_engine.py`, `analysis/evidence_review.py`, focused tests.

Interface: `_review_execution_stopped(engine: Any) -> bool` owns the known stop
labels and cancellation flag; it performs no mutation or I/O.

- [x] Reproduce review/verification continuing after stopped developer execution.
- [x] Guard the next operation throughout code, objective, and evidence review.
- [x] Reproduce zero-budget evidence rework suppression; treat nonpositive budgets as unlimited.
- [x] Run related review and objective suites.

## Task 3: Preserve review outcomes at the pipeline boundary

Files: `orchestration/pipeline.py`, evidence/review/adapter tests, reliability docs.

- [x] Reproduce discarded evidence rejection and cancellation outcome overrides.
- [x] Propagate rejection while preserving stopped execution; return on failed evidence review.
- [x] Verify actual Task adapter propagation and existing Graph behavior.
- [x] Run full suite, independent review, and diff checks; document results.

## Execution record

The previous iteration's full suite passed 5,296 tests. A fresh focused baseline
is running in `/tmp/infinidev-review-baseline.log`. Semantic Ken discovery is
unavailable because its configured embedder requires torch; read the ranked
tests and actual code instead. Ken CLI recall and blast-radius reads worked.

- Fresh focused baseline: 183 passed.
- Eight malformed review cases reproduced false approval before the parser fix.
- Fifteen lifecycle/budget cases failed before stop and unlimited-budget fixes.
- Twelve pipeline/adapter cases failed before verdict propagation was corrected.
- Nine preparation/empty-rework cases failed before the final boundary fixes.
- Focused final run: 279 passed in 4.77 seconds. Full suite: 5,346 passed,
  13 skipped, 11 pre-existing Pydantic warnings in 191.59 seconds. Output:
  `/tmp/infinidev-review-integrity-full.log`.
- Ruling: preserve the previous partial text when rework returns empty, rather
  than generating a `Done` placeholder. This was an existing inconsistency with
  the other two rework paths and part of preserving honest outcomes.
- Ruling: cancellation is checked between operations; an in-flight provider or
  verification operation may still finish. This iteration does not replace their
  transports or process cancellation mechanisms.
- Independent review: no Critical, Important, or Minor findings. The reviewer
  traced real Task, Graph leaf, and Staged consumers and independently ran
  evidence/lifecycle/coordinator suites: 97 passed in 3.02 seconds.
- Review scope decisions: retain the documented limits on in-flight cancellation
  and user-facing prose. Preserve partial text and rely on the normalized status
  for the existing clients. Earlier changes retain their prior review plus this
  full-suite run; no claim about live provider behavior is based on controlled
  response tests. No new live calls were needed for these deterministic fixes.
- Final whitespace checks passed. Dependency lockfile is unchanged. Changes stay
  uncommitted on `improve/agent-judgment-and-verification`.
