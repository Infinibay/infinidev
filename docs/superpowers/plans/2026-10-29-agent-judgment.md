# Agent Judgment and Verification Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Preserve honest completion outcomes and improve guidance for sustained engineering work.

**Architecture:** The completion gate owns its typed result; the engine consumes it without reparsing an overridden success claim. Prompt changes use existing composition and profile boundaries.

**Tech Stack:** Python, Pydantic, pytest, existing LiteLLM transport.

**Spec:** `docs/superpowers/specs/2026-10-29-agent-judgment-design.md`

## Global constraints

- Use `uv run` and existing dependencies, permissions, and prompt profiles.
- Preserve local models and existing gate entry points.
- Leave changes reviewable on the working branch; no push or release.

## Review focus

- A failed check after an earlier pass must revoke stale positive evidence.
- A verifier exception must not accept the model's success claim.
- A new Step must not inherit the previous Step's forced failure.
- End-of-step hooks must not hold an already terminal verification failure.
- Compact prompts must preserve recoverable evidence and ordinary autonomy.

## Task 1: Typed completion decisions

Files: `loop/step_complete_gate.py`, `loop/engine.py`, settings documentation,
`tests/test_objective_verification.py`, `tests/test_loop_termination.py`.

Interface: `StepCompleteGate.evaluate(...) -> StepResult | None`, with the same
arguments as `blocks`. Existing boolean methods remain available.

- [x] Add regression tests for exhaustion, verifier errors, stale evidence,
  recovery, Step isolation, and actual inner-loop consumption; observe failures.
- [x] Implement the gate decision and consume it in the engine; retain model
  arguments unchanged and bypass user hooks on terminal verification failure.
- [x] Run objective, termination, user-hook, and scope suites: 220 passed.

## Task 2: Engineering guidance and documentation

Files: `engine/loop/prompt/text.py`, `prompts/variants/lean.py`, prompt tests,
`README.md`, and a concise audit note under `docs/`.

- [x] Run existing composition tests and live before/after behavior probes.
- [x] Align default and compact prompts with the existing engine; remove invented
  scoring pressure, preserve user authority, and make communication actionable.
- [x] Update the README and document rationale, validation, and future work.
- [x] Run prompt suites, the full repository suite, and diff checks.
- [x] Obtain an independent review and address substantive findings.

## Execution record

- Baseline objective/recovery suites: 52 passed.
- Full baseline suite started before product edits; output in
  `/tmp/infinidev-baseline-tests.log`.
- Working branch: `improve/agent-judgment-and-verification`.
- The MCP code search did not respond; Ken CLI recall and blast-radius reads
  succeeded. Only `.agents/` was untracked before this work; leave it untouched.
- Prompt validation: 778 passed, including a new subprocess import-order test.
- Ruling: use existing composition checks plus live scenario probes instead of
  tests that only assert prose contains chosen words. The latter would detect
  wording changes without establishing better model behavior.
- The completion-schema memory description was aligned because it otherwise
  contradicted the corrected system guidance.
- Ruling: fix the baseline benchmark failures within the user's broad maintenance
  request. Preserve archived macOS path semantics across hosts, reuse the existing
  diff implementation, and isolate price arithmetic from a changing catalog.
  No provider pricing behavior was changed.
- Live subscription probes: six completed requests; observations and limits are
  documented in `docs/AGENT_RELIABILITY.md`.
- Independent review found an overly broad archive-alias normalization. New
  failing tests reproduced distinct-change and live-path loss; the correction
  limits archive deduplication to identical missing-path sections.
- Independent reviewer found no further runtime or prompt defects and separately
  ran 40 focused tests. Final benchmark regressions: 44 passed. Task-level
  recovery after a blocked Step remains an explicit existing behavior, not a
  guarantee that the model's final prose covers all unresolved requirements.
- Final `uv run --frozen pytest -q`: 5,296 passed, 13 skipped, 11 pre-existing
  Pydantic serialization warnings, in 187.09 seconds. Full output:
  `/tmp/infinidev-verified-tests.log`. `git diff --check` passed.
- Kept the existing dependency lockfile; uv's automatic lock-format rewrite is
  excluded from the changes. Work remains uncommitted on the working branch.
