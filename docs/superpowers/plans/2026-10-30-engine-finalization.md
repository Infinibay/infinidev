# Engine Finalization Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Stop false task completion, make final corrections cancellable, and
remove duplicated closure/protocol decisions.

**Architecture:** Typed guardrail outcomes feed one completion boundary. A legacy
string adapter remains; runtime configuration lives in ExecutionContext. Existing
tool transcript helpers own provider formatting, and the user-message queue owns
stable event identities and synchronized pending snapshots.

**Tech Stack:** Existing Python/Pydantic/pytest and standard library only.

**Spec:** `docs/superpowers/specs/2026-10-30-engine-finalization-design.md`

## Global Constraints

- Preserve previous changes, public `LoopEngine.execute` and `apply_guardrail`.
- Use `uv run --frozen --no-sync`; do not commit or publish.
- Do not change schemas, permissions, model target or checkpoint version.
- A budget, rejected validator, exception or empty plan cannot prove success.

## Review Focus

- A corrected final candidate is validated before reporting retry/token exhaustion.
- Cancellation wins over positive late validation and stops later calls in a batch.
- Pending instructions are durable but still delivered urgently with their images.
- Duplicate literal texts remain distinct; snapshot plus drain never duplicates an event.
- Native reasoning and manual tool transcripts obey existing provider contracts.

### Task 1: Typed guardrail evaluation with a legacy string adapter

**Files:** `src/infinidev/engine/loop/guardrail_runner.py`;
new `tests/test_guardrail_outcomes.py`.

**Interfaces:** `GuardrailOutcome(text, status, reason)`;
`evaluate_guardrail(ctx, result, *, hooks=None, build_context_messages=None,
run_tool=None, is_cancelled=None) -> GuardrailOutcome`;
legacy `apply_guardrail(...) -> str` keeps its current signature.

- [x] Write RED tests for acceptance/rejection, exceptions, final revalidation,
      token/tool/round fuses, cancellation before/after validation/model/tools,
      and unchanged legacy text/configuration behavior.
- [x] Implement outcomes and context-owned configuration; use a shallow context
      copy in the legacy adapter while keeping shared runtime state/counters.
- [x] Preserve context/caches and accounting; run outcome/extraction regressions.

### Task 2: One honest, interruptible completion boundary

**Files:** `engine.py`, `user_message_injector.py`, `user_guidance.py` in the loop
package; new `tests/test_engine_finalization.py`; migrate private API consumers
in existing termination/context tests.

**Interfaces:** `_validate_final_result(ctx, result) -> GuardrailOutcome` and a
single private finalization operation; `has_pending`, `retain_pending(state)`;
stable `message_id` on injected and retained user guidance, defaulting for old state.

- [x] Reproduce false `done` on validation failure/cancellation and empty plan;
      reproduce a late user message being bypassed by closure.
- [x] Replace Queue polling with a lock-protected deque, idempotent retention by
      event ID and nondestructive pending snapshots. Test duplicates and durable input.
- [x] Route explicit completion through typed evaluation; map non-success statuses
      honestly and retain pending guidance before another iteration/resource stop.
- [x] Replace implicit plan-empty completion with one confirmation notice per streak.
- [x] Wrap corrective tools in normal foreground/cancellation scope with finally;
      migrate all discovered private call sites and run related tests.

### Task 3: Reuse canonical provider transcripts in final corrections

**Files:** `guardrail_runner.py`; new `tests/test_guardrail_protocol.py`.

**Interfaces:** Reuse `ToolRunner.append_assistant_message`, `ClassifiedCalls`,
`LLMCallResult`, existing content normalization and manual tool parser.

- [x] Write RED tests for required reasoning echo, structured content/arguments,
      manual calls/results, all native result IDs, trailing tools after completion
      and an empty completion that must not loop repeatedly.
- [x] Remove duplicate assistant serialization; retain provider objects unchanged.
- [x] Finish every tool-call block, skip post-terminal/over-budget tools, and keep
      usage, cancellation and correction budgets coherent.
- [x] Run related suites, independent review and `uv run --frozen --no-sync pytest`.

## Execution record

- Builder check: nothing important left.
- Baseline: 136 related tests passed in 2.62 seconds.
- Pre-flight: Task 2 consumes Task 1's outcome; Task 3 changes only transcript
  transport inside the same evaluator. No conflicting return types remain planned.

- Task 1 RED→GREEN: rejected/malformed validation, exceptions, cancellation and
  final candidate validation. Legacy adapter leaves caller configuration unchanged.
- Task 2 RED→GREEN: late input durability, cancellation scopes, explicit completion
  and guidance consumed during correction still requiring Task reassessment.
- Task 3 RED→GREEN: reasoning echo, structured/manual protocols, skipped result
  IDs and empty terminal calls. Reused canonical assistant serialization.
- Related suites after implementation: 338 passed in 4.23 seconds.
- First full run: 5450 passed, 13 skipped, 11 pre-existing warnings in 183.84s.
- Independent review: two important findings, both reproduced RED and fixed:
  worker forwarded guidance bypassed local-injector closure checks; explicit
  corrective blocked status was overwritten by output-format acceptance.
- Added a nondestructive TeamRuntime guidance snapshot with real SQLite cursor
  coverage. It keeps collaborator evidence and image delivery owned by normal poll.
- Explicit blocked/explore correction now remains blocked; default continue is a
  candidate proposal and still requires final validator acceptance.
- Deferred review behavior: prior rounds' context/cache/evidence-gate changes were
  outside this round; preserved and covered by related/full suite regression runs.
- Final related suites including team runtime: 389 passed in 5.63 seconds.
- Final full suite after review fixes: 5457 passed, 13 skipped, 11 pre-existing
  Pydantic serializer warnings in 182.77 seconds; process exit 0.
- Added 47 regression cases this round; no commits, pushes or dependency changes.
- `git diff --check` passed for the changed tracked runtime/test paths.
