# Engine Context Continuity Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Preserve literal guidance, recoverable context and honest snapshot
provenance through the main engine's Step, resume and correction boundaries.

**Architecture:** Keep LoopEngine as coordinator. User guidance becomes typed
Task state; the existing prompt builder projects it. Resume marks observations
as historical, and guardrail corrections reuse the normal context constructor.

**Tech Stack:** Existing Python/Pydantic, standard library, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-29-engine-context-continuity-design.md`

## Global Constraints

- Preserve existing local changes and `StepCompleteGate.evaluate` integration.
- Keep current checkpoint version, 2 MiB capacity, public return types and budgets.
- Never persist image payloads or summarize literal user guidance.
- Retain guidance within the Task; fresh Task states start empty.
- Use `uv run --frozen --no-sync`; no commits or publication.

## Review Focus

- Every drain route must preserve guidance once, including repeated identical text.
- Late images must be user turns without duplicate tool-result IDs; no-vision
  models receive explicit references without pretending to see image pixels.
- Checkpoint compaction must retain literal guidance or reject the checkpoint.
- Resume retains cached bodies but removes the claim that they are current;
  actual read/refresh clears only the updated entry's revalidation flag.
- Guardrail context must be lazy, fresh per correction, protocol-safe, and keep
  existing tools, usage, budgets and fallback behavior.

### Task 1: Preserve user guidance through prompts and image delivery

**Files:** Create `src/infinidev/engine/loop/user_guidance.py`; modify
`loop_state.py`, `user_message_injector.py`, `context.py`, `context_builder.py`,
and the checkpoint calls in `engine.py` under the same loop package.
Create `tests/test_engine_user_guidance.py`.

**Interfaces:** `UserGuidance(text: str, attachment_paths: tuple[str, ...] = ())`;
`LoopState.user_guidance: list[UserGuidance]`;
`UserMessageInjector.drain_for_state(state) -> list[InjectedUserMessage]`;
`render_user_guidance(guidance) -> str`. Preserve `drain() -> list[str]`.

- [x] Write failing tests: boundary/mid/late guidance survives a fresh Step prompt,
      chronological duplicates remain, compact profiles retain literal guidance,
      and boundary/late images are delivered with vision or explicit text fallback.
- [x] Confirm failures before modifying production code.
- [x] Add typed state, shared structured drain/retention and mandatory projection.
- [x] Preserve image metadata separately from payloads and use the existing
      image-content helpers for delivery at each boundary.
- [x] Checkpoint delivered guidance before the next model call, including late
      completion rejection, without checkpointing on an empty drain.
- [x] Run new tests plus extraction, carryover, multimodal and manual parity tests.

### Task 2: Preserve literal instructions and observation provenance on resume

**Files:** Modify `resume_checkpoint.py`, `opened_file.py`, `loop_state.py`,
`context_builder.py`, `context.py`, and the real shell-cache refresh in
`tool_runner.py`. Create `tests/test_engine_context_provenance.py`.

**Interfaces:** Add `OpenedFile.requires_revalidation: bool = False`; older states
use defaults. `UserGuidance` from Task 1 is an essential checkpoint field.

- [x] Write failing tests for essential-field compaction and irreducible guidance,
      old-state compatibility, retained-but-marked resume snapshots, and clearing
      the mark only after a real entry refresh.
- [x] Observe the failures, then preserve guidance and mark restored observations.
- [x] Render honest labels and allow revalidation; keep content, pinned and TTL.
- [x] Run resume/cache/prompt tests and the Task 1 regressions.

### Task 3: Give guardrail corrections their owning Task context

**Files:** Modify `guardrail_runner.py` and `engine.py`.
Create `tests/test_guardrail_context_continuity.py`.

**Interfaces:** Add optional keyword `build_context_messages:
Callable[[], list[dict[str, Any]]] | None = None` to `apply_guardrail`.
Keep `LoopEngine._apply_guardrail(...) -> str` unchanged externally.

- [x] Write failing tests for Task/notes/history/guidance in fallback correction,
      callback invocation per rejected attempt, successful-validation laziness,
      retained caller messages, and real LoopEngine callback wiring.
- [x] Observe failures, then build correction context lazily and append feedback
      after it; use the normal renderer for standalone consumers.
- [x] Run guardrail, engine-coordinator, termination and continuity tests.
- [x] Obtain an independent review of the combined change; fix material findings.
- [x] Run `uv run --frozen --no-sync pytest` and inspect the final diff.

## Execution record

- Design builder check: nothing important left.
- Initial baseline: 148 tests passed.
- Pre-flight: Task 2 serializes Task 1's model; Task 3 reuses Task 1's prompt
  projection. These share the existing LoopState and prompt-builder interfaces.
- Task 1 RED: eleven regressions reproduced lost text/images and missing durable
  delivery. GREEN: 59 initial related tests passed.
- Task 2 RED: six cases reproduced omitted essential guidance and absent snapshot
  provenance. GREEN: 88 related tests passed.
- Task 3 RED: five tests reproduced missing correction context/callback behavior.
  Compatibility run also caught partial counter-only state; the renderer uses
  an empty LoopState for context while accounting keeps the original counter.
- Extended related suite: 943 passed before the windowed-notice fix below.
- Final related suite including the three windowed-notice regressions:
  946 passed in 16.20 seconds. Thirty-two new regression/compatibility cases.
- Ruling: retain forwarded literal user guidance inside the receiving worker's
  Task state too — its otherwise-identical rebuild lost instructions; two new
  RED regressions demonstrate the loss. Keep collaborator evidence separate.
- Ruling: update urgent-message prose to match actual plan authority — the old
  text incorrectly declared all Steps user-approved and encouraged early closure.
- Independent review found two material correction handoffs: stale source after
  a corrective tool and worker guidance not polled by the context factory. Both
  were reproduced and fixed using normal cache bookkeeping and injection paths.
- Review fix RED→GREEN coverage includes file edits, shell edits, failed shell
  commands with partial writes, worker text/images and relative sources resolved
  in the agent workspace. Disabling cache bookkeeping in a separate process
  made the two real-source retry tests fail again without editing the checkout.
- Final related suite after review fixes: 951 passed in 16.19 seconds. Thirty-seven
  new regression/compatibility cases overall.
- Ruling: shell reconciliation uses the agent's workspace for relative cache keys
  — a real-file regression observed source from the wrong cwd; existing tool
  context and cwd remain fallbacks for standalone callers.

### Task 4: Consume notices from live state under history projections

**Files:** `context.py`, `context_builder.py`, `tests/test_engine_user_guidance.py`.

**Interfaces:** Optional `build_iteration_prompt(..., pending_messages_state:
LoopState | None = None)`; the history view is read-only, the live state owns
consumption. No external caller must change.

- [x] Reproduce repeated notices with `history_window=1` and two history records.
  Three RED tests covered engine notices, reactive guidance and interventions.
- [x] Pass the actual Task state as the owner of consumable messages; preserve the
  projected history and earlier standalone behavior.
- [x] Run related tests, independent review and the full suite with Tasks 1–3.

## Final validation

- `uv run --frozen --no-sync pytest`: exit 0, 5,410 passed, 13 skipped,
  11 Pydantic serialization warnings, 188.26 seconds.
- Review findings were fixed with real-file/forwarded-message regressions and
  verified by the related suite and full suite. No required finding remains.
- `git diff --check` passed. Existing local changes and `uv.lock` were preserved;
  no commits, merges, dependency changes or publication were performed.
