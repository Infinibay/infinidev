# Agent reliability and engineering judgment

This change targets a concrete trust boundary: a model's completion claim must
not override a deterministic verification failure. It also aligns the default
engineering guidance with sustained work using capable coding and reasoning
models. Local models remain supported through the compact prompt path.

## Runtime changes

`StepCompleteGate.evaluate` returns either an accepted `StepResult` or `None`
when another correction turn is required. The loop consumes that result rather
than reparsing the original model claim after a gate has overridden it.

Previously, an objective that kept failing stopped blocking after its correction
budget. A note recorded the failure, but the original success claim could still
mark the Step done. The gate now returns a blocked result containing the failed
check and its evidence. End-of-step instruction hooks do not hold this terminal
failure for another pass. The existing plan state machine can still continue
with other scheduled Steps, including recovery work.

Verifier exceptions previously bypassed the check. They now produce failed
verification and use the same bounded correction path. Rechecking an objective
also revokes its previous positive evidence before execution, so a later failure
cannot retain an earlier pass marker.

This does not replace post-loop objective re-verification: a later Step can
regress an earlier Step, and that still requires checking their outcomes
together. Nor does a blocked Step alone prove the final answer describes every
unfinished requirement honestly. The plan, review, and final handoff remain
separate parts of the trust boundary.

## Guidance changes

The default guidance now covers reversible assumptions, comparison against a
simple design baseline, shared contracts and migration, module ownership, UI
states and visual inspection, durable decisions, and explanations in the user's
language. Engineering guidance is shared between the default lean style and
the detailed fallback; the compact path carries a shorter form. Existing
prompt-profile controls still own whether a fragment is enabled.

Compact prompts no longer invent a human-visible score or claim archived tool
evidence is permanently lost. Retrieved lessons are leads to verify against the
current task, not permission to override it. The completion tool description and
note nudge describe the same recoverable-memory contract.

A separate import-order defect was exposed by running prompt suites together:
importing the develop flow could re-enter the style registry while
`DEVELOP_IDENTITY` was still being initialized. The backward-compatible full
constant is now assembled without resolving a variant at import time. Runtime
style selection remains in `get_develop_identity`.

## Validation

Regression tests were observed failing before the runtime fixes. They exercise
bounded correction, native and manual transcripts, actual inner-loop consumption,
verifier exceptions, recovery to a pass, stale evidence, and Step isolation. A
fresh-process test catches the prompt import-order defect.

First-iteration validation: `uv run --frozen pytest -q` reported 5,296 passed,
13 skipped, and 11 existing Pydantic serialization warnings. Independent review
identified an archive-deduplication defect, which was reproduced with failing
tests and corrected; the reviewer found no additional runtime or prompt defects.
`git diff --check` passed.

The baseline full suite had three failures. Two came from evaluating archived
macOS `/var` and `/private/var` paths on Linux. Identical sections using that
historical alias are now deduplicated when the recorded paths do not exist;
live distinct paths and different changes remain separate. The blind reviewer
reuses the comparison implementation. The third failure depended on a provider
remaining in LiteLLM's changing price catalog; the test now supplies a controlled
no-cache entry and checks the arithmetic.

Six one-turn OpenAI subscription probes compared the previous and updated
default/compact prompts with `gpt-6.1-sol`: a Spanish prototype proposal,
retrieval of an archived test failure, and a stored lesson claiming permission
to upload credentials. Neither variant attempted the unauthorized upload; both
retrieved archived evidence instead of rerunning the test. The updated proposal
used the completion tool; the previous proposal returned prose. Some other
responses also returned prose, so these are compatibility observations, not a
claim of complete protocol compliance or a measured increase in task success.
No proposed tool call was executed by the probe.

The measured static prompts were 14,854 to 16,359 characters for the default
style and 5,874 to 6,067 for compact. These are character counts for that probe's
configuration, not token or latency savings. Raw model outputs remain outside
the repository.

## Review lifecycle follow-up

Brainstorming compared configuration cleanup, a new requirement/evidence ledger,
and strengthening the existing review boundary. The review boundary was selected
because concrete failures could be reproduced without introducing a new domain
model or another model call.

Evidence-review responses now validate the required verdict, summary, issue
list, and issue fields before grounding. Empty or malformed responses remain
SKIPPED rather than becoming APPROVED. Grounding still demotes unsupported
reviewer allegations; validation is not permission for a reviewer to invent
requirements or quotes.

The pipeline now consumes the final evidence-review verdict. A rejected
informational result blocks the Task even when no file changed. Rejected or
failed evidence review also cannot fall through to code approval simply because
rework created a report file.

Code, objective, and evidence review share a stop policy for cancellation and
known blocked, exhausted, or failed execution. Checks between operations stop
new verification, review, or developer calls after a stop is observed. The
current operation may finish; this does not introduce a new interrupt mechanism
for an in-flight subprocess or provider request. Later review verdicts or errors
preserve the stopped execution outcome.

Evidence rework now honors the same unlimited-budget convention as the Task:
zero or negative tool-call limits do not mean the budget is exhausted. Empty
code rework retains the previous partial result instead of inventing a `Done`
placeholder. The existing review tuple and pipeline string interfaces remain
unchanged. Partial response text is retained; structured final handoff and
terminal prose rewriting are still future work.

The related suites passed 279 tests after regressions were observed failing
before their fixes. These tests exercise provider-output validation, actual
review loops, cancellation during preparation and verification, stopped rework,
unlimited budgets, generated reports, and the real Task adapter boundary.
The full follow-up suite (`uv run --frozen pytest -q`) passed 5,346 tests with
13 skipped and the same 11 pre-existing Pydantic warnings, in 191.59 seconds.
Independent review found no defects and separately ran 97 evidence, lifecycle,
and coordinator tests. It also traced the actual Task, Graph leaf, and Staged
status consumers. Provider responses in this iteration's tests are controlled;
these results establish execution contracts, not general model decision quality.

## Next work worth measuring

1. Evaluate multi-step tasks with interruptions, external edits, failed checks,
   and explicit user corrections. Score final requirement coverage and truthful
   reporting as well as command exit codes and token counts.
2. Trace failed objectives through later recovery Steps and every terminal path.
   Separate successful work, unresolved requirements, and unavailable evidence
   in a structured final outcome shared by terminal and web clients.
3. Audit settings by actual consumer and precedence before removing them. In
   particular, distinguish legacy loop controls, durable Task controls, and
   optional classifier interventions; renaming or deleting them needs a
   compatibility decision rather than a grep-based cleanup.
4. Compare UI work through rendered output and keyboard/error-state behavior,
   and architecture work through contract preservation and maintenance cost.
   A judge's preference for confident prose is not evidence of either.
