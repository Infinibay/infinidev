# Review integrity during sustained work

The user requested brainstorming and continued improvements, retaining their
earlier delegation of implementation choices. This design is the implementer's
choice under that authorization. Build on the existing Python engine and review
contracts; preserve the prior iteration's uncommitted changes.

## Brainstorming and selection

Three routes were considered:

1. Audit configuration consumers and retire misleading controls. Valuable, but
   compatibility and precedence need a separate migration decision.
2. Add a durable requirement/evidence ledger. Potentially useful for long tasks,
   but the existing Goal/Stage/Task domain and objective ledger already own part
   of that problem. Another persistence model would be premature.
3. Repair review integrity at the existing boundary. Selected because source
   inspection exposes concrete failures: evidence verdicts are discarded by the
   pipeline, empty reviewer objects can become approvals, a zero unlimited Task
   budget suppresses evidence rework, and stopped rework can start more work.

## Required behavior

- Only a well-formed evidence-review response may reach grounding. Missing or
  invalid verdicts, summaries, issue lists, or issue fields produce SKIPPED,
  never an invented approval. Keep the existing protection that demotes a
  blocking allegation not quoted from the actual response.
- A final evidence rejection must set the engine outcome to blocked, just as a
  final code rejection does. The original partial answer remains available.
- Known stopped execution outcomes (cancelled, blocked, exhausted, failed) and
  the cancellation flag prevent additional review, verification, or developer
  operations. An already running operation may finish; no later operation should
  start after the stop is observed. Preserve the stopped outcome and last text.
- A reviewer verdict or error must not overwrite a cancellation or an existing
  stopped execution outcome. Failed evidence review must not fall through into
  code review merely because its rework created a file.
- `max_total_tool_calls <= 0` retains the existing unlimited Task semantics;
  only a positive exhausted budget prevents evidence-review rework.

## Implementation boundaries

Keep review-loop returns as `(text, review_result)` and `_run_review_phase` as a
string-returning boundary. Share stop-policy logic between the review loops and
the pipeline without importing UI or execution adapters into analysis. Do not
add settings, dependencies, model calls, database formats, or prose classifiers.
Keep approved and explicitly skipped review paths compatible. Terminal text
rewriting and a structured user-facing report remain separate future work.

## Validation

Reproduce each bug with focused tests before changing production code. Exercise
real parser/grounding and review loops with only provider calls, verification,
and developer execution controlled by test fakes. Cover cancellation before and
during operations, failed/blocked rework, positive and unlimited budgets, invalid
and grounded responses, and propagation through the Task adapter. Run the related
suites and full repository suite; obtain an independent review of this iteration.
