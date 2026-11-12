# Task latency repair plan

User authorized fixing the seven-minute edit and pushing the reviewed changes to main.
Preserve prior uncommitted session work; exclude personal .agents configuration and runtime data.

## Approach

Measure the real TUI/provider boundary in a temporary ~/Proyects workspace, with the same
local Strata model and reasoning setting before/after, and no concurrent test suite.
Attribute startup probes, policy classification, developer rounds, review and hidden summary.
Repair demonstrated unnecessary work at its owner; preserve write-authority boundaries,
verification, continuity and cancellation. No alternate engine selected to make a benchmark win.

## Steps

- [x] Capture baseline timings, provider calls/usage and final artifact correctness.
- [x] RED/GREEN: explicit Spanish edits must retain literal modify authority and start a
  delivery Step, while explanations, quoted examples and read-only requests remain read-only.
- [x] RED/GREEN: optional selected-model classification also returns requested delivery intent
  (implementation/informational), independently of literal permissions. Legacy profiles remain
  readable; explicit read-only constraints take precedence. User explicitly requested this.
- [x] RED/GREEN: a compact factual handoff must not require an additional synthesis request;
  larger handoffs retain optional model synthesis and deterministic failure fallback.
- [x] Inspect remaining measured overhead and fix additional confirmed causes as needed.
- [x] Repeat the same live task and follow-up; compare calls, wall time and correctness.
- [x] Run focused tests, independent review and the full suite.
- [ ] Review all included changes, commit, integrate and push main without force-pushing.

## Baseline observations

Initial live trace: 3 capability probes plus policy-classifier call take 51.7 s before
execution; request is classified with answer authority and becomes an investigation.
Complete trace and raw provider request payloads are in
/home/andres/Proyects/infinidev-latency-6wfumwfu (not committed).

That initial trace is unsuitable for end-to-end comparison: instrumentation wrote request
dumps inside the workspace, contaminating tracked file changes and reviewer input. Its early
classification/probe observations remain diagnostic, but its total latency is not evidence
of product latency. The clean repeat stores all instrumentation outside the workspace under
/tmp/infinidev-latency-evidence. The isolated baseline source snapshot restores the three
latency-change modules from HEAD while retaining the first-pass continuity improvements.

## Clean experiment and follow-up

- Same Strata model, THINKING_ENABLED=true, identical five-field JSON and seeded agreements;
  fresh process/workspace per edit, no concurrent inference or full test suite. This is a
  diagnostic single-run comparison, not a statistically controlled throughput benchmark.
- Clean baseline: 458.8383 s, 19 LiteLLM boundary calls, 2 developer Steps, 9 regular tools;
  49.8779 s was approval waiting. Two step-summary calls cost 48.7348 and 56.4853 s;
  final hidden synthesis cost 34.8291 s. Server metrics confirm retry work behind the
  supposedly 30-second optional summaries.
- First candidate: 212.6334 s, 11 boundary calls, 1 Step, 7 regular tools. Both optional
  summary phases returned without provider calls. JSON preserved the other four fields.
- Resumed real TUI follow-up: recalled Faro Azul, SQLite, no login, 50 minutes, Wednesday;
  second message correctly explained the one-line change and Python JSON verification.
  Both messages used only one chat completion each, without file tools.
- That candidate exposed a mandatory small-model note reminder after two reads even though
  ~11k prompt tokens were far below the actual 262144-token capacity. Reuse the existing
  context-pressure predicate for this reminder; preserve the nudge for compact windows and
  actual pressure. RED confirmed against HEAD, GREEN 150 termination/transcript tests.
- Reviewer found accented cambiá still incurred classification/elaboration because the
  shared execution-action detector omitted it. Added it and a non-masking accented regression.
  Reviewer found no further defects in the latency changes or the note-pressure delta.
- Strata metrics during this session report variable decode/prefill rates below the user's
  reported peak. Provider request times agree with client observations; do not attribute all
  remaining latency to orchestration or promise 90 TPS end-to-end.
- A further repeat took 283.9241 s: the model attempted Git in this non-repository fixture,
  and the engine reopened completed work because final_answer omitted a verification command
  already present in evidence_summary. The second iteration repeated orientation and checks.
  This is a wording gap, not missing work. Make that check a pure diagnostic and preserve
  terminal status, final answer and recorded evidence. Existing substantive edit, Step-effect,
  guardrail and verification checks remain authoritative.
- That repeat also exposed another note-reminder owner in iteration prompt rendering. Pass
  actual context capacity to it and use the same pressure predicate; preserve the historical
  compact/unknown-capacity fallback. Regression covers the real prompt builder wiring.
- RED/GREEN for both additional causes: 172 focused termination/context/guidance tests;
  six note-rendering tests after adding the caller-level regression.

## Final live observation and limits

The last repeat completed correctly in **608.0988 s**, with **13 boundary calls**, one
developer Step and seven regular tool calls. Its final reviewer alone took **441.4935 s**
and generated 7485 tokens; developer completion occurred at trace t=169.348. Compact handoff
generation took 0.0009 s and no provider call. The omitted-command diagnostic fired without
reopening the completed Step. All five JSON fields match the independently asserted result.

This does **not** establish an end-to-end speedup for the final change. Earlier complete
candidate latency was 212.6334 s, but elapsed-time comparisons are confounded by changing
backend expert placement/cache state and memory pressure. The user confirmed that Strata
moves experts dynamically and expects throughput to improve as it settles. No shared model
service or configuration was changed or restarted. Read-only diagnostics observed active
swap I/O and substantial model reads; they do not justify tuning backend allocation here.

The review request still inherits enabled thinking with no explicit output cap; that is a
remaining latency lead, not a fixed or measured-safe policy change. Main-task reasoning was
preserved. The supported result of this pass is elimination of demonstrated orchestration
rounds and correctness/continuity preservation, not a stable local latency guarantee.

Two test runners briefly overlapped after the automatic benchmark completion trigger and a
manual launch. Both runs were stopped and their shared output discarded; final verification
uses a single fresh run and a separate /tmp/infinidev-all-fixes-final.log.

Final verification: **5616 passed, 12 skipped, 11 existing provider serializer warnings**
in 210.80 s (`uv run pytest -q`, exit 0). `git diff --cached --check` passed. Independent
review found no remaining defects after the accented-request correction and reviewed the
note-pressure and non-blocking handoff changes separately.
