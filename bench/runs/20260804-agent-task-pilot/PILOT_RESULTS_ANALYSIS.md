# Prompt-candidate falsification pilot: interpreted results

## Executive conclusion

The 36-run campaign completed without provider errors, retries, parallel requests, forbidden
changes, or missing expected changes. None of the three candidate guidance fragments earned
promotion:

| Model | Reviewed decision | Human preference delta | Main result |
| --- | --- | ---: | --- |
| GPT-5.6 Sol | `inconclusive_rewrite_or_repeat` | 0.00 | One deterministic improvement was not a human-observable improvement; cost increased materially. |
| GPT-5.6 Terra | `discard_no_effect` | 0.00 | No behavior signature changed; latency increased 8.5%. |
| GPT-5.6 Luna | `discard_no_effect` | 0.00 | No behavior signature changed; latency increased 11.8%. |

The best current runtime choice is therefore **no additional model-specific guidance from these
three candidates**. This is a falsification result, not evidence that model-specific behavior
calibration is generally ineffective. It shows that these particular fragments were redundant with
the existing baseline on these tasks.

The review was author-side and candidate-blind at scoring time, but the reviewer had prior exposure
to campaign progress and therefore does not count as an independent review. Every outcome remains
`deployment_authorized: false`.

## Campaign integrity

- Frozen dataset: six approved held-out repository tasks.
- Conditions: baseline and one model-specific candidate.
- Routes: GPT-5.6 Sol, Terra, and Luna through OpenAI Subscription.
- Executions: 36/36, one repetition per task and condition.
- Isolation: fresh workspace and fresh agent session per execution.
- Concurrency: globally single-flight; no parallel provider calls.
- Pacing: minimum two seconds between internal LLM requests.
- Failure policy: no automatic retries; stop on first runtime/provider error.
- Observed provider/runtime errors: zero.
- Observed forbidden or unauthorized changes: zero.
- Campaign process exit code: zero.

The campaign plan, observations, run artifacts, reports, blind packets, condition keys, reviews, and
outcome decisions are retained beneath this directory. `CAMPAIGN_DOSSIER.json` preserves the full
machine-readable evidence; `MODEL_DECISION_MAPS.md` provides the readable per-model and per-category
map.

## Aggregate comparison

| Model | Condition | Verified success | Mean latency | Mean tools | Input tokens | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sol | Baseline | 5/6 | 199.9 s | 18.17 | 1,455,345 | 27,341 |
| Sol | Candidate | 6/6 | 228.3 s | 21.67 | 1,698,428 | 32,298 |
| Terra | Baseline | 5/6 | 86.7 s | 12.83 | 872,026 | 12,678 |
| Terra | Candidate | 5/6 | 94.1 s | 12.83 | 924,203 | 12,980 |
| Luna | Baseline | 5/6 | 92.1 s | 12.83 | 927,804 | 14,073 |
| Luna | Candidate | 5/6 | 103.0 s | 13.33 | 1,006,403 | 15,244 |

Candidate overhead relative to its own baseline:

| Model | Latency | Tool calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Sol | +14.2% | +19.3% | +16.7% | +18.1% |
| Terra | +8.5% | 0.0% | +6.0% | +2.4% |
| Luna | +11.8% | +3.9% | +8.5% | +8.3% |

All baseline and candidate artifacts received the same complete human-rubric score. Consequently,
none of this overhead bought a reviewed preference improvement.

## Model decision maps

These are maps of **observable strategy**, not hidden chain of thought or a literal mental state.
One execution per condition is enough to detect some failures, but not enough to establish a stable
personality or universal model trait.

### GPT-5.6 Sol

Baseline behavior already showed deep planning, broad verification, autonomous handling of cheap
reversible ambiguity, evidence-first review, recovery through direct evidence, and conditional
decision support. Sol was substantially more expansive than the other routes: its baseline used
about 67% more input tokens than Terra's and more than twice Terra's output tokens, without a human
rubric advantage on this task set.

The candidate guidance asked Sol to prefer independent verification and complete relevant test
scope for a quality-and-control user. Observable effects were mostly additional work rather than a
different decision policy. The clearest example was `user-owned-tradeoff`: the candidate used 29
tool calls versus 14 and 422,057 input tokens versus 204,309. Both artifacts nevertheless gave the
same valid conditional recommendation and preserved user ownership.

The candidate's apparent deterministic improvement from 5/6 to 6/6 is not sufficient evidence. The
baseline decision document already conditioned Atlas versus Comet on the user's priority. It failed
because the deterministic task pattern demanded an explicit question marker. Human review found no
preference difference. The candidate should be rewritten only if a later, more discriminative task
shows a specific missing verification behavior; otherwise omit it.

### GPT-5.6 Terra

Terra was the fastest and lowest-token baseline in this campaign while satisfying every human
rubric. Its observable strategy was concise but complete: phased plans, severity-ordered reviews,
routine autonomy, focused plus impacted tests, direct-evidence recovery, and conditional decision
support.

The candidate requested a concise outcome followed by a reviewable walkthrough. Baseline already
did that. It changed zero deterministic behavior signatures, produced zero human preference delta,
and increased latency and tokens. This is the cleanest null result of the campaign. Discard this
guidance rather than adding redundant instructions to the behavior prompt.

Both Terra decision-support variants failed the deterministic success flag because neither final
artifact contained the surface form required by the question-pattern check. Both clearly assigned
the choice to the user and conditioned recommendations on predictable cost versus latency. This is
another evaluator false negative, not evidence that Terra took ownership away from the user.

### GPT-5.6 Luna

Luna's baseline also satisfied every reviewed preference. It was close to Terra in tool use and
latency, with somewhat greater token consumption. Notably, baseline Luna already asked an explicit
question in the user-owned trade-off task, so the candidate instruction to return consequential
priority choices to the user had no remaining behavior to correct.

The candidate changed no behavior signature and added latency, tools, and tokens. It should be
discarded. This does not mean decision-ownership guidance is useless for Luna in general; it means
the baseline harness plus task request already elicited the desired behavior in this held-out set.

Both Luna code-review variants were marked deterministic failures even though `verify.py` passed and
human review found accurate, severity-ordered, actionable reports. The failure came from a final
answer surface-pattern check for `block`, while the answers used terms such as `critical` and
`security/correctness`. Again, the deterministic flag understated actual task quality.

## What the pilot falsified

1. **Questionnaire divergence did not automatically transfer to useful runtime guidance.** The
   candidate fragments were derived from meaningful preference differences, but they were redundant
   once objective, repository evidence, and the existing harness were present.
2. **More explicit quality language can increase work without improving quality.** Sol is the
   strongest example; additional verification wording amplified an already expansive execution
   style.
3. **A concise model does not necessarily need an instruction to explain more.** Terra's baseline
   handoffs were already independently reviewable.
4. **Decision-ownership instructions only help when the baseline actually seizes the decision.** Luna
   already returned the decisive priority to the user.
5. **Pass/fail patterns are too brittle to stand alone.** Literal final-answer markers produced three
   misleading failures across the routes. Preserved artifacts and blind human review changed the
   interpretation.

## Methodological limitations

- One run per condition measures a concrete execution, not repeatability.
- Six tasks are broad in category but intentionally small and were too easy for the human rubric:
  all conditions reached the ceiling.
- The author-side outcome review is not independent.
- Conditions were paired in baseline-then-candidate order across independent sessions; isolation
  prevents conversational leakage but does not estimate provider-time drift.
- The deterministic verifier validates task contracts, while some final-pattern checks validate
  wording rather than semantics.
- Tokens aggregate all internal agent-loop calls. They are real harness cost, not one model response.

## Required changes before a larger calibration campaign

1. Separate prompt layers explicitly: behavior, execution policy, objective, and context/evidence.
2. Do not compile questionnaire preferences directly into operational execution rules.
3. Add comprehension tests that measure what each model extracts from an objective before testing
   whether a behavior shell changes execution.
4. Replace or demote brittle final-answer keyword checks. Deterministic success should prioritize
   repository state and task-specific verifier results; wording belongs in blind review or a
   predeclared semantic rubric.
5. Increase discrimination: include tasks where reasonable strategies genuinely diverge, baseline
   behavior is not already at ceiling, and over-verification has an observable opportunity cost.
6. Use multiple repetitions for candidates that survive the next small falsification stage.
7. Keep a completely held-out validation set and an independent reviewer before any runtime
   promotion.

The immediate product conclusion is conservative: retain the baseline behavior prompt, do not
deploy any of these three guidance fragments, and use the new comprehension layer to determine what
belongs in behavior versus execution policy.
