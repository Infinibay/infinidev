# GPT-5.6 Sol prompt-comprehension pilot review

## Scope and provenance

`gpt-5.6-sol` reviewed the current 16-family/48-case blind pilot through Infinidev's
`openai_subscription` transport. Each family used a fresh request with one user message, no system
prompt, no prior conversation, and no authored interpretation key. Calls were single-flight,
sequential, and separated by at least two seconds. The Responses transport explicitly disabled
server-side storage through the existing subscription adapter.

- Dataset SHA-256: `9f492dc5b5d234d67aae23f3890ba4ae56de9f7dc0d9308ad8c1c4f9a354adf4`
- Provider-reported model identity: `gpt-5.6-sol`
- Reasoning effort: `high`
- Valid completed reviews: 16/16
- Verdicts: 1 `accept`, 15 `revise`, 0 `reject`
- Check results: 60 `pass`, 35 `fail`, 1 `not_applicable_by_design`
- Prompt tokens: 18,875
- Completion tokens, including provider reasoning: 55,743
- Total tokens: 74,618
- Successful-call latency: 1,112.04 seconds total; 69.39 seconds median; 99.42 seconds maximum

Two responses were rejected because Sol normalized long mixed-separator case IDs. The protocol was
changed to use short `variant_N` review slots with deterministic canonical mapping. A later stream was
closed before completion; the runner was hardened to persist that provider failure and resume without
repeating successful families. These transport events are not semantic reviews.

All literal responses, parsed reviews, progress, dossier, and frozen source bytes are under
`bench/runs/gpt-5.6-sol-comprehension-review/`. No review changes a case from `draft`.

## What Sol found that MiniMax-M3 missed

Sol applied a stricter single-variable standard and identified several plausible confounds:

1. **Nested scope:** the conditional-permission contrast removes the anchor's requirement to report a
   verification failure. Permission scope and reporting behavior change together.
2. **Examples versus rules:** the consistent example introduces an explicit two-mode test statement,
   while the rule-only anchor does not state that detail. The adversarial variant depends on a prior
   report that is not actually supplied.
3. **Failure recovery:** one supposed equivalent says to record the failure once and adapt, but does not
   explicitly preserve the exact error or prohibit retrying as the anchor does.
4. **Incremental execution:** the anchor requires preserving a working intermediate state; the
   equivalent only says to check both states.
5. **Priority resolution:** the adversarial variant changes `required` verification to `exhaustive`
   verification and drops the instruction to minimize extra work, in addition to changing priority.
6. **Web-research vagueness:** one equivalent says `label style`, while the other leaves a generic
   cosmetic detail unspecified. This changes ambiguity scope as well as wording.
7. **Missing referents:** the contrast adds verification and changes the forbidden action instead of
   changing only referent availability.
8. **Modal force:** the contrast adds test staleness and singular-edit framing in addition to changing
   the modal policy.
9. **Temporal authorization:** `until I explicitly approve it` has an unclear singular referent after
   a list of several external actions.

These are stronger generator-level findings than surface complaints about repeated templates. They
should be adjudicated against the hidden keys and, where confirmed, fixed at the generator.

## Where Sol appears too strict for this study

Sol failed `requests_are_self_contained` in 13 families because prompts did not include concrete queue
attributes, the authentication diff, a named library, the rollout alternatives, normalization rules,
or repository paths. That standard is appropriate for executing a task to completion, but this battery
asks the model to reconstruct what a request means. A request may validly target an external artifact
without embedding that artifact's contents.

The rubric therefore conflates two questions:

1. **Semantic completeness:** does the request contain enough information to reconstruct objective,
   authority, constraints, and the intended manipulated variable?
2. **Execution sufficiency:** does it contain or point to enough evidence to perform the task now?

Only the first is a universal validity gate for this comprehension instrument. Execution sufficiency
should be recorded separately as an observed ambiguity, and may intentionally fail in missing-context
families. Repeated scenario scaffolding is also expected inside a controlled family; diversity should
be judged across families, not demanded between variants whose causal value depends on similarity.

## Comparison with MiniMax-M3

MiniMax-M3 was more accepting and correctly recognized the corrected seven-family rerun, but it missed
several small confounds that Sol identified. Sol was more sensitive to omitted clauses, altered
verification strength, and differences in authorization wording. Conversely, Sol over-applied an
execution-readiness standard to semantic reconstruction and treated intentionally abstract task
targets as invalid context gaps.

The useful conclusion is not that one reviewer is better. Their disagreement reveals two review
dimensions that the schema should separate:

- causal cleanliness of the controlled family;
- whether the request is semantically interpretable versus immediately executable.

## Recommended next action

Adjudicate the nine likely single-variable confounds above and revise those confirmed by source-key
comparison. Replace `requests_are_self_contained` with separate semantic-completeness and
execution-sufficiency checks. Then repeat only affected families with both reviewers. Do not approve
or reject the full battery from either model's verdict count alone.
