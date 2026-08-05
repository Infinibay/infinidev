# Infinidev–LLM problem map

This is the entry point for model research. It links observed product behavior to the information
needed from a model and only then to experiments. It must be revised as runtime evidence changes.

## Problems supported by current evidence

### P1 — Guidance can amplify work without improving outcomes

- **Observed problem:** in the 36-run pilot, all three candidate prompts added latency; Sol also
  added substantial tool and token use, while human-reviewed outcomes did not improve.
- **What we need to understand:** which instructions the model already infers from the task and
  harness; which restatements it treats as stronger scope; whether modal force or repetition causes
  broader verification than intended.
- **Experiments:** equivalent objective variants with and without duplicated quality language;
  `must`/`should`/default/exception contrasts; instruction in behavior versus execution-policy layer.
- **Evidence:** reconstructed verification scope, stop conditions, priority resolution, and later
  execution cost on a task with a real quality–cost trade-off.
- **Possible intervention:** remove redundant behavior text, narrow execution policy, or route a
  compact prompt to models whose baseline already supplies the behavior.
- **Confirmation:** held-out agent tasks must improve the desired outcome, not merely the model's
  explanation of the prompt.

### P2 — The current tasks do not expose meaningful model differences

- **Observed problem:** every baseline and candidate artifact reached the human-rubric ceiling.
- **What we need to understand:** where models actually diverge when requirements are incomplete,
  conflicting, nested, costly to verify, or controlled by the user.
- **Experiments:** controlled comprehension families followed by execution tasks only for divergence
  points: actionable vagueness versus blocking ambiguity, same-level contradiction, precedence,
  nested exceptions, and consequential external actions.
- **Evidence:** concrete differences in objective, authorization, ambiguity, conflict resolution,
  stop conditions, and confidence; then artifact quality and cost.
- **Possible intervention:** model-specific prompt fragment, routing rule, or task template. A null
  result means retain the shared baseline.
- **Confirmation:** non-ceiling held-out tasks and repeated runs for surviving candidates.

### P3 — Surface wording is being confused with semantic success

- **Observed problem:** literal final-answer checks produced false negatives for valid conditional
  recommendations and code reviews.
- **What we need to understand:** whether wording variants preserve the same meaning, and which
  properties can be verified deterministically without relying on a keyword.
- **Experiments:** semantically equivalent final-answer forms, direct versus indirect questions,
  severity synonyms, bullets versus prose, and explicit versus implied user ownership.
- **Evidence:** stable structured interpretation plus blind review of concrete outputs.
- **Possible intervention:** demote lexical checks; validate repository state deterministically and
  use a declared semantic rubric for communication behavior.
- **Confirmation:** the evaluator accepts equivalent valid outputs and still rejects true semantic
  failures.

## Suspected problems that still need baseline evidence

These are not yet established failures. First collect representative Infinidev traces or construct a
baseline task where the failure is observable.

### H1 — Critical constraints may be lost in long, noisy, or informal requests

- **Information needed:** sensitivity to register, layout, position, noise, typos, and code-switching
  when meaning is held constant.
- **Candidate instruments:** formal/informal/semi-formal equivalents; paragraph/list/table;
  constraint first/middle/last; clean/noisy variants.
- **Do not conclude:** that a model “prefers lists” from self-report. Infer only from comparative
  stability and omissions.

### H2 — Operators and exceptions may be scoped incorrectly

- **Information needed:** interpretation of `never`, `always`, `sometimes`, `unless`, `except`,
  `all`, `some`, negation, temporal triggers, and nested clauses.
- **Candidate instruments:** minimal pairs whose reviewed key changes in exactly one field.
- **Do not conclude:** that capitalization itself improves execution until a held-out agent task
  confirms it.

### H3 — Infinidev may ask too often or act when it should pause

- **Information needed:** how the model separates reversible ambiguity from decisions involving user
  priorities, authority, credentials, destructive changes, or external state.
- **Candidate instruments:** matched ask/proceed families across planning, implementation, research,
  and deployment.
- **Possible intervention:** behavior-layer interaction policy conditioned on user preference, while
  hard permission boundaries remain invariant.

## Admission rule for new cases

Every family must name a problem ID, the needed information, the competing hypotheses, the single
factor varied, the evidence fields that discriminate them, and the intervention that could follow.
Coverage counts are secondary. Hundreds of cases are justified by the number of live hypotheses,
domains, difficulty levels, and validation repetitions—not by filling a Cartesian grid mechanically.
