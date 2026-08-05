# Prompt-comprehension research method

## Start with an Infinidev problem, not the question

Cases are generated only after documenting this chain:

1. **Observed Infinidev problem:** what does the product do poorly, wastefully, inconsistently, or
   unsafely with the current model/harness combination?
2. **Information needed about the model:** what must we learn to explain or solve that product
   problem? This must be narrower than a general personality description.
3. **Observable hypothesis:** what should remain stable, and what should change?
4. **Questions or examples:** which minimal controlled variants collect the needed information?
5. **Evidence:** which concrete parts of the reconstruction demonstrate the effect?
6. **Product intervention:** behavior prompt, execution policy, objective template, context layout,
   evaluator, routing, or no change.
7. **Held-out confirmation:** does that intervention improve the original Infinidev problem?

A question with no originating product problem and no plausible intervention is excluded, even if it
is interesting. A linguistic phenomenon is an experimental variable, not a reason to run a study.

## Three different objects of study

- **Comprehension:** what objective, constraints, authority, conflicts, exceptions, and completion
  conditions the model reconstructs. This is the scope of this battery.
- **Execution:** whether it follows that interpretation when tools, cost, uncertainty, and failure are
  present. This requires agent-task trials and cannot be inferred from a JSON explanation.
- **Preference:** which action it chooses when several interpretations or policies remain valid.
  This is measured by the behavior questionnaire and interpreted relative to user priorities.

The reports must not claim to reveal hidden chain-of-thought or to understand a model perfectly.
They produce an empirical decision map from externally observable answers.

## Family design

Each family has an anchor plus controlled variants:

- **Equivalent variants** change register, layout, position, or phrasing while preserving the reviewed
  interpretation key. Divergence reveals sensitivity or fragility.
- **Contrast variants** change one operator such as `never`, `sometimes`, `unless`, `all`, or `some`.
  The expected key changes only where that operator changes meaning.
- **Adversarial variants** introduce contradiction, misleading examples, noise, or ambiguous scope.
  They test whether the model detects the problem instead of silently resolving it.

Self-reports such as “I prefer lists” are weak evidence. Format guidance should be inferred from
accuracy and stability across semantically equivalent list, paragraph, table, formal, informal, and
example-led variants.

## Research-question catalog

| What we want to know | Why it is useful | Families needed | Evidence to retain |
|---|---|---|---|
| Does register change perceived authority or scope? | Select natural wording without weakening constraints. | Formal, semi-formal, informal equivalents. | Full reconstruction plus authorization fields. |
| Does structure change completeness? | Choose paragraph, bullets, table, or schema per model. | Same content in each structure and order. | Omissions, invented relationships, deliverables. |
| How are modal terms interpreted? | Calibrate policy language rather than assuming capitals help. | `NEVER`/“do not” equivalents; `always`, `sometimes`, defaults, and exceptions as contrasts. | Constraints, exceptions, unauthorized actions. |
| Does the model handle negation and quantifier scope? | Prevent broadening `some` to `all` or attaching `not` to the wrong clause. | Minimal operator and scope contrasts. | Exact affected entities and actions. |
| How are conflicts resolved? | Encode precedence explicitly enough for the route. | User/repo conflicts, same-level contradictions, rule/example conflicts. | Conflict detection and stated resolution basis. |
| How much vagueness is safely resolvable? | Decide when the agent may proceed and when it should ask. | Reversible ambiguity, user-owned ambiguity, missing referents. | Ambiguities, ownership, stop conditions, confidence. |
| Do examples clarify or override rules? | Use examples without accidental imitation or policy drift. | Consistent, format-only, incomplete, and conflicting examples. | Rule/example distinction and copied assumptions. |
| How robust is meaning under noise? | Support real user prompts with typos, code-switching, and irrelevant detail. | Clean/noisy equivalents at controlled noise levels. | Semantic stability and new unsupported inferences. |
| Does instruction position matter? | Place critical constraints where they survive long prompts. | Constraint first, middle, last, and nested equivalents. | Retention of the same constraint and scope. |
| Are temporal and nested conditions scoped correctly? | Avoid applying a rule before its trigger or outside its branch. | `if`, `until`, `after`, nested and exception contrasts. | Trigger, duration, branch, and stop condition. |

These questions cross task domains because a linguistic result observed only in planning may not
generalize to implementation, review, research, user interaction, or external-state authorization.

This table is a library of possible instruments, not the study backlog. The backlog is derived from
the problem map in `INFINIDEV_LLM_PROBLEM_MAP.md`; only the rows needed to discriminate a live
problem are instantiated.

## Blind semantic review and adjudication

The generated interpretation key cannot validate itself. Export a packet that contains the requests,
controlled relations, and required reconstruction schema but withholds every authored `expected`
field:

```bash
uv run python -m bench.prompt_comprehension_review export \
  bench/prompt_comprehension_battery.draft.jsonl \
  bench/prompt_comprehension_battery.review-packet.json
```

Each reviewer writes one JSONL row per family using the `review_row_contract` embedded in the packet.
The reviewer must reconstruct all variants, judge every semantic check, and record possible template
or diversity dependence without seeing the authored keys. Build the key-revealing comparison dossier
only after those reviews are frozen:

```bash
uv run python -m bench.prompt_comprehension_review dossier \
  bench/prompt_comprehension_battery.draft.jsonl REVIEWS.jsonl DOSSIER.json \
  --min-reviews 1
```

`ready_for_adjudication` is deliberately not approval. A separate adjudicator compares the authored
keys with the blind reconstructions and records one `approve` or `reject` JSONL decision per family,
including the dataset hash, adjudicator identity, and rationale. Applying those decisions creates a
new dataset; it never mutates the draft:

```bash
uv run python -m bench.prompt_comprehension_review apply \
  bench/prompt_comprehension_battery.draft.jsonl DOSSIER.json \
  ADJUDICATIONS.jsonl APPROVED_OR_REJECTED.jsonl
```

Approval is family-atomic. Missing variants, stale hashes, duplicate reviewers, failed checks,
revision requests, and approval of a family that is not ready all fail closed. One reviewer is the
mechanical minimum; consequential release sets should use multiple independent reviewers.

Before reviewing all 224 families, use the deterministic 16-family pilot documented in
`PROMPT_COMPREHENSION_PILOT_REVIEW.md`. It samples one linguistic and one execution family from every
domain while maximizing distinct research dimensions. The pilot is intended to find generator-level
defects and template dependence early, not to approve a convenient subset for model execution.

## Interpretation rule

Numbers summarize coverage, latency, parsing, and cost. Conclusions are written from the model's
actual reconstructions, grouped by category and controlled family. A finding may justify a model-
specific behavior or execution-policy candidate, but only a separate held-out execution trial can
show that the candidate improves the desired user outcome.
