# Conditional-prompt example categories

This document is the human-readable view of the executable catalog in
`bench/mini_model_example_catalog.py`. Counts are quality floors, not the
objective. A category is useful only when it generalizes on a separate split.

## Evaluation gates

The target for an active detector is:

- exact match greater than 95% on a sufficiently diverse held-out set;
- selective precision greater than 97%;
- useful coverage reported next to precision (abstaining on everything is not
  success);
- zero semantic grants of `modify`, `commit`, or `publish`;
- zero unsafe activations on explicit read-only, negated, and quoted actions.

`20` remains only the absolute minimum that prevents an accidentally tiny
class. The normal target is 48 positives and 64 contrastive negatives.
Open-ended catch-all classes have a target of at least 256 positives.

## Task-method categories

Each single method currently has 48 positive semantic prototypes and 64
explicit hard negatives:

- `bugfix.root_cause`
- `feature.contract_first`
- `refactor.preserve_behavior`
- `research.evidence_first`
- `review.read_only`
- `performance.measure_first`

The larger fitting corpus adds 96 causal-family examples per method. These
numbers do not include validation or holdout rows.

## Multi-label task combinations

Each pair has 48 calibration positives. The other six pair categories provide
288 contrastive negatives. Validation and holdout use different templates and
fictional components.

- `bugfix + refactor`
- `feature + refactor`
- `bugfix + performance`
- `feature + research`
- `bugfix + research`
- `feature + performance`
- `research + review`

These pairs are intentionally compatible. Invalid combinations such as a
write-required method plus an explicit read-only constraint remain authority
conflicts, not valid multi-label targets.

## Uncategorized task discourse

The task gate does not treat `uncategorized` as one homogeneous phrase list.
It contains eight independently audited subfamilies with 48 calibration
examples each:

- acknowledgement;
- quoted action;
- conceptual question;
- status-only request;
- future hypothetical;
- explanation-only request;
- ambiguous discussion of a method without authority to perform it;
- out-of-domain conversation.

Together with the earlier neutral families, the available task-gate pool has
544 `uncategorized` rows. Each subfamily has separate validation and holdout
phrasing. The 384 new rows are not folded into the packaged v4 head: doing so
reduced its retained-holdout selective precision from 92.3% to 88.35%. They
remain input for the replacement architecture once that architecture passes
the gates.

## Literal authority and safety

Authority remains deterministic: embeddings may propose a method but may never
grant writes.

- `answer_only`: 256 examples;
- `diagnose_only`: 48;
- `modify`: 48;
- `read_only`: 48;
- `commit`: 48;
- `publish`: 48;
- `negated_or_quoted_action`: 48.

Every category is evaluated against both required and forbidden authority.

## Reasoning-window categories

The reasoning head has 48 new synthetic examples plus the original curated
examples for every actionable or healthy class:

- excessive exploration;
- unchanged retry loop;
- premature completion;
- speculative claim;
- verification gap;
- healthy progress.

`uncategorized` has 269 calibration examples because it must absorb ordinary
planning, cautious hypotheses, scoped reading, pending verification, and
answer-only reasoning without producing corrective guidance.

## Visible-message categories

This corpus is prepared but its head is not yet active at runtime:

- evidence-free completion: 48;
- avoidable user question: 48;
- repeated hypothesis: 48;
- unsupported claim: 48;
- healthy progress: 48;
- uncategorized: 256.

The distinction between reasoning and visible messages is deliberate. Provider
reasoning bodies and assistant-visible prose have different distributions and
must not share a threshold simply because some labels sound similar.

## Current measured state

The expanded literal-authority corpus passes the greater-than-95% gate. The
reasoning head reached 96% exact match and 96% selective precision on its old
25-row holdout, but that holdout is too small to certify the new target.

The first true multi-label v5 experiment did **not** pass: its independent
thresholds achieved 100% validation precision only by dropping recall to
44.5%, then fell to 87.9% precision and 28.5% recall on holdout. It is kept as
an experiment in `bench/task_policy_multilabel_head.py` and is not a packaged
runtime artifact. The next iteration must use clause-level intent detection or
another cardinality-aware architecture and must beat the active hybrid router
before replacement.

The contextual E5-small experiment improved holdout exact match from 59.38%
with the static frozen baseline to 82.29%, but remained below the 95% gate.
Partial fine-tuning reached 83.33% and did not improve macro F1 or precision.
See `docs/CONTEXTUAL_EMBEDDING_BENCHMARK.md` for the full comparison and the
decision not to alter the runtime yet.
