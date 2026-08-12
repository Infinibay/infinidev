# Natural user-request data

Infinidev builds its task-policy corpus from natural user requests, then reviews every request
individually. Acquisition rules may find likely programming requests or diversify a queue, but
they never assign `bugfix`, `feature`, `performance`, `refactor`, `research`, or `review`.

Two sources serve different parts of the distribution:

- [Open-SWE-Traces](OPEN_SWE_EXTERNAL_DATA.md) provides repository-grounded change requests;
- [WildChat](https://huggingface.co/datasets/allenai/WildChat) provides natural conversational
  requests, questions, ambiguity, multilingual phrasing, and read-only requests that issue-based
  benchmarks rarely contain.

Downloaded candidate text is not committed or relicensed as part of Infinidev. The minimized
human annotation ledgers (candidate IDs, labels, zero-label reasons, and short decision notes) are
committed under `data/task-policy-reviews/` as a separately licensed data component. The software
remains MIT licensed; see the data directory's license and upstream attribution notice.

## Acquire WildChat candidates

Run:

```bash
uv run python -m bench.wildchat_candidate_sampler \
  .infinidev/external-data/wildchat/candidates.jsonl \
  --scan-limit 100000 --limit 2000 --max-per-language 1000 --seed 811
```

The sampler uses a pinned dataset revision. It keeps only the first non-empty user utterance,
rejects conversations or utterances marked toxic or redacted, rejects exact normalized-text
duplicates, and records source provenance. Coarse lexical signals create a diverse review queue;
the manifest explicitly records that these are selection hints rather than labels.

WildChat is distributed under ODC-By. That license governs database rights but does not necessarily
grant independent rights in every piece of content. Consequently, downloaded conversation text in
`.infinidev/external-data/` remains ignored, source attribution stays attached to every row, and a
release-specific legal and privacy review is required before distributing text or a derived model
artifact. The committed ledgers omit conversation text and retain the applicable ODC-By notice.

## Freeze before reviewing

The original conversation-only partition is useful for acquisition bookkeeping but is not a valid
model-evaluation split. Manual review found repeated prompt templates and near-identical requests in
different conversations. A conversation-disjoint holdout would therefore leak task families and
inflate measured accuracy.

Create the family-disjoint split before training or evaluating:

```bash
uv run python -m bench.external_candidate_family_split \
  .infinidev/external-data/wildchat/candidates.jsonl \
  .infinidev/external-data/wildchat/family_round1 \
  --review-ledger .infinidev/external-data/wildchat/round1_0_reviews.jsonl \
  --review-ledger .infinidev/external-data/wildchat/round1_1_reviews.jsonl \
  --review-ledger .infinidev/external-data/wildchat/round1_2_reviews.jsonl \
  --reserve-target 120 --queue-partitions 16 --seed 1223
```

The splitter keeps conversation identity atomic and groups conservative lexical near-duplicates.
It never assigns labels. Every family containing a reviewed candidate goes to development; no
member of such a family may enter the reserve. Over-grouping is safer than leakage because it costs
some independent examples but cannot inflate held-out quality. Once reserve text, labels,
predictions, or errors influence a decision, it becomes development data and a new reserve is
required.

The first acquisition scanned 100,000 conversations and found 11,883 unique safe requests matching
the broad programming filter. It selected 2,000 candidates across 51 detected languages. Selection
hints include 103 review and 105 research candidates, in addition to questions, repairs,
implementations, performance, refactors, and general programming requests. These are queue
statistics, not ground-truth label counts.

The authoritative family split contains 1,562 families. It initially placed 549 rows from 256
already exposed families in development, 1,331 rows in sixteen family-disjoint annotation queues,
and 120 rows in the sealed reserve. All 549 development rows have now been reviewed manually. The
reserve is `family_round1_reserve.jsonl`; its pre-inspection SHA-256 is
`2cb4818be6221a51ecc4a2b16e5151e80f7851b20cac5d5f57e9b1ba89d790ae`.
The old `round1_19.jsonl` conversation-only reserve is obsolete and must not be used for final
metrics.

## Review every request manually

One review row has this form:

```json
{
  "candidate_id": "wildchat:conversation-id:0",
  "include": true,
  "policies": ["review"],
  "uncategorized_reason": null,
  "notes": "Evaluates supplied code without requesting an implementation."
}
```

The reviewer reads the complete request and writes each decision independently. Scripts may check
schema, duplicates, counts, provenance, and split leakage; scripts and model predictions may not
write or prefill labels. In particular:

- a word such as “review” is not evidence if it appears in quoted text or if the user also asks to
  implement the findings;
- “investigate” is `research` only when gathering external evidence is an outcome; repository-local
  diagnosis is not automatically research;
- conceptual and how-to questions are `answer_only` even when their answer discusses fixing,
  refactoring, or implementing code;
- maintenance, migration, documentation-only, test-only, deployment, and configuration requests
  remain `unsupported_method` unless one of the six policies genuinely changes the workflow;
- multiple labels are used only for independently requested outcomes, not implementation details.

Every included zero-label row needs one of `answer_only`, `out_of_domain`, `unsupported_method`,
or `ambiguous_method`. `out_of_domain` is retained because broad natural-data acquisition creates
important lexical false positives such as “collaboration program” or prose containing “research”.
Every labeled row needs no uncategorized reason. Every row needs a concise rationale tied to the
requested outcome.

After each batch, run `bench.external_review_family_audit`. It reports label differences within a
family and selects unreviewed family members, but never writes a decision. A conflict is an audit
target rather than automatically an error: two prompts may share almost all code while one asks to
add behavior and another asks to repair it. The reviewer must read the differing operative request.

## Evidence standard

Thousands of rows improve coverage only when their families differ. Report per-label precision,
recall, F1, positive support, and hard-negative source. A class does not pass the 95% individual
accuracy target on a negative-only holdout or a handful of positives. Calibration and final
evaluation must be source- and conversation-disjoint, and the final holdout is opened once.

Open-SWE and WildChat should also be reported separately. A combined number can otherwise hide a
classifier that performs well on issue-style prose but fails on short conversational requests.
