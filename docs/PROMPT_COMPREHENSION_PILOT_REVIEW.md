# Prompt-comprehension semantic review pilot

This pilot tests the instrument before spending review effort on all 224 families. It contains 16
families and 48 requests: one linguistic and one execution-policy family for each of the eight task
domains. The deterministic selector prefers previously unused research dimensions, so the current
pilot covers 16 different dimensions rather than repeatedly sampling the same behavior.

## Reviewer inputs

- `bench/prompt_comprehension_pilot.review-packet.json` is the only content packet the blind reviewer
  should receive.
- `bench/prompt_comprehension_pilot.reviews.template.jsonl` contains one fail-closed row per family.
- `bench/prompt_comprehension_pilot.manifest.json` records the source dataset hash, selection seed,
  exact family IDs, and balance counts.
- `bench/prompt_comprehension_pilot_shards/` contains four independent four-family assignments, each
  with its own blind packet and fail-closed review template. `assignments.json` records the exact
  partition without assigning reviewer identities.

Do not give the reviewer the draft dataset, family registry, generator, or authored interpretation
keys before their JSONL review is frozen.

For every variant, the reviewer reconstructs the objective, deliverables, constraints, user-owned
decisions, authorization boundary, verification, ambiguities, stop conditions, conflicts, precedence,
and interpretation risks. They then assess whether:

1. equivalent variants preserve meaning;
2. the contrast changes only the intended variable;
3. the language is natural rather than benchmark-shaped;
4. the request is self-contained;
5. authorization is unambiguous;
6. calibration and validation do not leak the same semantic example.

The template deliberately starts with `verdict: revise`, every check false, and `TODO` content. This
prevents an untouched template from becoming approval evidence. Record scenario repetition or
template dependence in `diversity_concern` even when the family is otherwise acceptable.

## Reproduce the selection

```bash
uv run python -m bench.prompt_comprehension_review pilot \
  bench/prompt_comprehension_battery.draft.jsonl \
  bench/prompt_comprehension_pilot.review-packet.json \
  bench/prompt_comprehension_pilot.reviews.template.jsonl \
  bench/prompt_comprehension_pilot.manifest.json
```

Changing `--seed` creates another deterministic stratified sample. Do not change it after reviews
start; the dataset SHA-256 in every review row binds the review to the current source bytes.

The four reviewer-sized shards are reproduced with:

```bash
uv run python -m bench.prompt_comprehension_review shard \
  bench/prompt_comprehension_pilot.review-packet.json \
  bench/prompt_comprehension_pilot_shards \
  --shards 4
```

Assign a shard packet and its matching template to a reviewer; do not send `assignments.json` as a
substitute for the actual packet. Reviewer identity is deliberately not prefilled.

Validate a completed shard before accepting it into the dossier input:

```bash
uv run python -m bench.prompt_comprehension_review check \
  bench/prompt_comprehension_pilot_shards/shard-01.review-packet.json \
  SHARD-01.completed-reviews.jsonl \
  SHARD-01.progress.json
```

The command exits unsuccessfully for missing families, stale hashes, duplicate identities, malformed
reconstructions, or remaining `TODO` placeholders. A successful completeness report still does not
judge semantic agreement and cannot approve a family.

## After blind reviews

Freeze the completed review JSONL, then create a dossier that reveals the authored keys alongside
the independent reconstructions:

```bash
uv run python -m bench.prompt_comprehension_review dossier \
  bench/prompt_comprehension_battery.draft.jsonl \
  COMPLETED-REVIEWS.jsonl \
  PILOT-DOSSIER.json \
  --min-reviews 1
```

A separate adjudicator records explicit family decisions after examining disagreements. The pilot's
main output is a correction plan for the generator: recurring defects should be fixed at their source
and the complete 672-case draft regenerated. Pilot approval alone does not approve the full dataset
and does not authorize provider calls.
