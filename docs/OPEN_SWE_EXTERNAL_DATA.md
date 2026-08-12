# Open-SWE external candidate data

Infinidev uses a source-data boundary for candidate requests taken from
[NVIDIA Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces).
The repository contains the MIT-licensed extraction code, but it does not contain copied issue
texts, reviewed derivatives, or generated training artifacts.

## Download and extract

Run:

```bash
uv run python -m bench.open_swe_candidate_sampler
```

The command streams a bounded dataset prefix from a pinned revision and writes:

- `.infinidev/external-data/open-swe/candidates.jsonl`
- `.infinidev/external-data/open-swe/candidates.jsonl.provenance.json`

The entire `.infinidev/` directory is ignored by Git. The first file is a deterministic,
repository-diverse manual-review queue. The second records the source revision, upstream
license, selection settings, row count, and SHA-256 digest.

The extractor retains only the original `<issue_description>` body. It does not treat an
upstream category such as `bug-fix` or `feature-request` as an Infinidev label. Every candidate
must be reviewed individually, including exclusion and `uncategorized` decisions.

The local review ledger is external data too. A review row records a candidate ID, an explicit
`include` decision, zero to three short policy names, and a rationale. A row with no policy must
also record `uncategorized_reason`; a labeled row must not. Accepted reasons are `answer_only`,
`out_of_domain`, `unsupported_method`, and `ambiguous_method`. The loader resolves short names to
canonical policy IDs and rejects repeated labels, more than three labels, unknown labels, missing
rationales, invalid uncategorized reasons, and source rows without provenance.

## Semantic review contract

Upstream issue titles and category hints are evidence, not labels. Review the actionable request
and its observable outcome:

- `bugfix` restores behavior already promised by code, documentation, a protocol, or a supported
  input contract;
- `feature` creates or intentionally changes an observable capability, schema, or public API;
- `refactor` changes internal structure while preserving observable behavior;
- `performance` requires measuring, diagnosing, or improving a resource or latency objective;
- `research` requires gathering external evidence rather than merely inspecting the repository;
- `review` evaluates an existing artifact without implementing a change.

Do not label a dependency upgrade, deprecation, removal, platform-baseline change, or public API
migration as `refactor` merely because its implementation moves code. Until a dedicated
maintenance/migration policy exists, use `uncategorized_reason: unsupported_method` when none of
the six methods describes the requested workflow without distortion.

Use multiple labels only when each label contributes a separately requested outcome and would
materially change how the agent works. An implementation detail needed to deliver a feature is
not automatically an independent refactor. Conversely, a request can legitimately be
`feature + performance` when it both adds a capability and requires a measured resource target.

The manually reviewed local ledger now contains 1,021 requests across four repository-disjoint
rounds. Its current label counts are 431 feature, 334 bugfix, 54 refactor, 32 performance,
5 research, and 3 review, with 174 zero-label decisions and 12 genuine multi-label requests.
Zero-label reasons include 137 unsupported maintenance/documentation/test operations, 34
answer-only requests, and 3 semantically ambiguous requests. These counts are an audit result,
not a balance target: do not change a correct label to make a class larger. In particular,
Open-SWE cannot establish research or review recall with only five and three natural positives.

To create a new repository-disjoint review round, exclude the prior queue and then partition the
new candidates before reading them:

```bash
uv run python -m bench.open_swe_candidate_sampler \
  .infinidev/external-data/open-swe/candidates_round2.jsonl \
  --scan-limit 5000 --limit 240 --max-per-repo 2 --seed 191 \
  --exclude-candidates .infinidev/external-data/open-swe/candidates.jsonl

uv run python -m bench.open_swe_candidate_partition \
  .infinidev/external-data/open-swe/candidates_round2.jsonl \
  .infinidev/external-data/open-swe/round2 --partitions 4 --seed 313
```

Candidate exclusion is by both ID and repository unless
`--allow-excluded-repositories` is supplied. Repository groups remain atomic during partitioning.
Choose and record the held-out shard before opening its contents; once its examples or errors
influence labeling, thresholds, architecture, or prompts, it is development data rather than a
sealed holdout.

The current development queue is partitioned into four manually reviewed 60-row blocks.
Repository identity is disjoint across blocks. Use them sequentially: after observing a block's
metrics it may join the next training run, while at least one later block remains untouched. Do
not call a block a holdout after its predictions or errors influenced a model or threshold.

The fourth acquisition round contains 600 additional candidates from 487 repositories, split into
ten repository-atomic blocks of roughly 60. Blocks 0 through 4 are manually reviewed development
data. Block 9
is the current sealed reserve; before any of its text was opened, its SHA-256 was recorded as
`bc283abf8322a517dccc66c2b5305770f2aef4d87c331a835eeb518709d67b73`. The intervening blocks
remain an annotation queue, not automatically labeled training data.

The first E5-small/LoRA pilot with 600 natural training rows and 60 natural calibration rows did
not pass the gate. On its then-sealed 60-row round-3 holdout, binary accuracy was 81.7% for bugfix
and 80.0% for feature. High accuracy on performance, research, or review was negative-only or had
one positive and is not accepted as evidence. That holdout is development data after inspection.
The run also exposed a calibration flaw: synthetic validation outnumbered natural calibration and
dominated threshold selection. Domain calibration now chooses per-label thresholds from natural
data only when it has at least five positive and five negative examples, with a synthetic fallback
for unsupported natural labels. A new sealed reserve is required to evaluate that correction.

Open-SWE represents change-oriented requests, not all Infinidev traffic. The reviewed queue has
many bug fixes, features, and refactors, very little explicit performance work, and no genuine
read-only research or code-review requests. High binary accuracy for an absent label is
negative-only evidence, not proof that the method can be recognized. Natural validation for
`research` and `review` requires another repository-disjoint source. WildChat supplies that
complementary natural-request queue; its acquisition and review boundary is documented in
[`NATURAL_REQUEST_EXTERNAL_DATA.md`](NATURAL_REQUEST_EXTERNAL_DATA.md).

For a smaller trial:

```bash
uv run python -m bench.open_swe_candidate_sampler \
  --scan-limit 500 \
  --limit 60 \
  --max-per-repo 2
```

## License boundary

Open-SWE-Traces declares `CC-BY-4.0`; individual rows also carry source-repository license
metadata. Downloaded text remains external data and is not silently relicensed under
Infinidev's MIT license. Keep both levels of provenance with reviewed datasets and derived
artifacts, and perform a release-specific license review before distributing any such artifact.

The dataset revision is intentionally pinned in the script. Updating it is a deliberate source
change: inspect the new dataset card and schema, update the constant, rerun tests, and regenerate
the local queue and manifest.
