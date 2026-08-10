# Static Qwen3 embedding calibration

Infinidev uses the bundled `ken/static-qwen3-r512-v2` table as its primary
embedding backend. It is a distilled, additive approximation of
`Qwen/Qwen3-Embedding-0.6B`: tokenize, sum learned token rows, project from rank
512 to 1024 dimensions, then L2-normalize.

## What a coordinate means

An individual output coordinate does not have a stable human label such as
"testing" or "Spanish". The 1024 coordinates are a basis chosen by the teacher
and the reduced-rank regression. Rotating that basis would preserve every
cosine while changing every coordinate.

Meaningful objects are therefore geometric:

- a direction estimated from controlled contrasts;
- a subspace learned from several related contrasts;
- a distribution of similarities for a particular consumer;
- a token's exact contribution to one of those directions.

Because this model is additive, token attribution is exact before final
normalization. That makes it unusually inspectable, but also means it cannot
represent word order or compositional negation reliably.

## Reproducible measurements

Run the dependency-light study against the artifact shipped by Infinidev:

```bash
uv run python bench/static_qwen3_calibration.py
```

To compare it with the original teacher when the optional torch dependencies
and model weights are available:

```bash
uv run python bench/static_qwen3_calibration.py \
  --teacher Qwen/Qwen3-Embedding-0.6B
```

The controlled study on 2026-08-09 found:

| Measurement | Result |
| --- | ---: |
| English held-out action direction accuracy | 84.7% |
| Spanish held-out action direction accuracy | 50.0% |
| English raw classifier with abstention | 88.2% precision, 17/24 coverage |
| Spanish raw classifier with abstention | 70.0% precision, 10/24 coverage |
| Static/teacher sentence cosine on Spanish cases | 0.344 mean |
| Teacher Spanish action accuracy | 87.5% |
| Teacher retained by the static rank-512 projection | 0.974 mean cosine |
| Teacher projected into that subspace, Spanish accuracy | 86.1% |
| Throughput on one CPU thread | about 10,000-11,000 texts/s |

An external evaluation on the 341-example Spanish split of MCoNaLa, kept
entirely out of fitting, produced:

| Encoder | R@1 | R@5 | R@10 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static v2 | 0.153 | 0.361 | 0.466 | 0.258 | 13 |
| Qwen3 teacher | 0.501 | 0.921 | 0.974 | 0.684 | 1 |

This is a cross-language intent-to-code retrieval task, not the exact Ken file
ranking task, but it exposes the same pattern: the static table contains useful
Spanish signal and still leaves a large, measurable gap to its teacher.

The last two rows are the important diagnosis: rank 512 and the shared
projection preserve enough capacity for Spanish. The loss is primarily in the
fitted token table, whose training distribution was code-shaped and whose
published ranking result is much stronger for English prompts than Spanish
prompts.

A small causal experiment updated only the 115 token rows touched by a
controlled Spanish training set. Held-out Spanish action accuracy rose from
50.0% to 70.8% while English stayed at 83.3%. A global 512x512 residual head did
not generalize. This proved that token-table adaptation was viable, but also
that a synthetic action set was not sufficient evidence for shipping it.

## Spanish query adapter

The production adapter uses the exact English/Spanish translation pairs already
present in `python-docs-es` PO files. Spanish rows are fitted toward the static
v2 vector of the corresponding English text. This target is more useful than
blind teacher imitation for code retrieval: it places the Spanish intent in the
English/code geometry v2 already models well.

Several symmetric candidate tables improved Spanish while causing small but
statistically detectable regressions in Python, Go, or Java. They were rejected.
The shipped design is asymmetric instead:

- passages, stored vectors, and non-Spanish queries use v2 unchanged;
- a token-level Bernoulli language classifier selects Spanish queries;
- selected queries add a sparse rank-512 residual before the existing fixed
  projection;
- the 10.0 MB adapter records the SHA-256 of the exact v2 parent and refuses to
  load with any other table.

The selector is fitted on source-path-disjoint train families. Its threshold is
chosen on validation for at most 0.5% empirical English false positives. On the
held-out parallel test it reaches 94.2% Spanish recall with 0.48% English false
positives; it selects all 341 MCoNaLa Spanish intents.

End-to-end paired results on MCoNaLa are:

| Encoder | R@1 | R@5 | R@10 | R@20 | MRR | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static v2 | 0.152 | 0.361 | 0.466 | 0.572 | 0.258 | 13 |
| v2 + Spanish query adapter | 0.179 | 0.408 | 0.522 | 0.639 | 0.299 | 8 |

With 10,000 paired bootstrap samples, adapter-minus-v2 intervals were
`R@5 +0.047 [0.006, 0.085]`, `R@10 +0.056 [0.018, 0.094]`, and
`MRR +0.041 [0.014, 0.069]`. CodeSearchNet test sets of 1,000 queries each for
Python, JavaScript, Java, Go, and Ruby remain effectively unchanged because the
adapter is never applied to passages and almost never selected for English.

The current artifact blends that production residual with a second residual
fitted on 12,368 Spanish translations of CodeSearchNet validation docstrings.
The blend weight was selected on the validation mixture, not on MCoNaLa or the
CodeSearchNet test split. A weight of 0.25 improved validation macro MRR from
0.6665 to 0.6695. On untouched external sets it changed MCoNaLa MRR from 0.299
to 0.300, Spanish M2CRB Java from 0.564 to 0.575, JavaScript from 0.403 to
0.407, and Python from 0.935 to 0.937. Five 1,000-query English CodeSearchNet
test sets were unchanged except for a 0.001 gain on Go. These are deliberately
small gains: the blend was accepted because it broadens code-shaped Spanish
coverage without a material regression, not because one benchmark moved
dramatically.

The reproducible offline pipeline is:

1. `build_spanish_embedding_corpus.py` builds licensed, metadata-rich parallel
   records and source-family splits.
2. `fit_static_qwen3_spanish.py` performs stochastic sparse-row fitting with a
   fixed projection, collision freezing, replay, and retrieval gates.
3. `build_static_qwen3_spanish_adapter.py` extracts the sparse query residual,
   fits the language selector, and binds it to v2.
4. `static_qwen3_calibration.py` reports controlled behavior and paired
   bootstrap retrieval deltas.

## OpenAI teacher experiment

A larger multilingual distillation study collected 221,653 unique
`text-embedding-3-large` vectors at 1,024 dimensions. The final mixture contains
122,584 unique pairs (95,866 train, 12,904 validation, 13,814 test), including
93,756 English and 22,457 Spanish records plus smaller Portuguese, French, and
Italian samples. The exact paid input was 18,349,251 tokens, approximately
USD 2.3854 at the price used on 2026-08-09. The SQLite cache passed
`PRAGMA integrity_check` and is kept outside Git.

The first trainer appeared to leak memory because retrieval validation built a
95,000 by 95,000 float32 similarity matrix, roughly 36 GB, and recomputed ridge
sufficient statistics for every hyperparameter. The corrected implementation
accumulates `X^T X` and `X^T Y` in chunks once, reuses them, caps model-selection
examples at 8,192, and hash-samples at most 2,048 retrieval pairs. The same fit
then completed in about one minute with bounded memory.

More teacher data did not by itself produce a better static model. The global
ridge candidate reached about 0.532 teacher cosine on its test split but
regressed MCoNaLa MRR from 0.299 to 0.105 and Spanish M2CRB Java from 0.564 to
0.262. A sparse OpenAI contrastive adapter improved its internal test MRR from
0.494 to 0.551 but regressed MCoNaLa to 0.280. Dense residual and commit-diff
candidates also failed external gates. They were rejected; no OpenAI-distilled
weights are shipped. This result is why current work favors real natural-language
query to exact-code pairs and hard negatives over adding general technical
prose or another teacher call.

## Product boundaries

Use the static table now for candidate retrieval, fuzzy code navigation, and
topic relevance. Those tasks benefit from high throughput and tolerate an
abstaining or lexical fallback.

Do not treat one cosine threshold as a universal semantic detector. In the
curated calibration set, paraphrased duplicates overlapped heavily with
same-topic-but-different-action pairs. Deduplication must retain structural or
lexical evidence, and action routing must use held-out, language-specific
calibration.

### Runtime stagnation control

The one production use of sentence similarity is deliberately narrower than a
phase classifier. A 2026-08-09 MiniMax M3 trace on
`pytest-dev__pytest-5103` produced consecutive same-Step summary cosines of
0.9195 and 0.8343 while making no edit and producing no new test outcome. In
separate successful small and medium runs, the highest observed adjacent
transition was 0.7793. The runtime therefore uses hysteresis: two records must
reach 0.90, or three consecutive records must each reach 0.80.

Those thresholds never act alone. Records must belong to the same explicit
implementation Step, contain no net workspace transition, and have identical
deterministic test fingerprints. Embedding failure abstains from the semantic
decision. Independently, the engine counts complete implementation windows
whose net diff and test outcomes remain static; two such windows activate the
recovery surface even if model-authored summaries alternate between unrelated
phrasings. A match does not declare failure or completion; it
narrows one following Step to workspace edits, test commands, plan transitions,
and two local source reads. The source
allowance exists because large-file excerpts may be archived at a Step
boundary and therefore cannot safely be replaced by a cross-Step "already
delivered" notice. Within one Step, native ranges and simple shell reads still
share an exact revision-bound evidence ledger.

The first live version used a discovery-tool denylist. M3 bypassed it with web
search and fetched irrelevant upstream pages. The shipped policy uses a
positive action surface instead, so new search tools do not silently become a
new escape route. No extra LLM or embedding API call is introduced: the static
Qwen table runs locally, and all edit/test gates are computed from engine state.

The edit gate uses the task diff fingerprint rather than edit-tool count. This
matters for weaker models that tentatively patch a file and restore the exact
input in the same Step: that episode has no net workspace transition, does not
renew the Step window, does not satisfy an implementation Step, and does not
mask semantic stagnation. When a net change lands, pending model-authored
discovery that explicitly names the changed file is retired before activation;
user-approved or unrelated discovery is preserved.

Negation remains a protocol/language problem rather than an embedding problem.
For example, in "do not modify; only inspect", the token contribution of
"modify" toward the change direction dwarfs the contribution of "not". Exact
guards should continue to handle negation, secrets, paths, commands, and other
safety-critical syntax.

## Future calibration boundaries

Keep repository and intent-family holdouts frozen before fitting. Any future
adapter must measure Spanish retrieval, action distinction, English and
multi-language code regression, negation, and duplicate-versus-related
separation. Retrieval gates must be reported per programming language; an
aggregate can hide a localized regression.

Do not replace v2 bytes in place. A symmetric new table is a new embedding
space and requires reindexing. A query-only adapter can retain the passage space
only when it is cryptographically bound to v2 and never affects passage calls.

MCoNaLa stays evaluation-only because it is small and CC-BY-SA. Spanish Stack
Overflow can supply a much larger real-query evaluation set, while a training
corpus should use sources whose redistribution and model-training terms are
explicitly compatible. General Spanish retrieval corpora may improve token
coverage, but they must not replace code-shaped examples or developer-intent
holdouts.

The first reproducible source pass is implemented in
`bench/build_spanish_embedding_corpus.py`. A pinned `python-docs-es` snapshot
produces 39,445 unique, short technical passages under the PSF license. An
explicit research-only switch adds 1,263 Django Girls records under CC BY-SA;
it is excluded by default because source availability and artifact
redistributability are different questions. See `bench/data/README.md` for
snapshot hashes, commands, and the broader source/license catalog.

Programming books are a plausible auxiliary domain, unlike a large undirected
mixture of general literature. The fitting experiment should still sweep their
weight rather than assume more Spanish is better. The acceptance surface is a
Pareto frontier across Spanish technical retrieval, English/code regression,
action distinction, negation, and source-family holdouts. This controls domain
drift algorithmically and does not add prompts or runtime model calls.

Ken's published repository contains the runtime, model provenance, benchmark
figures, re-embedding command, and embedding-space probe. It does not contain
the original static-table corpus builder or fitting program. Infinidev therefore
keeps its reconstructed Spanish calibration and evaluation pipeline in `bench/`
so the shipped adapter can be audited and regenerated.

The `midudev/libros-programacion-gratis` catalog was audited at commit
`f62f847d14af3b1931850068a5d65d4bbf4c186f`. It contains 115 catalog entries
and 117 locally hosted PDF/EPUB files, but its catalog schema does not record a
per-book license. A document-level sample found exercise books under
CC BY-NC-SA, CC BY-NC-ND, CC BY-SA, and older jurisdiction-specific Creative
Commons variants. Therefore "listed as free" is only a discovery signal. No
book text is included in the production adapter. Future use must record the
license found inside each document, keep NC/ND material evaluation-only, and
extract description-to-code or exercise-to-solution pairs rather than undirected
chapters. Local books with unverified licenses under
`/home/andres/Documentos/Libros` remain research-only and must not be copied
into the repository or a redistributed training corpus.
