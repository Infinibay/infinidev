# External embedding calibration data

External datasets are downloaded into this directory when needed and are not
committed. They retain their own licenses.

## MCoNaLa Spanish

MCoNaLa contains 341 human-rewritten Spanish programming intents paired with
Python snippets. It is used only as an external holdout, never to fit the table.

- Dataset: `neulab/mconala`, Spanish test split
- Revision: `3b1cd700203f4a613a4f306270220f32514be2cd`
- License: CC-BY-SA-4.0
- Expected SHA-256:
  `edc53be1c4675b33dbdd1891c49b66dcf4eed06001519891bc0156755ca40f3f`

```bash
curl -L \
  https://huggingface.co/datasets/neulab/mconala/resolve/3b1cd700203f4a613a4f306270220f32514be2cd/es_test.json \
  -o bench/data/mconala-es.json
sha256sum bench/data/mconala-es.json
uv run python bench/static_qwen3_calibration.py \
  --mconala bench/data/mconala-es.json
```

Add `--teacher Qwen/Qwen3-Embedding-0.6B` to compare against the original
teacher when its optional runtime and weights are installed.

## Spanish programming corpus

`build_spanish_embedding_corpus.py` converts source checkouts into auditable
JSONL. Every record carries its origin, path, license class, language, kind,
and a stable content identifier. Python PO records also retain an exact
`parallel_text` English source when Spanish and English chunk boundaries align;
37,814 of 39,445 records meet that conservative condition. Downloaded archives
and generated JSONL stay outside Git.

The default source is `python/python-docs-es`: official Spanish Python
documentation under the PSF license. The Django Girls Spanish tutorial is a
useful second domain, but it is CC BY-SA 4.0 and therefore requires the
explicit `--include-sharealike` research flag. That flag does not assert that
a derived model can be redistributed under Infinidev's MIT license.

```bash
curl -L \
  https://github.com/python/python-docs-es/archive/refs/heads/3.14.tar.gz \
  -o /tmp/python-docs-es-3.14.tar.gz
curl -L \
  https://github.com/DjangoGirls/tutorial/archive/refs/heads/master.tar.gz \
  -o /tmp/djangogirls-tutorial.tar.gz

sha256sum /tmp/python-docs-es-3.14.tar.gz /tmp/djangogirls-tutorial.tar.gz
# Expected for the snapshots studied on 2026-08-09:
# fb1aabca27fd8b158f3a5ff185db874d2aae9213ad4d8d23c9dd27cda858c7f4
# 3148fb24d9d9fe7f717266dc0999ee75413e341503822aa214e0732c6cd99217

mkdir -p /tmp/infinidev-spanish-corpus
tar -xzf /tmp/python-docs-es-3.14.tar.gz -C /tmp/infinidev-spanish-corpus
tar -xzf /tmp/djangogirls-tutorial.tar.gz -C /tmp/infinidev-spanish-corpus

uv run python bench/build_spanish_embedding_corpus.py \
  --python-docs /tmp/infinidev-spanish-corpus/python-docs-es-3.14 \
  --output /tmp/python-docs-es-corpus.jsonl

uv run python bench/build_spanish_embedding_corpus.py \
  --python-docs /tmp/infinidev-spanish-corpus/python-docs-es-3.14 \
  --django-girls /tmp/infinidev-spanish-corpus/tutorial-master \
  --include-sharealike \
  --output /tmp/spanish-programming-corpus-research.jsonl
```

The pinned snapshots produce 39,445 unique PSF-licensed technical passages.
Adding Django Girls yields 40,708 records: 40,540 prose passages and 168
contextualized code fragments. These counts are corpus-construction evidence,
not evidence that the resulting mixture improves the embedder.

The language selector and first Spanish residual use only the PSF-licensed
Python documentation pairs. The current adapter also includes a 0.25 blend of
a residual fitted on Spanish translations of 12,368 CodeSearchNet validation
docstrings. MCoNaLa and every CodeSearchNet test split remain evaluation-only.
Because CodeSearchNet aggregates repositories with their original licenses and
the dataset card labels the collection license as `other`, retain repository
provenance in research data and do not redistribute the raw code corpus.

### Source policy

| Source | Format | License | Use |
| --- | --- | --- | --- |
| Python docs in Spanish | PO / rendered HTML | PSF-2.0 | Default fitting candidate |
| Django Girls Spanish | Markdown / HTML | CC-BY-SA-4.0 | Research-only pending license review |
| MDN Spanish | HTML / Markdown | CC-BY-SA-2.5 | Research candidate, not downloaded yet |
| Wikibooks programming books | HTML / PDF | CC-BY-SA | Research candidate, not downloaded yet |
| Programar en Python (ElenQ) | PDF / source | CC-BY-SA-4.0 | Research candidate, not downloaded yet |
| MCoNaLa Spanish | JSON | CC-BY-SA-4.0 | Evaluation only |
| CodeSearchNet | Parquet | Per-source repository / dataset `other` | Fitting research; raw data not distributed |
| Midudev Spanish book catalog | PDF / EPUB / HTML | Per-document; not recorded uniformly | Discovery and license audit only |
| Pro Git Spanish | HTML / EPUB / PDF | CC-BY-NC-SA-3.0 | Excluded from fitting |
| Think Python Spanish | PDF / HTML | CC-BY-NC-SA | Excluded from fitting |
| OpenStax Introduction to Python | HTML / PDF | Restricted for model ingestion | Excluded |

Books about programming are preferable to unrelated literature because they
add Spanish syntax and explanations without abandoning developer semantics.
They are still an auxiliary domain: future fitting sweeps should cap them by
source and compare 0%, 5%, 10%, 20%, and 40% auxiliary mixtures against fixed
Spanish-code, English-code, action, and negation holdouts. A candidate wins
only on the Pareto frontier; corpus size alone is never the selection rule.

## Midudev Spanish programming catalog

The catalog snapshot studied on 2026-08-09 is:

- Repository: `midudev/libros-programacion-gratis`
- Commit: `f62f847d14af3b1931850068a5d65d4bbf4c186f`
- Archive SHA-256:
  `02280918b1c1dab50f2a719eaadbf74875b49b4dafc1f063091ece52eb0d44c0`
- Catalog: 115 resources in 32 sections
- Bundled assets: 59 PDF and 58 EPUB files in the pinned snapshot

Of the 59 PDFs, 58 yielded at least 500 characters from their first 20 pages.
A conservative lexical first pass classified 12 as likely CC BY-SA, 12 as
CC BY-NC-SA, 11 as CC BY-NC-ND, 11 as another CC BY variant, 6 as needing
manual license review, 6 as unknown in the inspected pages, and 1 as all rights
reserved. These are triage counts, not legal conclusions; the original license
and any third-party exclusions still require manual confirmation.

The repository prioritizes resources that are legal and free to read, but the
`LibraryBook` schema has no license field. Inspect each document and its
original source before corpus use. The first exercise-oriented audit found:

| Document | Declared terms | Calibration use |
| --- | --- | --- |
| Python basic solved exercises | CC BY-NC-SA 4.0 | Evaluation only |
| Java programming exercises | CC BY-NC-ND 3.0 ES | Evaluation only; no adaptation |
| Haskell/Python exercises | CC BY-NC-SA 2.5 ES | Evaluation only |
| C introduction | CC BY-NC-SA 2.5 ES | Evaluation only |
| Small Go book | CC BY-NC-SA 4.0 | Evaluation only |
| PHP through examples | CC BY-SA 2.5 ES | Research candidate pending derived-artifact review |
| Learning Rust | CC BY-SA | Research candidate pending source-level provenance |

The catalog is valuable because it identifies missing Spanish language and
topic coverage. It is not itself permission to train. Any future book builder
must emit source URL, document hash, detected license, page/section location,
pair extraction method, and a redistribution class for every record.
