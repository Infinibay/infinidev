# Bundled embedding artifact

`ken__static-qwen3-r512-v2.npz` is the `ken/static-qwen3-r512-v2` static
embedding table distributed by [Infinibay/ken](https://github.com/Infinibay/ken).
Ken and Infinidev are MIT-licensed Infinibay projects.

The table was distilled from `Qwen/Qwen3-Embedding-0.6B`, whose model card
declares Apache-2.0. The artifact metadata preserves that teacher identifier;
keep both this provenance note and the project license notices when packaging.

The file is self-contained: it includes Qwen3's tokenizer, a token-to-row lookup
table, a quantized rank-512 token table, the 512x1024 projection, and provenance
metadata. Infinidev vendors it so semantic indexing works offline and does not
depend on a separately installed Ken CLI.

`ken__static-qwen3-r512-v2-es-query-adapter.npz` is an asymmetric sparse
Spanish-query residual bound to the exact table above. It is applied only to
queries selected as Spanish; stored passages and other queries remain in the
original v2 space. Its embedded metadata records the parent SHA-256, language
selector, source adapter hashes, 0.25 blend weight, and 19,567 residual rows.
The artifact SHA-256 is
`33f8424261c08377d830601ddceccbbf6a805dd6e848114a4e649da37c5d7b84`.
See `docs/static-qwen3-calibration.md` for source boundaries and held-out
retrieval results.

Do not replace the file without changing the model identifier and rerunning
`bench/static_qwen3_calibration.py`. Cosine similarity is meaningful only among
vectors produced by the exact same embedding space.
