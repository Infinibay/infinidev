# Ken code search transport plan

Goal: decode modern Owner/Piece SearchResponse without losing source or authority; preserve legacy calls. User authorized bounded implementation, fixtures agreed with Ken. No live install, Strata, indexing, prompt change or knowledge-store change.

Design: preserve server order and complete row/envelope metadata independently. Modern ranking_score is not cosine; missing score stays unknown. A list-compatible response retains diagnostics even when empty. Only unavailable/malformed transport permits explicitly marked file fallback; legitimate empty/preparation errors do not become synthetic hits.

Files: ken_client.py decodes/renders; CodeFileHit subclasses McpHit locally with optional metadata/nullable score; generic mcp_client.py stays unchanged; search_symbols_tool.py carries empty/error diagnostics to its public result. Frozen source-derived fixtures have provenance and are not endpoint captures.

- [x] Write offline Owner/Piece/legacy/stale/empty/error/extension/consumer regressions and observe RED.
- [x] Implement the bounded decoder and preserve per-result/global truncation and provenance, without guessing fresh source.
- [x] Run focused tests and independent Ken review; address blockers.
- [x] Run full Python3.13 suite, verify source hashes, commit and push only the reviewed files.

Non-goals: ken_session recent-finding expansion, retrieval ranking, enabling Ken in Bruma, actual source revalidation in Infi, or claiming a cause of Bruma errors.

## Validation

Initial regression25FAIL/2PASS; legacy single-record2FAIL→GREEN; public error JSON1FAIL→GREEN. Focused final138PASS. Independent Ken review found no contract blockers (read-only; no independent runtime test). Full Python3.13 suite5917PASS/14SKIP/0FAIL,11warnings,150.54s, with test-only discovery isolation and no environment modification.

Earlier full attempts were incomplete or failed because of the external harness (PYTHONPATH injection, then missing multiprocessing __main__ guard). Neither was fixed in product or tests. Both affected tests pass with the corrected guarded runner. Logs and isolation source are retained in the candidate receipt.
