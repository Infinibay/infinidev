# Exact archive retrieval implementation plan

Base:b976468. Contract:../specs/2026-11-15-exact-archive-retrieval.md. Owner:Infinidev;
Ken read-only review; Root alone schedules live installation. No Strata, prompt
policy, eviction, checkpoint semantics, bootstrap or live DB access for this work.

- [x] Codec/storage RED: Unicode pages/full SHA, same-title distinct artifacts,
  deduplicated occurrences, empty/EOF/limits; malformed/missing/corrupt parts and
  rollback after several inserts before manifest. Implement stdlib codec in
  engine/archive_output.py and WorkingMemory adapter using existing retry ledger.
  All chunks+manifest commit together; raw immutable records may be reused by
  content only after exact match. Manifest kind excluded from semantic search.
- [x] Public tool RED: optional reference route with strict ranges/session scope,
  legacy search untouched, full record retrieval labeled memory_record, structured
  exact responses under serialized byte budget. No search/embedding/model on
  reference mode; validate/return same reconstruction. Add record references to
  legacy excerpt provenance while preserving its text/type.
- [x] Runtime RED: archive returned tool outputs via stable manifest references;
  keep runtime_trace current, storage error/oversize falls back to legacy path.
  Original results/pairs/closure/cancellation and iteration budgets unchanged.
- [x] Ken+fresh source review, focused/full locked Python3.13 suite, match tested
  src/tests hashes. Publication receipt records commit/push, frozen archive/hash
  manifest and installation instructions separately from the tested source.

Intended source scope: engine/archive_output.py(new), engine/working_memory.py,
tools/meta/recall_context_tool.py, engine/turns/runtime.py. No DB migration,
additional dependency, index or knowledge subsystem. Shared problemas.md append
only for distinct results; no books/private-history/DB artifacts committed.

Validation ledger: initial core/public RED22failed; corrected an initial schema
test that had accidentally failed on missing legacy query rather than the tested
range. Core/old-memory50passed. Runtime RED1failed before stable references;
integration then caught two existing diagnostic-count regressions because generic
host rejection reports were getting manifests too. Manifests now wrap only
granted tool outputs; trace/host diagnostics keep their prior archival path and
existing oracles were retained. Exact reference replies skip recursive archive/
footer so their serialized JSON remains intact.

First full6012passed/14skipped/0failed. Review then found unsupported wm-record
advertisements for non-UUID traceable IDs, empty schema input regression and
manifest exposure via exact-title legacy search;3RED before their minimal fixes.
Final164focused; full6015passed/14skipped/0failed/11warnings,172.44s on locked
Python3.13. All1140source/test files match the retained test worktree by SHA256.
Both reviewers closed the delta without blockers; neither ran tests/Strata.
No live installation, model/provider flag change or causal quality claim.
