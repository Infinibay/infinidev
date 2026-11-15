# Exact archived output retrieval

Status: implemented/validated offline, reviewed by Ken and an independent reviewer.
Installation/live verification remain deferred to Root. Authorized offline
scope: retrieval and integrity only; no eviction, semantic checkpoint, prompt
policy, live installation, process changes or Strata inference.

## Decision and compatibility

Reuse WorkingMemory's private SQLite ledger, its raw output chunks and DB retry
helper. Add an immutable manifest referencing ordered chunk occurrences. A new
filesystem store or DB migration is unnecessary. Exact row lookup alone would
fix the1500-character render limit but would not establish whole-output custody;
the manifest is therefore included in this prerequisite.

Keep `recall_context(query, limit, all_sessions)` and its excerpt rendering.
Extend the schema with explicit `reference`, `start` and `length`; query becomes
optional only to allow reference-only calls. Nonempty query plus reference is an
error. An exact-reference miss never falls back to title/semantic search.
Missing both query and reference remains a schema error; explicitly empty legacy
query retains its old behavior. Only canonical UUID ledger IDs advertise wm-record
references; traceable note/analysis occurrence IDs retain their existing citations.
References are session-scoped; all_sessions remains a legacy search option,
rejected with reference mode. No original title or user text becomes identity.

Two explicitly different units:

- `wm-archive:<uuid>` identifies one complete original tool-output artifact.
- `wm-record:<uuid>` identifies one legacy ledger row, not the whole output that
  may have been split across other rows. Its page response says memory_record;
  it never retrospectively certifies an old multi-part artifact.

The runtime issues new archive references only after successful durable commit.
Existing query references continue to work. Archived runtime_trace records retain
their current path; no private reasoning is added to the ledger or artifacts.
Exact retrieval responses already carry a durable reference: runtime will not
recursively re-archive them or append its legacy footer to their complete JSON.
Legacy query mode and other tool responses retain their existing annotation path.

## Ranges and Unicode

Offsets count Unicode code points in the exact stored Python string, not UTF-8
bytes, UTF-16 units or grapheme clusters. Start is zero based; response uses an
exclusive end. No Unicode normalization, newline rewriting or replacement of
invalid characters. SHA256 covers exact UTF-8 bytes; unencodable surrogates cause
a controlled archival failure, never an agent-result failure.

`start` defaults0 and must be an integer in[0,total_characters]. `length` defaults
1500 and is a requested maximum in[0,8000]. Bool/float/negative values are rejected,
not coerced/clamped. The final page ends at min(start+length,total_characters),
explicit in response metadata; start beyond EOF is an error. Length0 is a
metadata-only request; start==EOF returns an empty valid interval. Clients use
end/next_start and never assume the requested maximum was filled at EOF.

## Manifest and durability

Version1 manifest: reference/session, historical-data provenance/title, total
codepoints, total UTF-8 bytes, global SHA256 and ordered occurrences with index,
record_id, start, length and SHA256. Chunk bodies remain at most8000codepoints.
Different occurrences may share a content-deduplicated record ID, but each retains
its position and digest. New immutable manifests use a distinct ledger kind and
are excluded from legacy semantic search. Visible titles may collide freely.

Writer commits all chunks and the manifest in one SQLite transaction through
execute_with_retry. An existing content-addressed row can be reused only if its
actual content matches. Commit is the publication point: no stable complete
reference is returned earlier. Failure rolls back; no partial manifest is called
complete. Observational storage failure preserves the original tool result and
can retain the existing legacy best-effort reference without completeness claims.

Bounds:128chunks/artifact (at most1,024,000codepoints),32KiB serialized manifest,
8000codepoints/page,32KiB complete serialized response. Exceeding bounds is an
explicit unavailable/limit error, never silent truncation. The existing legacy
archive path remains available for unsupported/oversized outputs without a new
integrity guarantee.

## Reader and response

An exact read uses current session and stable identity, never search/embedding,
an LLM or engine/session bootstrap. Codec/validation is stdlib-only; the storage
adapter uses the existing WorkingMemory/DB layer. Tests invoke it against an
already-created private ledger without starting the engine or model. No public
standalone CLI, new index or external endpoint is required by this change.

Before positive whole-artifact integrity, validate manifest schema/bounds and all
ordered occurrences: session-bound row identities, contiguous coverage, lengths,
per-piece hashes and global reconstructed UTF-8 length/hash. Missing/altered parts,
gaps, malformed references, exhausted budgets or partial failures cannot expose
partial text as a complete page. No cache silently trusts prior validation.
Validate and return from the same reconstructed snapshot; never validate one read
and then re-fetch unverified bodies for the page.

Reference-mode JSON separates:

- archive_complete: entire manifest/artifact integrity verified, independent of
  requested page; null for a legacy memory_record with no artifact manifest.
- range_complete: the defined requested interval was reconstructed completely.
- fully_returned: that complete interval was actually included in the final
  serialized response within its byte budget; no hidden render truncation.
- artifact_fully_returned: entire artifact is present in this one response;
  never inferred merely from reaching EOF on a later page.

Include scope/reference/start/end/next_start, total lengths, SHA256 and explicit
historical-data/non-instruction provenance. Successful page JSON contains exact
content. Budget failure returns no content and fully_returned=false; it may report
archive_complete/range_complete=true if independently verified before encoding.
Other failures return fixed error codes without raw DB/error text or excerpts.
Storage integrity and declarations never certify current source or task success.

## Acceptance before publication

1. Public tool round-trip >8000characters with astral Unicode, combining marks,
   CRLF, controls and repeats; concatenate pages and verify literal equality/SHA.
2. Distinct artifacts with identical visible titles remain independently exact;
   repeated identical chunks preserve ordered occurrences, not deduplicated gaps.
3. Failed chunk/manifest writes roll back; missing/corrupt/wrong-session chunks,
   malformed/overlapping/duplicated occurrence descriptors never claim complete.
4. Stable reference absent/cross-session/corrupt never invokes semantic fallback.
5. Invalid integer/range inputs, empty data, EOF, length0 and JSON-escaped response
   budget all have explicit behavior. No partial serialized page is called full.
6. Legacy query behavior and record-only exact reads remain distinct; exact
   reads call no search/embedding/model/bootstrap.
7. Runtime tool-call/result structure, literal bodies, cancellation, hook budgets
   and closure remain unchanged. Archive failure cannot fail the agent's result.
8. Focused RED→GREEN, Ken format/source review, full locked Python3.13 suite,
   frozen candidate and commit/push. No installation or causal quality claim.

Review precision: inject failure after multiple chunk inserts but before manifest
commit, including a preexisting deduplicated row; only new work rolls back. Empty
artifact with zero occurrences is valid. Repeated IDs for identical chunk bytes at
different offsets are valid; malformed indices/coverage duplicates are not.
