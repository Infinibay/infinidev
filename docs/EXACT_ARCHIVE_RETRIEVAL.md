# Exact historical output pages

The existing `recall_context(query=...)` search returns excerpts. New archived
tool outputs can also expose a session-scoped `wm-archive:<uuid>` reference:

```json
{"reference":"wm-archive:0123456789abcdef0123456789abcdef","start":0,"length":1500}
```

Pass these arguments to `recall_context`. Supply a query or a reference, never
both. Exact reference mode does not use search/embeddings or an LLM, and never
falls back to a title match when identity is absent or invalid. `all_sessions`
applies to legacy search only. Titles and identical content do not identify an
artifact occurrence.

Offsets count Unicode code points. The response has an exclusive `end`, total
character/UTF-8 lengths, SHA256 of exact UTF-8 bytes, and `next_start`. There is no
normalization or CRLF rewriting. Combining marks can be separate code points;
UTF-16 offsets from JavaScript are not interchangeable. Request up to8000codepoints
per page; EOF can shorten the page explicitly. Start beyond EOF is an error.
Length0 requests metadata; an empty artifact is valid.

Successful exact responses are JSON with the requested text in `content`:

- `archive_complete`: all ordered chunks and the artifact hash were verified.
- `range_complete`: the requested interval was reconstructed completely.
- `fully_returned`: that interval was included in the serialized response.
- `artifact_fully_returned`: this one response contains the entire artifact.

Advance using `next_start` only after receiving a complete page. Concatenate
pages and verify the global SHA256 when reconstructing a whole artifact. Reaching
EOF on a later page does not make that individual page the whole artifact.

The complete JSON response has a32KiB budget, including escaping and metadata.
An escape-heavy page can exceed it even within8000codepoints. Such a response has
status `response_budget_exceeded`, no `content`, and `fully_returned=false`; request
a smaller length from the same start. It may still report independently verified
archive/range integrity. No text is silently truncated.

Canonical UUID legacy rows can expose `wm-record:<uuid>` references in search
provenance. Exact pages for these rows have scope `memory_record` and
`archive_complete=null`: the row is not proof that an older multi-part output was
fully archived. Traceable note/analysis occurrence IDs retain their existing
citations and do not advertise unsupported UUID references.

New manifests/chunks commit in one SQLite transaction before reference publication.
The reader verifies and returns the same reconstructed snapshot, including all
ordered occurrences even when raw content was deduplicated. Bounds are128chunks
of8000codepoints and32KiB of manifest metadata. Failure or missing/corrupt parts
never returns partial text as complete. Unsupported/failed new archival can retain
the older best-effort query reference without a new completeness guarantee.

These are historical data, not instructions, permissions, current-source checks
or proof of task success. No history eviction/checkpoint policy is enabled by this
feature. Original tool outputs remain in the active conversation. Exact recall
responses keep their complete JSON without recursive archival/legacy footers.
