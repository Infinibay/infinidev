# Private command-output capture (phase 1)

Infinidev normally returns only the final 10,000 characters of `stdout` and
5,000 characters of `stderr` from `execute_command`. Phase 1 can preserve a
larger stream behind an opaque handle so the agent can retrieve bounded ranges
without putting the full output into the prompt, SQLite full-text search, or
working-memory notes.

The feature is additive and disabled by default. It does not change command
execution, truncation, result ordering, or prompt reconstruction until capture
is explicitly enabled.

## Configuration

Settings may be placed in the project-local `.infinidev/settings.json` or set
with the `INFINIDEV_` environment-variable prefix.

| Setting | Default | Purpose |
|---|---:|---|
| `COMMAND_OUTPUT_CAPTURE_ENABLED` | `false` | Persist an over-limit decoded stream and add a path-free handle to the existing truncated result. |
| `COMMAND_OUTPUT_MAX_ARTIFACT_BYTES` | `10485760` | Maximum UTF-8 bytes for one captured stream (10 MiB). |
| `COMMAND_OUTPUT_MAX_SESSION_BYTES` | `104857600` | Maximum retained bytes for one project/session pair (100 MiB). |
| `COMMAND_OUTPUT_MAX_PROJECT_BYTES` | `524288000` | Maximum retained bytes for one project (500 MiB). |
| `COMMAND_OUTPUT_STORE_TIMEOUT_SECONDS` | `5` | Deadline for locking, sweeping, writing, verifying, and cataloguing. |
| `COMMAND_OUTPUT_RETENTION_SECONDS` | `604800` | Retain a valid capture for seven days. |
| `COMMAND_OUTPUT_SWEEP_GRACE_SECONDS` | `3600` | Minimum age before reclaiming orphaned files or dangling rows. |
| `COMMAND_OUTPUT_AUTO_NOTES_ENABLED` | `false` | At step close, create a traceable note containing the step summary and opaque artifact identity, never output text. |
| `COMMAND_OUTPUT_NOTE_COMPACTION_ENABLED` | `false` | Compact multiple newly-created closure notes while preserving source notes and ordered citations. |

Every numeric bound is mandatory and must be a positive integer whenever
capture is enabled. The session limit must be at least the artifact limit, and
the project limit must be at least the session limit. Invalid limits, quota
exhaustion, lock timeout, write failure, verification failure, or catalog
failure disable capture for that command: the caller still receives the exact
legacy truncated result and no handle is announced.

## Storage and catalog contract

Captured data is stored below:

```text
.infinidev/private/command_output/
```

The directory is mode `0700`; blobs, sidecars, temporary files, and the lock
are mode `0600`. Symlinked roots/components, traversal references, non-regular
files, substitutions, hash mismatches, unsafe permissions, and changing files
are rejected rather than followed or partially trusted.

The existing `artifacts` table is used only as an opaque catalog. A
`command_output` row has:

- the owning `project_id` and `session_id`;
- `type = 'command_output'`;
- an opaque random storage reference in `file_path` (not a filesystem path);
- `name`, `description`, and `content` set to `NULL`.

Consequently command text, output, blob bytes, sidecar JSON, and seeded secrets
are absent from `artifacts.content`, indexed metadata, FTS, previews, working
memory, and reconstructed prompts. There is no canonical schema migration.
The blob is intentionally **not encrypted** in phase 1; filesystem permissions
and the host's disk encryption are the confidentiality boundary.

Each sidecar uses format version `1` and binds the random storage identity,
project, session, stream, byte/character lengths, creation time, and SHA-256 of
the blob. The catalog reference also binds the sidecar digest. Reads require an
exact match across project, session, artifact ID, artifact type, stream, byte
count, and character count.

## Text semantics and bounded reads

Capture preserves the complete decoded Python `str` immediately before the
legacy character slice. It can therefore reconstruct that pre-cut **text**
exactly. It does not claim byte-for-byte preservation of the subprocess's
original stream: the sealed path uses text decoding and the interactive path
uses replacement decoding for malformed bytes. Binary output and lossless raw
byte capture are outside phase 1.

The handle contains only:

```json
{
  "artifact_id": 123,
  "type": "command_output",
  "stream": "stdout",
  "char_count": 25000,
  "byte_count": 25042
}
```

`read_command_output` requires those fields unchanged plus a UTF-8 byte
`offset` and `limit`. One call is capped at 65,536 source bytes. Start and end
must be character boundaries; the response returns `returned_end`, `has_more`,
and `next_offset` so repeated bounded reads reconstruct the text without gaps
or duplication. No storage path is accepted or returned, and a handle from a
different project or session fails closed.

## Traceable notes and explicit analysis

Traceable working-memory notes are immutable JSON envelopes with schema
`infinidev.traceable_note`, version `1`, an occurrence identity, source
artifact, step/tool-call identity, generation, ordered parent identities, and
ordered citations. Equal summaries from different occurrences remain distinct.
Compaction creates a new derived note, never updates or deletes sources, and is
idempotent for the same ordered source identities. Derivation depth, parent
count, citation count, identifier length, and summary length are bounded.

Private output is never analyzed automatically. `analyze_command_output()` is
an explicit internal council API. It validates the handle before a model call,
then creates an isolated subagent with only a fixed-handle range reader and a
structured terminator—no paths, shell, files, web, history, knowledge tools, or
ambient project/session context. Reads, bytes, iterations, claims, and citations
are bounded; every claim is marked as fact or inference and must cite an exact
range actually returned. Credentials are redacted before model input and again
before note persistence. Any scope, citation, model, redaction, or persistence
failure produces no analysis note.

## Retention and sweeping

A sweep runs under the same lock before each store operation. It removes valid
captures whose sidecar creation time exceeds the retention period, old managed
orphans/temporary files after the grace period, and old dangling catalog rows.
It ignores unrelated files and does not guess how to repair corrupt metadata.
Operators may also invoke `CommandOutputStore().sweep()` from trusted internal
maintenance code; phase 1 does not add a scheduler or daemon.

## Gradual rollout

1. **Baseline:** leave all three feature flags off. This preserves prior
   `execute_command`, `ToolRunner`, `LoopEngine`, and prompt behavior.
2. **Capture only:** set `COMMAND_OUTPUT_CAPTURE_ENABLED=true` and keep note
   flags off. Confirm quotas and retention fit the host, then verify handles can
   be read only from their owning project/session.
3. **Observe storage:** monitor the private directory size and confirm
   `command_output` catalog rows keep `name`, `description`, and `content` NULL.
   A failed capture must show only the normal truncated output and no handle.
4. **Closure notes:** independently enable `COMMAND_OUTPUT_AUTO_NOTES_ENABLED`.
   Notes contain summaries and opaque identities, not raw command output.
5. **Compaction:** enable `COMMAND_OUTPUT_NOTE_COMPACTION_ENABLED` only after
   note provenance has been observed. Source notes remain immutable.
6. **Explicit analysis:** call the isolated API only for a specific validated
   handle and question. It is never part of normal council execution.

Rollback is immediate: turn the flags off. Existing captures remain inaccessible
through new command results and age out according to retention; run an explicit
sweep if operational policy requires prompt cleanup.

## Deliberate phase-1 exclusions

Phase 1 does not add encryption, binary capture, streaming/backpressure,
deduplication, automatic analysis, cross-session access, path-based reads, a
new database table, or a schema migration. These require separate operational
or product decisions and are not implied by enabling capture.
