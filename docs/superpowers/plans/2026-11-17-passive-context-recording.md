# Passive per-execution context recording

Root authorization: opt-in host metadata tied to real principal assembly, no
body/COT/secrets, no model calls, no prompt changes or live restart. Ownership
Infinidev; Root schedules deployment/inference. Existing worktree7e28 is retained.

## Interface and scope

Explicit SDK argument run_task(record_context_dir=Path(...)), default None. One
private directory outside the workspace per invocation; no persistent setting,
new model tool, DB table or note elicitation. CLI/TUI can forward the same option
without sticky session state; first implement/test the real SDK boundary.

Metadata: source IDs/role/origin/order, redacted projection hash and clipping,
effective tool/schema snapshot, child/request linkage and existing typed result
facts. Public notes only if already emitted. Never classify authority/intent from
prose. Supplied history is data; a correct old grant is a negative control.

Known complete mappings only: transformations/compaction or unavailable provenance
are marked unknown/changed, not reconstructed with content substring matching.
Hash scope is explicit; no raw argument/model reasoning/user/source/book bodies
or environment/headers. Private/unknown content categories omit their digest.

## Implementation sequence

1. RED writer/disabled/symlink/collision/bounds tests and real run_task→run_turn
   fake-provider test with known selected DB history IDs and exact unchanged
   provider kwargs. Test two child continuations, cancellation/failed archival
   and history clipping/default compatibility.
2. Small contextvar-scoped metadata writer with allowlisted records, private
   exclusive files, bounded records/total bytes and incomplete marker on failure.
   Preflight opt-in errors stop before model/tools; mid-run I/O cannot rerun or
   alter the task. No global/session capture.
3. Reuse DB recent-history query for an ID-bearing internal view while keeping
   get_recent_turns_full return values/default path compatible. Real renderer
   reports selected IDs and loader/renderer clipping without saving content.
4. Wire parent context layers/system fragments and actual request snapshots.
   Preserve all prompt text/order/schemas; trace mappings carry host-only data
   removed before provider dispatch. Reuse existing runtime event facts.
5. Focused/default tests, exact independent privacy/source review, full suite,
   deployed-core contract gate when relevant. Commit/push only verified files.
   Root receives exact candidate and limitations before any deployment.

## Acceptance

Disabled: no recording I/O or extra history query and byte-identical provider
input. Enabled: real principal input selection includes known history IDs,
roles/order/clipping and exact effective grant; repeated identical text has
distinct source identities. Literal history/notes/messages stay unchanged.
Unknown mappings are not called complete. No model/private/argument bodies in
records, including nested unknown blocks and failure paths. Recording does not
certify model understanding or semantic progress.

## Not part of this change

No semantic history rewrite, prompt treatment, new decision-note requirement,
microexperiment, automatic retry/limits change, live tracing or installation.
