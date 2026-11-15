# Bounded /init prompt, first offline slice

The old prompt requires Ken storage using local RecordFinding arguments, numeric
certainty 1.0, automatic documentation downloads and exhaustive exploration. The
toolbox may contain no finding writer. The first slice removes these assumptions
without changing the existing document flow, engine, storage or command syntax.

## Implemented boundary

- A pure builder receives initial tool names, without discovery, settings, I/O or
  model calls. Ken advice uses the Ken schema; local advice uses the local schema;
  no writer means a public delivery without a database-persistence claim. If both
  are initially available, prefer the existing intended Ken path, without mixing
  arguments or duplicating the conclusion in the local backend.
- Routing/discovery can change tools later. The final offered schema prevails;
  failed/unavailable storage is reported as unrecorded, without silent switching.
  This is prompt guidance, not a host-enforced persistence transaction or budget.
- The builder supports an explicit language string as JSON data. The unchanged
  slash-command interfaces pass no language option: guidance uses the user's
  explicit conversation preference and asks when none is known. No language
  detector, locale setting or hidden inference from source language is added.
- Source observations, documented intent, hypotheses and unknowns stay distinct.
  No source reading implies runtime correctness, current validity or security
  certification. The task does not authorize builds, source edits, indexing or
  external downloads.
- CLI and TUI compose the prompt from their initial toolbox. TUI composition is
  inside its existing try/finally so failures still clear running state. Original
  INIT_TASK_DESCRIPTION/INIT_EXPECTED_OUTPUT imports remain capability-neutral.

The shared design for incremental coverage/backend identity remains a proposal at
`~/.cache/infinidev-validation/reviews/init-path-and-incremental-design-2026-11-15.md`.
This change does not implement its state, migration, aliases or principal routing.
I/O receipts and retained principal contracts are separate work.

Review found that the document system identity also mandated knowledge/web
lookups and mixed local arguments with Ken. The first user-prompt-only change
therefore did not resolve the assembled conflict. The document identity and its
coding/generalized/extra_simple variants now honor task-authorized sources and
destinations and each offered schema. Lean falls back to the corrected full
identity. Other flow identities, routing and style selection are unchanged.

## Validation

Five composition regressions first failed for the missing builder. They cover
absent/Ken/local/both finding writers and Unicode language with a temporary HOME
without books. These verify prompt payload selection and encoding, not model
obedience. Existing flow/command tests are included in the focused set.

Initial focused run: 43 passed. Ten system-assembly cases (five document styles,
two model-size paths) then reproduced the missing system-level boundary. After
the identity fix, the expanded focused set passed 780 tests. An exploratory
assembly test initially attempted an Ollama capability lookup (connection
refused); its fixture now uses synthetic already-probed capabilities and no
provider lookup. No inference occurred. The first full run had 6023 passed,
14 skipped and one prompt-style failure in two new phrases; those phrases were
made concrete and the style gates pass in the expanded focused run.

Final suite: 6034 passed, 14 skipped, zero failures, 11 serializer warnings,
385.33 seconds, persistent exit 0. All 1142 source/test files match the tested
worktree by SHA256. Ken closed the source delta without blockers in this prompt
scope; this is not an independent test run. The receipt retains failed exploratory
runs separately. Assembly covers full/lean/coding/generalized/extra_simple and
both model-size paths, with a local read-only toolbox; backend choices are tested
separately in the builder. It does not cover a live model or manual tool calling.

Tests use the retained locked Python3.13
worktree and a test-only guard against external Ken discovery; no Strata calls,
provider changes, books or new indexes are required. No live install/restart.
