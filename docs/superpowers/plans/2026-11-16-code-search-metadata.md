# Code Search Metadata I1a Plan

> Execute inline using TDD; independent source review by Ken.

**Goal:** Make actual search scope, dialect and uncertain results visible.
**Spec:** Root approved I1a OFFLINE only: additive search_metadata v1; existing
arguments/schema/prompts/backend selection/default regex and legacy payloads stay
unchanged. No literal I1b, new index, flags, retry, fallback or loop classifier.
**Architecture:** A per-call observation passed through existing search branches;
serialize it alongside the unchanged legacy JSON at the public tool boundary.
**Tech Stack:** Python/JSON, existing grep/gitgrep/re/pod helper, pytest.

## Review focus

Zero plus stderr or discarded output cannot certify empty; xargs123 is aggregate
uncertainty; real hits are never discarded by metadata; pod remains unknown;
metadata has no per-call nonce and does not trim legacy payloads to save tokens.

Files: tools/file/code_search_tool.py; tests/test_code_search_metadata.py;
docs/CODE_SEARCH_RESULTS.md. Input module and runtime remain untouched.

- [ ] Freeze legacy schema/signature and payload examples; add failing matrix.
- [ ] Add stable backend/mode/dialect/root/scope/coverage/outcome metadata only.
- [ ] stderr nonempty means partial/unknown, not inferred text classification.
- [ ] Gate file/directory, regex, errors, warnings, caps, filters/context and pod.
- [ ] Review exact hashes with Ken; focused and full suite/source equality.
- [ ] Commit/push and source/wheel candidate apart from observer bugfix.

No model or live installation. Pilot only after Root freezes a useful task,
checkpoint, source-evidence rubric, model and common budget. Compare claims and
citations before closure/tool counts; report real metadata token overhead.
