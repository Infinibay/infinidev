# Foreign Archive Parts Observer Fix

> Execute inline with TDD; independent source review by Ken.

**Goal:** Observe runB despite legitimate archived provider parts from older runA.
**Spec:** Root approved strict bounded producer-title attribution only to exclude
foreign parts; no authenticity/completeness/event-count claims. Selected, unknown,
corrupt or contradictory records remain unavailable/null. No body reconstruction.
**Architecture:** Project bounded legacy archive title/part size plus current JSON
metadata. Apply strict structural ownership before failing a foreign partial row.
No writer/runtime/DB change, extra scanning or budget increases.

Files: engine/result_observation.py, tests/test_result_observation.py,
docs/SESSION_STATUS.md.

- [ ] Reproduce RED with real _archive→remember, A8003/8024 and B2results.
- [ ] Gate selected/unknown/corrupt/contradictory labels, windows and size limits.
- [ ] Implement bounded title grammar; record foreign part rows skipped in coverage.
- [ ] Focused tests, independent review and full suite/source equality.
- [ ] Commit/push and prepare verified candidate, no live install/model.

Counts remain retained-window observations. A part is not an event; no concatenation
or reconstruction of model messages, arguments, results or reasoning. Invalid data
without a defensible foreign attribution cannot become a successful zero.
