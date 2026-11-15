# Ken recent-finding expansion plan

Goal: preserve the available provenance/validity from the existing findings CLI response and explicitly mark budget omissions. Ownership Infinidev; contract/review Ken. No additional recall, prompt architecture, index, live flags/process or Strata.

Contract: CLI stays ken findings --path ROOT --json -n3. Each final escaped finding block is at most1800characters; the entire expansion including wrapper/markers is at most5000. Prefer full content plus metadata; if it does not fit omit the whole conclusion with an exact topic expansion reference and retain complete metadata. If metadata cannot fit either, emit a reference with explicit omissions; never truncate assumptions or metadata JSON. Missing validity is unknown, no fabricated source validity. Long identifiers that cannot fit cause an explicit whole-record omission. Existing brief stays unchanged.

Remove the false Full versions description and unrelated generic one-read guidance; retain only a factual advisory/excerpt wrapper. This changes selected source data representation, not authority.

- [x] Add offline metadata/legacy/states/negation/HTML/Unicode/total-budget/omission/CLI-error fixtures and observe RED.
- [x] Update ken_session.py plus the outdated expanded-finding wording expectation in test_ken_session.py.
- [x] Independent Ken read-only review and focused regressions.
- [x] Full Python3.13 suite using verified guarded test runner outside repository; commit/push the tested files.

The installed Bruma launcher keeps KenOFF. This is a product defect independent of observed Bruma programming quality. No live validation is claimed.

## Validation and review

Initial RED15FAIL/4PASS; independent Ken review identified identity normalization and a one-character total-budget edge, both reproduced before fixing. Final focused144PASS includes67session/expansion cases; full Python3.13 suite5949PASS/14SKIP/0FAIL,11warnings,183.29s. Exact-source baseline replay73f40d6 renders9398characters, loses scope/validity and incorrectly claims full versions; durable evidence under ~/.cache/infinidev-validation/ken-finding-expansion-2026-11-15. Final escaped identities round-trip through standard XML; invalid identities are explicitly omitted. Public JSON uses ASCII escapes while decoded values remain exact. No live Bruma installation/validation.
