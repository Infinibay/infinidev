# Fixed-order versus counterbalanced MCQ report

Fixed-order and balanced runs differ in both presentation and sample count. A changed or tied canonical mode reveals sensitivity but does not identify a single causal token bias. Provider letters, mappings, and canonical actions are retained separately.

## Model: Sol

Modal relations: `{"changed_unique": 1, "same_unique": 7}`.
Displayed provider-letter counts in the balanced run: `{"A": 13, "B": 5, "C": 8, "D": 6}`.

### `preference-implementation-reuse-v2` — same_unique

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Implement the isolated 90-line component behind the common interface and keep duplication explicit.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Implement the isolated 90-line component behind the common interface and keep duplication explicit.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-interrupt-atomicity-v2` — same_unique

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
- **balanced** canonical counts: `{"A": 3, "D": 1}`; modal actions: ['Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.']; exact stability: False.
  - r0: provider **A** -> canonical **D** — Complete the remaining atomic rename, then respond with the finished local state clearly reported.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **D** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **C** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-recovery-fallback-v1` — same_unique

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Switch immediately to direct source/text search and document the changed evidence channel.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
- **balanced** canonical counts: `{"A": 4}`; modal actions: ['Switch immediately to direct source/text search and document the changed evidence channel.']; exact stability: True.
  - r0: provider **B** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **D** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **C** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-requirements-iteration-v1` — same_unique

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{}`.
- **balanced** canonical counts: `{"B": 1, "C": 2, "D": 1}`; modal actions: ['Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.']; exact stability: False.
  - r0: provider **A** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **A** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-test-scope-v1` — same_unique

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Accept the focused and impacted tests as sufficient, document their coverage, and stop.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
- **balanced** canonical counts: `{"A": 3, "D": 1}`; modal actions: ['Accept the focused and impacted tests as sufficient, document their coverage, and stop.']; exact stability: False.
  - r0: provider **D** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r1: provider **C** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r2: provider **A** -> canonical **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r3: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.

### `preference-uncertainty-presentation-v2` — same_unique

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Give the recommendation with one concise confidence qualifier and name the tested fallback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
- **balanced** canonical counts: `{"B": 3, "D": 1}`; modal actions: ['Give the recommendation with one concise confidence qualifier and name the tested fallback.']; exact stability: False.
  - r0: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **A** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **D** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **A** -> canonical **D** — Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-vague-prototype-v2` — same_unique

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Build one throwaway interactive prototype using existing components and collect concrete feedback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Build one throwaway interactive prototype using existing components and collect concrete feedback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **A** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **D** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **C** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-verification-independence-v2` — changed_unique

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
- **fixed** canonical counts: `{"B": 2, "C": 1}`; modal actions: ['Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.']; exact stability: False.
  - r0: provider **B** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{}`.
- **balanced** canonical counts: `{"B": 1, "C": 3}`; modal actions: ['Complete with the primary proof and give the user the exact independent check as an optional command.']; exact stability: False.
  - r0: provider **C** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

## Model: Terra

Modal relations: `{"balanced_tie_contains_fixed": 1, "changed_unique": 1, "same_unique": 6}`.
Displayed provider-letter counts in the balanced run: `{"A": 12, "B": 7, "C": 7, "D": 6}`.

### `preference-implementation-reuse-v2` — same_unique

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Implement the isolated 90-line component behind the common interface and keep duplication explicit.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Implement the isolated 90-line component behind the common interface and keep duplication explicit.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-interrupt-atomicity-v2` — same_unique

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
- **balanced** canonical counts: `{"A": 4}`; modal actions: ['Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.']; exact stability: True.
  - r0: provider **B** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **D** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **C** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-recovery-fallback-v1` — same_unique

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Switch immediately to direct source/text search and document the changed evidence channel.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
- **balanced** canonical counts: `{"A": 4}`; modal actions: ['Switch immediately to direct source/text search and document the changed evidence channel.']; exact stability: True.
  - r0: provider **B** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **D** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **C** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-requirements-iteration-v1` — changed_unique

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
- **fixed** canonical counts: `{"B": 2, "C": 1}`; modal actions: ['Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.']; exact stability: False.
  - r0: provider **B** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{}`.
- **balanced** canonical counts: `{"B": 1, "C": 2, "D": 1}`; modal actions: ['Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.']; exact stability: False.
  - r0: provider **A** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **A** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-test-scope-v1` — same_unique

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Accept the focused and impacted tests as sufficient, document their coverage, and stop.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
- **balanced** canonical counts: `{"A": 2, "C": 1, "D": 1}`; modal actions: ['Accept the focused and impacted tests as sufficient, document their coverage, and stop.']; exact stability: False.
  - r0: provider **D** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r1: provider **A** -> canonical **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r2: provider **A** -> canonical **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r3: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.

### `preference-uncertainty-presentation-v2` — same_unique

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Give the recommendation with one concise confidence qualifier and name the tested fallback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Give the recommendation with one concise confidence qualifier and name the tested fallback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **A** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **D** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **C** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-vague-prototype-v2` — same_unique

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Build one throwaway interactive prototype using existing components and collect concrete feedback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Build one throwaway interactive prototype using existing components and collect concrete feedback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **A** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **D** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **C** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-verification-independence-v2` — balanced_tie_contains_fixed

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
- **fixed** canonical counts: `{"B": 2, "C": 1}`; modal actions: ['Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.']; exact stability: False.
  - r0: provider **B** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{}`.
- **balanced** canonical counts: `{"B": 2, "C": 2}`; modal actions: ['Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.', 'Complete with the primary proof and give the user the exact independent check as an optional command.']; exact stability: False.
  - r0: provider **C** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **B** -> canonical **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

## Model: Luna

Modal relations: `{"balanced_tie_contains_fixed": 2, "changed_unique": 1, "incomparable_or_excluding_tie": 1, "same_unique": 4}`.
Displayed provider-letter counts in the balanced run: `{"A": 12, "B": 9, "C": 7, "D": 4}`.

### `preference-implementation-reuse-v2` — same_unique

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Implement the isolated 90-line component behind the common interface and keep duplication explicit.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Implement the isolated 90-line component behind the common interface and keep duplication explicit.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-interrupt-atomicity-v2` — balanced_tie_contains_fixed

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
- **balanced** canonical counts: `{"A": 2, "D": 2}`; modal actions: ['Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.', 'Complete the remaining atomic rename, then respond with the finished local state clearly reported.']; exact stability: False.
  - r0: provider **B** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Complete the remaining atomic rename, then respond with the finished local state clearly reported.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Complete the remaining atomic rename, then respond with the finished local state clearly reported.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-recovery-fallback-v1` — balanced_tie_contains_fixed

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Switch immediately to direct source/text search and document the changed evidence channel.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{}`.
- **balanced** canonical counts: `{"A": 2, "C": 2}`; modal actions: ['Switch immediately to direct source/text search and document the changed evidence channel.', 'Complete with direct evidence, then compare semantic results later if they become available.']; exact stability: False.
  - r0: provider **B** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Switch immediately to direct source/text search and document the changed evidence channel.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Complete with direct evidence, then compare semantic results later if they become available.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Complete with direct evidence, then compare semantic results later if they become available.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-requirements-iteration-v1` — same_unique

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{}`.
- **balanced** canonical counts: `{"B": 3, "C": 1}`; modal actions: ['Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.']; exact stability: False.
  - r0: provider **C** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **B** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **A** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-test-scope-v1` — incomparable_or_excluding_tie

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
- **fixed** canonical counts: `{"A": 1, "C": 1, "D": 1}`; modal actions: ['Accept the focused and impacted tests as sufficient, document their coverage, and stop.', 'Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.', 'Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.']; exact stability: False.
  - r0: provider **C** -> canonical **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; mapping `{}`.
- **balanced** canonical counts: `{"C": 2, "D": 2}`; modal actions: ['Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.', 'Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.']; exact stability: False.
  - r0: provider **C** -> canonical **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r1: provider **A** -> canonical **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r2: provider **A** -> canonical **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r3: provider **C** -> canonical **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.

### `preference-uncertainty-presentation-v2` — same_unique

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
- **fixed** canonical counts: `{"A": 1, "B": 2}`; modal actions: ['Give the recommendation with one concise confidence qualifier and name the tested fallback.']; exact stability: False.
  - r0: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Lead with the recommendation and qualifier, then add a compact expandable risk analysis.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Give the recommendation with one concise confidence qualifier and name the tested fallback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **A** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **D** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **C** -> canonical **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-vague-prototype-v2` — same_unique

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Build one throwaway interactive prototype using existing components and collect concrete feedback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Build one throwaway interactive prototype using existing components and collect concrete feedback.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **A** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **D** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **C** -> canonical **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-verification-independence-v2` — changed_unique

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
- **balanced** canonical counts: `{"A": 1, "C": 3}`; modal actions: ['Complete with the primary proof and give the user the exact independent check as an optional command.']; exact stability: False.
  - r0: provider **D** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
