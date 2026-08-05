# Fixed-order versus counterbalanced MCQ report

Fixed-order and balanced runs differ in both presentation and sample count. A changed or tied canonical mode reveals sensitivity but does not identify a single causal token bias. Provider letters, mappings, and canonical actions are retained separately.

## Model: Sol

Modal relations: `{"balanced_tie_contains_fixed": 1, "same_unique": 7}`.
Displayed provider-letter counts in the balanced run: `{"A": 14, "B": 6, "C": 6, "D": 6}`.

### `preference-implementation-reuse-v2` — same_unique

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Prototype both boundaries against the contract tests and present measured complexity before selecting.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
- **balanced** canonical counts: `{"C": 4}`; modal actions: ['Prototype both boundaries against the contract tests and present measured complexity before selecting.']; exact stability: True.
  - r0: provider **D** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-interrupt-atomicity-v2` — same_unique

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
- **fixed** canonical counts: `{"A": 1, "C": 2}`; modal actions: ['Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.']; exact stability: False.
  - r0: provider **C** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{}`.
- **balanced** canonical counts: `{"A": 1, "B": 1, "C": 2}`; modal actions: ['Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.']; exact stability: False.
  - r0: provider **D** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **A** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-recovery-fallback-v1` — same_unique

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-requirements-iteration-v1` — balanced_tie_contains_fixed

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
- **fixed** canonical counts: `{"A": 3}`; modal actions: ['Resolve and document every slice and presentation decision before implementing any code.']; exact stability: True.
  - r0: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{}`.
- **balanced** canonical counts: `{"A": 1, "B": 1, "C": 1, "D": 1}`; modal actions: ['Resolve and document every slice and presentation decision before implementing any code.', 'Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.', 'Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.', 'Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.']; exact stability: False.
  - r0: provider **A** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **A** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-test-scope-v1` — same_unique

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Run the complete suite once and investigate any failure before declaring verification complete.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Run the complete suite once and investigate any failure before declaring verification complete.']; exact stability: True.
  - r0: provider **A** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r1: provider **D** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r2: provider **C** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r3: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.

### `preference-uncertainty-presentation-v2` — same_unique

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['State the recommendation and offer either the concise qualifier or full risk table before elaborating.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
- **balanced** canonical counts: `{"C": 3, "D": 1}`; modal actions: ['State the recommendation and offer either the concise qualifier or full risk table before elaborating.']; exact stability: False.
  - r0: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **B** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **A** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **A** -> canonical **D** — Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-vague-prototype-v2` — same_unique

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Ask a focused workflow question first, then choose prototype or specification from the answer.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
- **balanced** canonical counts: `{"C": 4}`; modal actions: ['Ask a focused workflow question first, then choose prototype or specification from the answer.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **B** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **A** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **D** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-verification-independence-v2` — same_unique

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Also run the independent parser and require agreement before completion.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Also run the independent parser and require agreement before completion.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

## Model: Terra

Modal relations: `{"balanced_tie_contains_fixed": 1, "changed_unique": 1, "same_unique": 6}`.
Displayed provider-letter counts in the balanced run: `{"A": 11, "B": 8, "C": 7, "D": 6}`.

### `preference-implementation-reuse-v2` — same_unique

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Prototype both boundaries against the contract tests and present measured complexity before selecting.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
- **balanced** canonical counts: `{"C": 4}`; modal actions: ['Prototype both boundaries against the contract tests and present measured complexity before selecting.']; exact stability: True.
  - r0: provider **D** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-interrupt-atomicity-v2` — balanced_tie_contains_fixed

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
- **fixed** canonical counts: `{"B": 2, "C": 1}`; modal actions: ['Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.']; exact stability: False.
  - r0: provider **B** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{}`.
- **balanced** canonical counts: `{"B": 2, "C": 2}`; modal actions: ['Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.', 'Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.']; exact stability: False.
  - r0: provider **D** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **B** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **A** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-recovery-fallback-v1` — same_unique

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-requirements-iteration-v1` — same_unique

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
- **fixed** canonical counts: `{"A": 2, "D": 1}`; modal actions: ['Resolve and document every slice and presentation decision before implementing any code.']; exact stability: False.
  - r0: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{}`.
- **balanced** canonical counts: `{"A": 2, "B": 1, "D": 1}`; modal actions: ['Resolve and document every slice and presentation decision before implementing any code.']; exact stability: False.
  - r0: provider **B** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **D** -> canonical **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-test-scope-v1` — same_unique

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Run the complete suite once and investigate any failure before declaring verification complete.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Run the complete suite once and investigate any failure before declaring verification complete.']; exact stability: True.
  - r0: provider **A** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r1: provider **D** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r2: provider **C** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r3: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.

### `preference-uncertainty-presentation-v2` — same_unique

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['State the recommendation and offer either the concise qualifier or full risk table before elaborating.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
- **balanced** canonical counts: `{"C": 3, "D": 1}`; modal actions: ['State the recommendation and offer either the concise qualifier or full risk table before elaborating.']; exact stability: False.
  - r0: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **B** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **A** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **A** -> canonical **D** — Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-vague-prototype-v2` — same_unique

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Ask a focused workflow question first, then choose prototype or specification from the answer.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
- **balanced** canonical counts: `{"C": 4}`; modal actions: ['Ask a focused workflow question first, then choose prototype or specification from the answer.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **B** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **A** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **D** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-verification-independence-v2` — changed_unique

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
- **fixed** canonical counts: `{"A": 2, "D": 1}`; modal actions: ['Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.']; exact stability: False.
  - r0: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
- **balanced** canonical counts: `{"A": 1, "D": 3}`; modal actions: ['Also run the independent parser and require agreement before completion.']; exact stability: False.
  - r0: provider **A** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

## Model: Luna

Modal relations: `{"changed_unique": 3, "incomparable_or_excluding_tie": 1, "same_unique": 4}`.
Displayed provider-letter counts in the balanced run: `{"A": 8, "B": 8, "C": 7, "D": 9}`.

### `preference-implementation-reuse-v2` — same_unique

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Prototype both boundaries against the contract tests and present measured complexity before selecting.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{}`.
- **balanced** canonical counts: `{"C": 4}`; modal actions: ['Prototype both boundaries against the contract tests and present measured complexity before selecting.']; exact stability: True.
  - r0: provider **D** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-interrupt-atomicity-v2` — changed_unique

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
- **fixed** canonical counts: `{"B": 2, "C": 1}`; modal actions: ['Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.']; exact stability: False.
  - r0: provider **C** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{}`.
- **balanced** canonical counts: `{"B": 1, "C": 3}`; modal actions: ['Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.']; exact stability: False.
  - r0: provider **D** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **B** -> canonical **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **A** -> canonical **C** — Report the safe boundary and eight-second remainder, then ask whether to pause or finish before switching.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-recovery-fallback-v1` — same_unique

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
- **fixed** canonical counts: `{"D": 3}`; modal actions: ['Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.']; exact stability: True.
  - r0: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-requirements-iteration-v1` — changed_unique

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
- **fixed** canonical counts: `{"C": 1, "D": 2}`; modal actions: ['Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.']; exact stability: False.
  - r0: provider **D** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Draft all requirements provisionally, implement slice one, and ask the user to review the remaining draft against it.; mapping `{}`.
- **balanced** canonical counts: `{"A": 1, "C": 3}`; modal actions: ['Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.']; exact stability: False.
  - r0: provider **D** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **C** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **B** -> canonical **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **C** -> canonical **A** — Resolve and document every slice and presentation decision before implementing any code.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.

### `preference-test-scope-v1` — same_unique

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
- **fixed** canonical counts: `{"B": 3}`; modal actions: ['Run the complete suite once and investigate any failure before declaring verification complete.']; exact stability: True.
  - r0: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
  - r1: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
  - r2: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{}`.
- **balanced** canonical counts: `{"B": 4}`; modal actions: ['Run the complete suite once and investigate any failure before declaring verification complete.']; exact stability: True.
  - r0: provider **A** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r1: provider **D** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r2: provider **C** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r3: provider **B** -> canonical **B** — Run the complete suite once and investigate any failure before declaring verification complete.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.

### `preference-uncertainty-presentation-v2` — incomparable_or_excluding_tie

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
- **fixed** canonical counts: `{"A": 1, "C": 1, "D": 1}`; modal actions: ['Lead with the recommendation and qualifier, then add a compact expandable risk analysis.', 'State the recommendation and offer either the concise qualifier or full risk table before elaborating.', 'Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.']; exact stability: False.
  - r0: provider **C** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{}`.
  - r1: provider **D** -> canonical **D** — Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.; mapping `{}`.
  - r2: provider **A** -> canonical **A** — Lead with the recommendation and qualifier, then add a compact expandable risk analysis.; mapping `{}`.
- **balanced** canonical counts: `{"A": 1, "C": 1, "D": 2}`; modal actions: ['Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.']; exact stability: False.
  - r0: provider **D** -> canonical **D** — Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **D** -> canonical **A** — Lead with the recommendation and qualifier, then add a compact expandable risk analysis.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **A** -> canonical **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **A** -> canonical **D** — Provide a structured likelihood-impact-fallback table with the 10–20% range and bounded consequences.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-vague-prototype-v2` — same_unique

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
- **fixed** canonical counts: `{"C": 3}`; modal actions: ['Ask a focused workflow question first, then choose prototype or specification from the answer.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
  - r1: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
  - r2: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{}`.
- **balanced** canonical counts: `{"C": 4}`; modal actions: ['Ask a focused workflow question first, then choose prototype or specification from the answer.']; exact stability: True.
  - r0: provider **C** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r1: provider **B** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r2: provider **A** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
  - r3: provider **D** -> canonical **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.

### `preference-verification-independence-v2` — changed_unique

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
- **fixed** canonical counts: `{"A": 2, "D": 1}`; modal actions: ['Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.']; exact stability: False.
  - r0: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
  - r1: provider **A** -> canonical **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; mapping `{}`.
  - r2: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{}`.
- **balanced** canonical counts: `{"D": 4}`; modal actions: ['Also run the independent parser and require agreement before completion.']; exact stability: True.
  - r0: provider **A** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "D", "B": "A", "C": "B", "D": "C"}`.
  - r1: provider **D** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "A", "B": "B", "C": "C", "D": "D"}`.
  - r2: provider **C** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "B", "B": "C", "C": "D", "D": "A"}`.
  - r3: provider **B** -> canonical **D** — Also run the independent parser and require agreement before completion.; mapping `{"A": "C", "B": "D", "C": "A", "D": "B"}`.
