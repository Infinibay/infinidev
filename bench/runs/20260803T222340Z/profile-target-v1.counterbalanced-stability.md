# Repeated choice stability

Three repeated choices can reveal obvious instability but cannot estimate a population-level preference reliably. Every selected action remains primary evidence; modal choices and shares are compact summaries only.

Each repetition used a fresh conversation, no system message, no preference profile, and choice-only elicitation.

## Summary
- **Sol**: 4/8 probes exactly stable across repetitions.
- **Terra**: 2/8 probes exactly stable across repetitions.
- **Luna**: 3/8 probes exactly stable across repetitions.
- Cross-model modal agreement: 1/8 probes.

## Model: Sol

### `preference-requirements-iteration-v1` — unstable

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
Observed counts: `{"B": 1, "C": 3}`.
- Repetition 0: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-vague-prototype-v2` — stable

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
Observed counts: `{"B": 4}`.
- Repetition 0: **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 1: **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 2: **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 3: **B** — Build one throwaway interactive prototype using existing components and collect concrete feedback.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`

### `preference-verification-independence-v2` — stable

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
Observed counts: `{"C": 4}`.
- Repetition 0: **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-implementation-reuse-v2` — unstable

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
Observed counts: `{"B": 1, "D": 3}`.
- Repetition 0: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Add the two supported hooks to the shared pipeline and follow its existing extension conventions.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-recovery-fallback-v1` — stable

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
Observed counts: `{"A": 4}`.
- Repetition 0: **A** — Switch immediately to direct source/text search and document the changed evidence channel.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **A** — Switch immediately to direct source/text search and document the changed evidence channel.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **A** — Switch immediately to direct source/text search and document the changed evidence channel.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **A** — Switch immediately to direct source/text search and document the changed evidence channel.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-test-scope-v1` — unstable

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
Observed counts: `{"A": 1, "C": 1, "D": 2}`.
- Repetition 0: **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 1: **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 2: **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 3: **A** — Accept the focused and impacted tests as sufficient, document their coverage, and stop.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`

### `preference-uncertainty-presentation-v2` — unstable

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
Observed counts: `{"A": 1, "B": 3}`.
- Repetition 0: **A** — Lead with the recommendation and qualifier, then add a compact expandable risk analysis.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 1: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 2: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 3: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`

### `preference-interrupt-atomicity-v2` — stable

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
Observed counts: `{"B": 4}`.
- Repetition 0: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

## Model: Terra

### `preference-requirements-iteration-v1` — unstable

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
Observed counts: `{"B": 2, "C": 2}`.
- Repetition 0: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-vague-prototype-v2` — stable

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
Observed counts: `{"C": 4}`.
- Repetition 0: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 1: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 2: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 3: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`

### `preference-verification-independence-v2` — unstable

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
Observed counts: `{"B": 2, "C": 2}`.
- Repetition 0: **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Complete with the primary proof and give the user the exact independent check as an optional command.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-implementation-reuse-v2` — unstable

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
Observed counts: `{"C": 2, "D": 2}`.
- Repetition 0: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Prototype both boundaries against the contract tests and present measured complexity before selecting.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-recovery-fallback-v1` — unstable

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
Observed counts: `{"B": 2, "D": 2}`.
- Repetition 0: **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Pause for the preferred semantic service and resume when its ranking behavior returns.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Pause for the preferred semantic service and resume when its ranking behavior returns.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-test-scope-v1` — unstable

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
Observed counts: `{"C": 2, "D": 2}`.
- Repetition 0: **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 1: **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 2: **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 3: **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`

### `preference-uncertainty-presentation-v2` — stable

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
Observed counts: `{"B": 4}`.
- Repetition 0: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 1: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 2: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 3: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`

### `preference-interrupt-atomicity-v2` — unstable

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
Observed counts: `{"B": 3, "D": 1}`.
- Repetition 0: **D** — Complete the remaining atomic rename, then respond with the finished local state clearly reported.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

## Model: Luna

### `preference-requirements-iteration-v1` — unstable

Scenario: A new internal workflow has three independently deployable slices. Product behavior is understood, but low-impact copy and ordering preferences will improve after users see the first slice.
Observed counts: `{"B": 2, "C": 2}`.
- Repetition 0: **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Fully specify and implement slice one, collect feedback, then elaborate the next slices from evidence.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Define hard contracts for all slices, leave reversible presentation choices open, and implement incrementally.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-vague-prototype-v2` — stable

Scenario: Production remains untouched while an internal search presentation is explored. A reversible prototype costs 30 minutes and a comparison specification costs 20; user preference is not yet known.
Observed counts: `{"C": 4}`.
- Repetition 0: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 1: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 2: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 3: **C** — Ask a focused workflow question first, then choose prototype or specification from the answer.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`

### `preference-verification-independence-v2` — unstable

Scenario: Source hash and deterministic regeneration prove an immutable artifact matches. A separate parser with independent code can corroborate that invariant at a cost of 25 minutes.
Observed counts: `{"A": 1, "B": 2, "D": 1}`.
- Repetition 0: **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **A** — Run the independent parser on a deterministic risk-weighted sample and retain mismatch handling.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Also run the independent parser and require agreement before completion.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **B** — Use the hash plus deterministic regeneration as the complete proof and retain both artifacts.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-implementation-reuse-v2` — stable

Scenario: The same public contract and tests can be satisfied by two hooks in an established export pipeline or by an isolated 90-line implementation. Reuse adds convention complexity; isolation duplicates a small amount of mechanics.
Observed counts: `{"D": 4}`.
- Repetition 0: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Implement the isolated 90-line component behind the common interface and keep duplication explicit.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-recovery-fallback-v1` — stable

Scenario: A preferred semantic index is temporarily unavailable. Direct text/source search is slower but current and can answer the same question; waiting may preserve the preferred ranking behavior.
Observed counts: `{"D": 4}`.
- Repetition 0: **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Explain latency, evidence equivalence, and consistency differences, then ask whether to fall back.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-test-scope-v1` — unstable

Scenario: A localized parser fix has focused regression tests and all 140 impacted-module tests passing. The complete 2,700-test suite takes 38 minutes and historically finds unrelated failures in 2% of runs; no release gate explicitly requires it.
Observed counts: `{"C": 2, "D": 2}`.
- Repetition 0: **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 1: **C** — Run a deterministic cross-subsystem subset chosen from dependency and co-change evidence, then stop if it passes.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 2: **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 3: **D** — Deliver the focused result now and run the full suite as a non-blocking follow-up, clearly separating provisional and final evidence.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`

### `preference-uncertainty-presentation-v2` — unstable

Scenario: A reversible recommendation remains preferred across plausible estimates. Residual failure is approximately 10–20%, bounded in impact, with a tested two-minute fallback.
Observed counts: `{"B": 3, "C": 1}`.
- Repetition 0: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 1: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 2: **C** — State the recommendation and offer either the concise qualifier or full risk table before elaborating.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 3: **B** — Give the recommendation with one concise confidence qualifier and name the tested fallback.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`

### `preference-interrupt-atomicity-v2` — unstable

Scenario: An eight-second local operation is between temporary write and atomic rename when the user interrupts. The temp state is safe and resumable, and completion has no external effect.
Observed counts: `{"A": 2, "B": 2}`.
- Repetition 0: **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **B** — Pause immediately at the safe temp-file boundary, respond to the user, and resume only if still relevant.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **A** — Acknowledge the interruption immediately, finish the safe atomic step, then switch to the new request.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
