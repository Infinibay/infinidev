# Repeated choice stability

Repeated isolated choices can reveal obvious instability but cannot estimate a population-level preference reliably without a justified sampling model. Every selected action remains primary evidence; modal choices and shares are compact summaries only.

Each repetition used a fresh conversation, no system message, no preference profile, and choice-only elicitation.

## Summary
- **Sol**: 1/2 probes exactly stable across repetitions.
- **Terra**: 1/2 probes exactly stable across repetitions.
- **Luna**: 1/2 probes exactly stable across repetitions.
- Cross-model modal agreement: 0/2 probes.

## Model: Sol

### `preference-requirements-formality-v1` — stable

Scenario: A bounded change has twelve explicit requirements, six non-goals, and five acceptance checks. All fit on one page and no regulated traceability format is required.
Observed counts: `{"B": 4}`.
- Repetition 0: **B** — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **B** — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **B** — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **B** — Maintain a compact grouped checklist linking each requirement to its implementation and evidence.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

### `preference-context-compactness-v1` — unstable

Scenario: A task has 9 decisive code excerpts totaling 7,000 tokens and 30 related files totaling 80,000 tokens. The index is current; related files may provide background but no known unresolved dependency.
Observed counts: `{"A": 1, "D": 3}`.
- Repetition 0: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **A** — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

## Model: Terra

### `preference-requirements-formality-v1` — stable

Scenario: A bounded change has twelve explicit requirements, six non-goals, and five acceptance checks. All fit on one page and no regulated traceability format is required.
Observed counts: `{"D": 4}`.
- Repetition 0: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

### `preference-context-compactness-v1` — unstable

Scenario: A task has 9 decisive code excerpts totaling 7,000 tokens and 30 related files totaling 80,000 tokens. The index is current; related files may provide background but no known unresolved dependency.
Observed counts: `{"A": 2, "D": 2}`.
- Repetition 0: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **A** — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **A** — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

## Model: Luna

### `preference-requirements-formality-v1` — stable

Scenario: A bounded change has twelve explicit requirements, six non-goals, and five acceptance checks. All fit on one page and no regulated traceability format is required.
Observed counts: `{"D": 4}`.
- Repetition 0: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **D** — Show matrix and checklist examples with maintenance cost, then let the user select the artifact.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

### `preference-context-compactness-v1` — unstable

Scenario: A task has 9 decisive code excerpts totaling 7,000 tokens and 30 related files totaling 80,000 tokens. The index is current; related files may provide background but no known unresolved dependency.
Observed counts: `{"A": 1, "D": 3}`.
- Repetition 0: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **A** — Provide the nine decisive excerpts with source anchors and retrieve more only if the model identifies a concrete gap.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Show estimated token cost and omission risk for compact and full context, then let the user choose.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
