# Repeated choice stability

Repeated isolated choices can reveal obvious instability but cannot estimate a population-level preference reliably without a justified sampling model. Every selected action remains primary evidence; modal choices and shares are compact summaries only.

Each repetition used a fresh conversation, no system message, no preference profile, and choice-only elicitation.

## Summary
- **Sol**: 0/2 probes exactly stable across repetitions.
- **Terra**: 1/2 probes exactly stable across repetitions.
- **Luna**: 2/2 probes exactly stable across repetitions.
- Cross-model modal agreement: 1/2 probes.

## Model: Sol

### `decision-risk-posture-v1` — unstable

Scenario: A reversible migration choice has a small known downside and a larger uncertain upside; all options respect safety and rollback requirements.
Observed counts: `{"A": 2, "C": 1, "D": 1}`.
- Repetition 0: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **A** — Choose the higher-upside option with a predefined rollback trigger.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **A** — Choose the higher-upside option with a predefined rollback trigger.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Run a staged pilot with explicit stop conditions before choosing broadly.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-planning-depth-v1` — unstable

Scenario: A four-file internal refactor has good regression tests, no public API change, and every edit is reversible. The dependency graph is visible, but implementation may reveal one or two local details.
Observed counts: `{"A": 3, "B": 1}`.
- Repetition 0: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **B** — Map every file, dependency, invariant, test, and rollback step before editing, then execute while revising any assumption invalidated by evidence.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

## Model: Terra

### `decision-risk-posture-v1` — unstable

Scenario: A reversible migration choice has a small known downside and a larger uncertain upside; all options respect safety and rollback requirements.
Observed counts: `{"C": 1, "D": 3}`.
- Repetition 0: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **C** — Run a staged pilot with explicit stop conditions before choosing broadly.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-planning-depth-v1` — stable

Scenario: A four-file internal refactor has good regression tests, no public API change, and every edit is reversible. The dependency graph is visible, but implementation may reveal one or two local details.
Observed counts: `{"A": 4}`.
- Repetition 0: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`

## Model: Luna

### `decision-risk-posture-v1` — stable

Scenario: A reversible migration choice has a small known downside and a larger uncertain upside; all options respect safety and rollback requirements.
Observed counts: `{"D": 4}`.
- Repetition 0: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 1: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 2: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
- Repetition 3: **D** — Quantify the bounded downside and ask the user to select risk appetite.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`

### `preference-planning-depth-v1` — stable

Scenario: A four-file internal refactor has good regression tests, no public API change, and every edit is reversible. The dependency graph is visible, but implementation may reveal one or two local details.
Observed counts: `{"A": 4}`.
- Repetition 0: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"C"}`; provider letter: `C`; canonical mapping: `{"A": "C", "B": "D", "C": "A", "D": "B"}`
- Repetition 1: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"B"}`; provider letter: `B`; canonical mapping: `{"A": "D", "B": "A", "C": "B", "D": "C"}`
- Repetition 2: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"A"}`; provider letter: `A`; canonical mapping: `{"A": "A", "B": "B", "C": "C", "D": "D"}`
- Repetition 3: **A** — Write a compact dependency-ordered plan with acceptance checks, then begin and refine only if new evidence changes it.; raw: `{"answer":"D"}`; provider letter: `D`; canonical mapping: `{"A": "B", "B": "C", "C": "D", "D": "A"}`
